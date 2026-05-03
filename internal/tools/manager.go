// Package tools wraps an MCP ToolManager and augments it with built-in tools
// (process execution, etc.) that live entirely within werkler.
//
// Responsibilities:
//   - Forward tool listing and dispatch to the wrapped manager.
//   - Register built-in tool definitions alongside MCP tools.
//   - Enforce path-based permissions before executing process tools.
//   - Forward optional OAuth lifecycle methods when the wrapped manager supports them.
package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unicode/utf8"

	"github.com/icedream/werkler/docs"
	"github.com/icedream/werkler/internal/agents"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/config"
	"github.com/icedream/werkler/internal/memorystore"
	"github.com/icedream/werkler/internal/process"
	"github.com/icedream/werkler/internal/skills"
	"github.com/icedream/werkler/internal/todostore"
)

// PathApprover checks path-level access approvals.
type PathApprover interface {
	IsPathReadApproved(path string) bool
	IsPathWriteApproved(path string) bool
}

// reviewerApprover is used by the rubber duck agentic loop.
// It approves all reads and command execution so the reviewer can grep and
// explore the codebase freely, while write-capable tools (file_write,
// file_edit, etc.) are simply excluded from the reviewer's tool set.
type reviewerApprover struct{}

func (reviewerApprover) IsPathReadApproved(_ string) bool  { return true }
func (reviewerApprover) IsPathWriteApproved(_ string) bool { return true }

// reviewModeKey is used as a context key to signal that the current call is
// executing on behalf of the rubber duck reviewer.
type reviewModeKey struct{}

// withReviewMode returns a child context that marks tool calls as reviewer-initiated.
func withReviewMode(ctx context.Context) context.Context {
	return context.WithValue(ctx, reviewModeKey{}, true)
}

// activeApprover returns the path approver in effect for ctx.
// When ctx carries the review-mode marker and a reviewer approver is configured,
// that approver is returned instead of the default one.
func (m *Manager) activeApprover(ctx context.Context) PathApprover {
	if _, ok := ctx.Value(reviewModeKey{}).(bool); ok {
		return reviewerApprover{}
	}
	return m.pathApprover
}

// UnapprovedPathsError is returned by CallTool when one or more paths accessed
// by a built-in tool have not been approved by the user.
// It implements chat.PathApprovalError so Session.CallTool propagates it as a
// structured error to the TUI.
type UnapprovedPathsError struct {
	Requests []chat.PathAccessRequest
}

func (e *UnapprovedPathsError) Error() string {
	paths := make([]string, len(e.Requests))
	for i, r := range e.Requests {
		paths[i] = r.Path
	}
	return fmt.Sprintf("path access not approved: %s", strings.Join(paths, ", "))
}

// AccessRequests implements chat.PathApprovalError.
func (e *UnapprovedPathsError) AccessRequests() []chat.PathAccessRequest {
	return e.Requests
}

// oauthManager mirrors the optional interface on mcp.Manager that Session delegates.
type oauthManager interface {
	HasPendingOAuth() bool
	PendingOAuthNames() []string
	ConnectPendingOAuth(ctx context.Context, display func(serverName, authURL string)) error
}

// mcpConnector is the minimal interface the connect_server builtin needs from
// the MCP manager. Keeping it narrow avoids an import cycle and makes testing easier.
type mcpConnector interface {
	ConfiguredServers() []config.MCPServerConfig
	IsConnected(name string) bool
	ConnectByName(ctx context.Context, name string) (alreadyConnected bool, err error)
	ToolNamesForServer(ctx context.Context, name string) ([]string, error)
}

// UserAsker is implemented by interactive callers (e.g. the TUI) to suspend a
// tool goroutine and wait for user input. It is called from a tool execution
// goroutine and blocks until the user responds or ctx is cancelled.
type UserAsker func(ctx context.Context, question string, choices []string, recommendedChoice string, allowFreeform bool) (string, error)

// PlanConfirmer is implemented by interactive callers to present a plan-confirmation
// dialog to the user. The TUI owns all mode-switching and autopilot logic;
// the callback blocks until the user responds or ctx is cancelled, and returns
// one of: "approved", "approved_with_autopilot", or "rejected: <reason>".
// When nil (non-interactive), the tool auto-approves without a dialog.
type PlanConfirmer func(ctx context.Context, summary string) (string, error)

// subagentDef describes a subagent that the AI can invoke as a tool.
// Each registered subagent becomes one built-in tool in the manager.
type subagentDef struct {
	name         string          // tool name (e.g. "rubber_duck_review")
	description  string          // shown to the AI in the tool list
	systemPrompt string          // system prompt for the subagent's own agentic loop
	allowedTools map[string]bool // built-in tool names the subagent may call
	maxSteps     int             // max iterations before giving up
	completer    ai.Completer    // AI client for this subagent
	label        string          // display label used in descriptions (e.g. "openai/gpt-4o")
}

// Manager wraps a chat.ToolManager and adds built-in tools.
type Manager struct {
	wrapped       chat.ToolManager
	processes     *process.Manager
	pathApprover  PathApprover
	builtins      []builtin
	userAsker     UserAsker
	planConfirmer PlanConfirmer
	subagents     []subagentDef
	mcpMgr        mcpConnector
	skills        []skills.Skill
	agents        []agents.Agent
	todoStore     *todostore.Store
	memoryStore   *memorystore.MemoryStore
	taskTitle     func(string) // optional; called when the AI sets a task title via task_start
	agentActivate func(string) // optional; called when use_agent requests activation
	activeCallID  atomic.Value // stores string; set/cleared by doCallTool goroutine
	// allowedTools is the active agent's tool allowlist. nil = unrestricted;
	// non-nil = only these tools (plus infra tools) are visible and callable.
	allowedTools []string
	filterMu     sync.RWMutex
}

type builtin struct {
	def             ai.ToolDefinition
	handle          func(ctx context.Context, args map[string]any) (string, error)
	handleWithParts func(ctx context.Context, args map[string]any) (string, []ai.ImagePart, error) // optional; used when a tool returns image data
}

// OutputNotification is forwarded from the process manager to the caller
// (typically the TUI) so live output can be displayed.
type OutputNotification = process.OutputNotification

// New creates a Manager. pathApprover may be nil (disables path enforcement).
// notify is called whenever a managed process produces output; may be nil.
func New(wrapped chat.ToolManager, pathApprover PathApprover, notify OutputNotification) *Manager {
	pm := process.New(notify)
	m := &Manager{
		wrapped:      wrapped,
		processes:    pm,
		pathApprover: pathApprover,
	}
	m.builtins = m.makeBuiltins()
	return m
}

// SetOutputNotify sets the live output notification callback. Used by the TUI
// to receive process output for streaming into the viewport.
func (m *Manager) SetOutputNotify(fn OutputNotification) {
	m.processes.SetNotify(fn)
}

// SetSkills provides the list of skills available via the use_skill built-in.
// Rebuilds the built-in tool list. Must be called at setup time only (not concurrency-safe).
func (m *Manager) SetSkills(s []skills.Skill) {
	m.skills = s
	m.builtins = m.makeBuiltins()
}

// SetAgents provides the list of custom agents available via the use_agent
// built-in. When non-empty, registers the use_agent tool and adds an agent
// listing to the system prompt section. Must be called at setup time only
// (not concurrency-safe).
func (m *Manager) SetAgents(a []agents.Agent) {
	m.agents = a
	m.builtins = m.makeBuiltins()
}

// SetTodoStore wires an in-session todo store so the AI can manage todos.
// Must be called at setup time only (not concurrency-safe).
func (m *Manager) SetTodoStore(s *todostore.Store) {
	m.todoStore = s
	m.builtins = m.makeBuiltins()
}

// SetMemoryStore wires the per-project memory store so the AI can read and
// write persistent notes about the project. Must be called at setup time only
// (not concurrency-safe).
func (m *Manager) SetMemoryStore(s *memorystore.MemoryStore) {
	m.memoryStore = s
	m.builtins = m.makeBuiltins()
}

// SetActiveCallID records the ID of the tool call currently being executed.
// Called by the doCallTool goroutine in the TUI; safe for concurrent use.
func (m *Manager) SetActiveCallID(id string) { m.activeCallID.Store(id) }

// ActiveCallID returns the ID of the tool call currently being executed, or
// empty string if no call is in progress.
func (m *Manager) ActiveCallID() string {
	if v := m.activeCallID.Load(); v != nil {
		return v.(string)
	}
	return ""
}

// Session (which wraps this Manager) to serve as its own path approver without
// a circular dependency at construction time.
func (m *Manager) SetPathApprover(pa PathApprover) {
	m.pathApprover = pa
}

// SetUserAsker provides the callback used by the ask_user built-in to request
// user input interactively. When nil (the default), ask_user returns a static
// "not available in non-interactive mode" message.
func (m *Manager) SetUserAsker(fn UserAsker) { m.userAsker = fn }

// SetPlanConfirmer provides the callback used by the confirm_plan built-in to
// present a plan-confirmation dialog and handle mode/autopilot switching.
// When nil (the default), confirm_plan auto-approves without a dialog.
func (m *Manager) SetPlanConfirmer(fn PlanConfirmer) { m.planConfirmer = fn }

// SetTaskTitleNotify registers a callback invoked whenever the AI calls
// task_start to report what it is currently working on. Pass nil to disable.
func (m *Manager) SetTaskTitleNotify(fn func(string)) { m.taskTitle = fn }

// SetAgentActivateNotify sets an optional callback that is invoked when the AI
// calls use_agent. The TUI uses this to trigger session-side agent activation.
func (m *Manager) SetAgentActivateNotify(fn func(string)) { m.agentActivate = fn }

// SetReviewer provides an optional secondary AI model used by subagents
// (rubber_duck_review and doc_review). Registers both subagents when c is
// non-nil; clears them when nil. Must be called before the manager is in
// use (setup time only — not concurrency-safe).
func (m *Manager) SetReviewer(c ai.Completer, label string) {
	if c == nil {
		m.subagents = nil
	} else {
		m.subagents = []subagentDef{
			makeRubberDuckSubagent(c, label),
			makeDocReviewSubagent(c, label),
		}
	}
	m.builtins = m.makeBuiltins()
}

// IsSubagentTool reports whether name is a tool registered as a subagent.
func (m *Manager) IsSubagentTool(name string) bool {
	for _, sa := range m.subagents {
		if sa.name == name {
			return true
		}
	}
	return false
}

// SetMCPManager wires in the MCP manager so the connect_server builtin is
// registered. Must be called before the tool manager is in use.
func (m *Manager) SetMCPManager(mgr mcpConnector) {
	m.mcpMgr = mgr
	m.builtins = m.makeBuiltins()
}

// infraTools is the set of tool names that are always available regardless of
// any active agent tool allowlist. These are control-flow and infrastructure
// tools that must never be filtered out.
var infraTools = map[string]bool{
	"use_agent":      true,
	"use_skill":      true,
	"ask_user":       true,
	"task_start":     true,
	"task_complete":  true,
	"connect_server": true,
}

// IsInfraToolName reports whether name is one of the infrastructure tools that
// are always available regardless of any active agent tool allowlist.
func IsInfraToolName(name string) bool {
	return infraTools[name]
}

// SetToolFilter restricts the tools visible to the AI and callable in this
// session to the provided allowlist. nil removes any active restriction
// (all tools become available). An empty (non-nil) slice means no tools other
// than the permanent infra tools. Infrastructure tools are always available
// regardless of the filter.
//
// Tool names support a "server.*" wildcard to allow all tools from an MCP server.
//
// Safe to call concurrently with Tools() and CallTool().
func (m *Manager) SetToolFilter(names []string) {
	m.filterMu.Lock()
	m.allowedTools = names
	m.filterMu.Unlock()
}

// isAllowed reports whether the named tool should be visible / callable under
// the current filter. Infrastructure tools always return true.
func (m *Manager) isAllowed(name string) bool {
	if infraTools[name] {
		return true
	}
	m.filterMu.RLock()
	allowed := m.allowedTools
	m.filterMu.RUnlock()
	if allowed == nil {
		return true
	}
	for _, a := range allowed {
		if a == name {
			return true
		}
		// Wildcard: "server.*" matches any tool whose name starts with "server."
		if strings.HasSuffix(a, ".*") {
			prefix := strings.TrimSuffix(a, "*")
			if strings.HasPrefix(name, prefix) {
				return true
			}
		}
	}
	return false
}

// HasPendingOAuth forwards to the wrapped manager if it supports OAuth.
func (m *Manager) HasPendingOAuth() bool {
	if o, ok := m.wrapped.(oauthManager); ok {
		return o.HasPendingOAuth()
	}
	return false
}

// PendingOAuthNames forwards to the wrapped manager if it supports OAuth.
func (m *Manager) PendingOAuthNames() []string {
	if o, ok := m.wrapped.(oauthManager); ok {
		return o.PendingOAuthNames()
	}
	return nil
}

// ConnectPendingOAuth forwards to the wrapped manager if it supports OAuth.
func (m *Manager) ConnectPendingOAuth(ctx context.Context, display func(serverName, authURL string)) error {
	if o, ok := m.wrapped.(oauthManager); ok {
		return o.ConnectPendingOAuth(ctx, display)
	}
	return nil
}

// Tools returns the union of MCP tools and built-in tool definitions,
// filtered by the active agent tool allowlist (if any).
func (m *Manager) Tools(ctx context.Context) ([]ai.ToolDefinition, error) {
	mcpTools, err := m.wrapped.Tools(ctx)
	if err != nil {
		return nil, err
	}
	defs := make([]ai.ToolDefinition, 0, len(mcpTools)+len(m.builtins))
	for _, t := range mcpTools {
		if m.isAllowed(t.Name) {
			defs = append(defs, t)
		}
	}
	for _, b := range m.builtins {
		if m.isAllowed(b.def.Name) {
			defs = append(defs, b.def)
		}
	}
	return defs, nil
}

// CallTool dispatches to a built-in or the wrapped MCP manager.
// For process-starting tools, path permissions are checked first and
// *UnapprovedPathsError is returned if any paths lack approval.
func (m *Manager) CallTool(ctx context.Context, name string, args map[string]any) (string, error) {
	result, _, err := m.CallToolWithParts(ctx, name, args)
	return result, err
}

// CallToolWithParts is like CallTool but also returns any image parts the tool
// produced (e.g. read_image). Parts is nil for all tools that don't produce images.
func (m *Manager) CallToolWithParts(ctx context.Context, name string, args map[string]any) (string, []ai.ImagePart, error) {
	if !m.isAllowed(name) {
		return fmt.Sprintf("tool %q is not available for the active agent", name), nil, nil
	}
	for _, b := range m.builtins {
		if b.def.Name == name {
			if b.handleWithParts != nil {
				return b.handleWithParts(ctx, args)
			}
			result, err := b.handle(ctx, args)
			return result, nil, err
		}
	}
	result, err := m.wrapped.CallTool(ctx, name, args)
	return result, nil, err
}

// --- Path extraction ---

// knownShells is the set of command basenames that accept inline scripts via -c.
var knownShells = map[string]bool{
	"bash": true, "sh": true, "zsh": true, "fish": true, "dash": true, "ksh": true,
}

// pathHeuristic matches strings that look like filesystem paths.
var pathHeuristic = regexp.MustCompile(`(?:^|[\s"'=])(\./[^\s"']+|/[^\s"']+|~/[^\s"']+)`)

// ExtractPaths returns all filesystem paths referenced by the given process
// invocation. It:
//  1. Resolves command to an absolute path via exec.LookPath when no slash is present.
//  2. Resolves the cwd itself (if non-empty).
//  3. Resolves relative args against cwd.
//  4. Scans all args for embedded path-like strings via heuristic regex.
//  5. For known shell interpreters with -c, additionally scans the script text.
//
// Duplicates are removed and paths are returned in discovery order.
func ExtractPaths(command string, args []string, cwd string) []string {
	var paths []string
	seen := make(map[string]bool)

	add := func(p string) {
		if p == "" {
			return
		}
		if seen[p] {
			return
		}
		seen[p] = true
		paths = append(paths, p)
	}

	// 1. Resolve command.
	if strings.ContainsRune(command, '/') {
		add(command)
	} else if resolved, err := exec.LookPath(command); err == nil {
		add(resolved)
	} else {
		add(command) // unresolvable — include as-is so user can still approve
	}

	// Note: cwd is intentionally NOT added to the approval list.
	// It is the process's starting directory, not a path being read/written.
	// It is still used below to resolve relative argument paths.

	// 3 & 4. Each arg: resolve if relative, scan for embedded paths.
	for _, arg := range args {
		// Resolve relative paths against cwd.
		if isPathLike(arg) && !containsShellVar(arg) {
			add(resolvePath(arg, cwd))
		}
		// Heuristic scan for embedded paths (covers cases like "-I/usr/include").
		for _, m := range pathHeuristic.FindAllStringSubmatch(arg, -1) {
			if len(m) >= 2 {
				p := strings.Trim(m[1], `"'`)
				if !containsEllipsis(p) && !containsShellVar(p) {
					add(resolvePath(p, cwd))
				}
			}
		}
	}

	// 5. If command is a shell and -c is in args, scan the script string.
	base := filepath.Base(command)
	if knownShells[base] {
		for i, arg := range args {
			if arg == "-c" && i+1 < len(args) {
				script := args[i+1]
				// Scan embedded path references.
				for _, m := range pathHeuristic.FindAllStringSubmatch(script, -1) {
					if len(m) >= 2 {
						p := strings.Trim(m[1], `"'`)
						if !containsEllipsis(p) && !containsShellVar(p) {
							add(resolvePath(p, cwd))
						}
					}
				}
				// Extract individual sub-command executables from the script so each
				// command is also subject to approval, not just the shell itself.
				for _, sub := range extractShellSubCommands(script, cwd) {
					add(sub)
				}
			}
		}
	}

	return paths
}

// shellSplitRe splits a shell script on operators: ; | && || & newline.
// This is intentionally a simple heuristic — it does not handle quoting or
// subshells, but covers the common cases generated by AI models.
var shellSplitRe = regexp.MustCompile(`[;|&\n]+`)

// extractShellSubCommands extracts the first token (command name) of each
// simple command found in a shell script string. Commands that are built-in
// shell keywords (if, then, else, fi, do, done, while, for, case, esac,
// function, {, }) are skipped. Returns the resolved absolute paths of
// each distinct executable found.
func extractShellSubCommands(script, cwd string) []string {
	var cmds []string
	seen := make(map[string]bool)
	// Split on common command separators.
	parts := shellSplitRe.Split(script, -1)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		// Skip leading shell redirections/variable assignments before the command.
		fields := strings.Fields(part)
		cmdName := ""
		for _, f := range fields {
			// Skip variable assignments (VAR=value) and redirections (>, <, >>).
			if strings.ContainsRune(f, '=') || strings.HasPrefix(f, ">") || strings.HasPrefix(f, "<") {
				continue
			}
			cmdName = f
			break
		}
		if cmdName == "" || shellKeywords[cmdName] || containsShellVar(cmdName) {
			continue
		}
		// Strip surrounding quotes.
		cmdName = strings.Trim(cmdName, `"'`)
		if seen[cmdName] {
			continue
		}
		seen[cmdName] = true
		if strings.ContainsRune(cmdName, '/') {
			cmds = append(cmds, cmdName)
		} else if resolved, err := exec.LookPath(cmdName); err == nil {
			cmds = append(cmds, resolved)
		} else {
			cmds = append(cmds, cmdName)
		}
	}
	return cmds
}

// shellKeywords is the set of tokens that are shell built-in keywords or
// control-flow words; they are not independent executables and should not be
// added to the path approval list.
var shellKeywords = map[string]bool{
	"if": true, "then": true, "else": true, "elif": true, "fi": true,
	"for": true, "while": true, "until": true, "do": true, "done": true,
	"case": true, "esac": true, "in": true,
	"function": true, "{": true, "}": true, "!": true,
	"[": true, "[[": true, "]": true, "]]": true,
	"echo": true, "true": true, "false": true,
	"export": true, "local": true, "readonly": true,
	"return": true, "exit": true, "break": true, "continue": true,
	"set": true, "unset": true, "shift": true,
	"source": true, ".": true,
}

// Excludes Go package wildcards (e.g. ./..., ./foo/...) which start with ./ but
// are not real filesystem paths.
func isPathLike(s string) bool {
	if containsEllipsis(s) {
		return false
	}
	return strings.HasPrefix(s, "/") ||
		strings.HasPrefix(s, "./") ||
		strings.HasPrefix(s, "../") ||
		strings.HasPrefix(s, "~/")
}

// containsShellVar reports whether s contains shell variable syntax ($, {, }).
// Such strings are unexpanded variable references, not literal filesystem paths.
func containsShellVar(s string) bool {
	return strings.ContainsAny(s, "${}\\")
}

// containsEllipsis reports whether any path segment of s is "..." (Go wildcard).
func containsEllipsis(s string) bool {
	for _, seg := range strings.Split(s, "/") {
		if seg == "..." {
			return true
		}
	}
	return false
}

// resolvePath expands ~ and resolves a relative path against baseDir.
func resolvePath(p, baseDir string) string {
	if strings.HasPrefix(p, "~/") {
		u, _ := user.Current()
		home := ""
		if u != nil {
			home = u.HomeDir
		}
		p = filepath.Join(home, p[2:])
	}
	if !filepath.IsAbs(p) && baseDir != "" {
		p = filepath.Join(baseDir, p)
	}
	return filepath.Clean(p)
}

// canonicalizePath returns the canonical absolute path for permission checks.
// For existing paths it resolves symlinks; for non-existent paths (new files)
// it resolves the parent directory's symlinks and appends the basename.
func canonicalizePath(p string) string {
	p = resolvePath(p, "")
	if !filepath.IsAbs(p) {
		if cwd, err := os.Getwd(); err == nil {
			p = filepath.Join(cwd, p)
		}
	}
	if resolved, err := filepath.EvalSymlinks(p); err == nil {
		return resolved
	}
	// Path does not exist yet — resolve parent and join.
	if resolvedDir, err := filepath.EvalSymlinks(filepath.Dir(p)); err == nil {
		return filepath.Join(resolvedDir, filepath.Base(p))
	}
	return filepath.Clean(p)
}

// checkWritePaths returns unapproved paths from ExtractPaths.
// The command binary (first path) requires execute-level approval; all other
// paths (file arguments) require write-level approval.
// The active approver is resolved from ctx (reviewer vs. normal).
func (m *Manager) checkWritePaths(ctx context.Context, command string, args []string, cwd string) []chat.PathAccessRequest {
	approver := m.activeApprover(ctx)
	if approver == nil {
		return nil
	}
	paths := ExtractPaths(command, args, cwd)
	var unapproved []chat.PathAccessRequest
	for i, p := range paths {
		if !approver.IsPathWriteApproved(p) {
			req := chat.PathAccessRequest{Path: p}
			if i == 0 {
				req.Execute = true
			} else {
				req.Write = true
			}
			unapproved = append(unapproved, req)
		}
	}
	return unapproved
}

// checkSingleRead returns a non-nil error if path lacks read approval.
func (m *Manager) checkSingleRead(ctx context.Context, path string) *UnapprovedPathsError {
	approver := m.activeApprover(ctx)
	if approver == nil {
		return nil
	}
	if !approver.IsPathReadApproved(path) {
		return &UnapprovedPathsError{Requests: []chat.PathAccessRequest{{Path: path, Write: false}}}
	}
	return nil
}

// checkSingleWrite returns a non-nil error if path lacks write approval.
func (m *Manager) checkSingleWrite(ctx context.Context, path string) *UnapprovedPathsError {
	approver := m.activeApprover(ctx)
	if approver == nil {
		return nil
	}
	if !approver.IsPathWriteApproved(path) {
		return &UnapprovedPathsError{Requests: []chat.PathAccessRequest{{Path: path, Write: true}}}
	}
	return nil
}

// --- Built-in tool definitions ---

func (m *Manager) makeBuiltins() []builtin {
	builtins := []builtin{
		{
			def: ai.ToolDefinition{
				Name: "process_start",
				Description: `Start a subprocess. Returns a handle for subsequent interaction.
Use pty=true for interactive programs (editors, REPLs, password prompts) — output will contain ANSI codes that are stripped before being returned to you.
Use pty=false (default) for non-interactive commands (builds, scripts) — output is clean plain text.

IMPORTANT: "command" is the executable only; "args" are the arguments WITHOUT the command name.
Example — to run "git status --short": command="git", args=["status", "--short"].
Do NOT put "git" (or any command name) in args.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"command":         map[string]any{"type": "string", "description": "Absolute path or PATH-resolvable command name"},
						"args":            map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Command arguments only — do NOT include the command name as the first element (unlike argv[0])"},
						"title":           map[string]any{"type": "string", "description": "REQUIRED. Short human-readable phrase describing the purpose of this command (e.g. \"Build project\", \"Run tests\", \"Search for TODO comments\"). Shown to the user in the approval dialog and chat log."},
						"cwd":             map[string]any{"type": "string", "description": "Working directory; empty = inherit werkler's cwd"},
						"pty":             map[string]any{"type": "boolean", "description": "Allocate a PTY (pseudo-terminal); required for interactive programs"},
						"env":             map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}, "description": "Extra environment variables merged on top of the current environment"},
						"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait for initial output before returning (0 = don't wait, default 5)"},
					},
					"required": []string{"command", "title"},
				},
			},
			handle: m.handleProcessStart,
		},
		{
			def: ai.ToolDefinition{
				Name:        "process_send",
				Description: "Send text to a running process's stdin / PTY.",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"handle":          map[string]any{"type": "string", "description": "Process handle from process_start"},
						"text":            map[string]any{"type": "string", "description": "Text to write"},
						"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait for new output after sending (default 2)"},
					},
					"required": []string{"handle", "text"},
				},
			},
			handle: m.handleProcessSend,
		},
		{
			def: ai.ToolDefinition{
				Name: "process_send_key",
				Description: `Send a named key to a running process.
Available keys: enter, tab, escape, backspace, delete, home, end, page_up, page_down,
up, down, left, right, ctrl+a through ctrl+z, f1-f12.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"handle":          map[string]any{"type": "string", "description": "Process handle from process_start"},
						"key":             map[string]any{"type": "string", "description": "Key name, e.g. \"enter\", \"ctrl+c\", \"up\""},
						"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait for new output after sending (default 2)"},
					},
					"required": []string{"handle", "key"},
				},
			},
			handle: m.handleProcessSendKey,
		},
		{
			def: ai.ToolDefinition{
				Name:        "process_read",
				Description: "Read output from a running process since the last read.",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"handle":          map[string]any{"type": "string", "description": "Process handle from process_start"},
						"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait for new output (default 5)"},
					},
					"required": []string{"handle"},
				},
			},
			handle: m.handleProcessRead,
		},
		{
			def: ai.ToolDefinition{
				Name:        "process_stop",
				Description: "Stop a running process and collect its final output.",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"handle": map[string]any{"type": "string", "description": "Process handle from process_start"},
						"force":  map[string]any{"type": "boolean", "description": "Use SIGKILL instead of SIGTERM (default false)"},
					},
					"required": []string{"handle"},
				},
			},
			handle: m.handleProcessStop,
		},
		{
			def: ai.ToolDefinition{
				Name: "run_command",
				Description: `Run a command synchronously and return its output in a single call. No polling required.
Use this for one-shot non-interactive commands (builds, scripts, grep, CLI tools, etc.).
Do NOT use for interactive programs, programs that require a TTY, or long-running background processes -- use process_start for those.

IMPORTANT: "command" is the executable only; "args" are the arguments WITHOUT the command name.
Example -- to run "git status --short": command="git", args=["status", "--short"].
Do NOT put "git" (or any command name) in args.

SHELL MODE: Any time your command uses pipes (|), redirects (>, >>), logical operators (&&, ||),
subshells ($(...)), globs, or any other shell syntax, you MUST set shell=true and put the entire
command string in "command" with no "args".
Examples requiring shell=true: "ps aux | grep nginx", "cat /proc/loadavg", "free -m && uptime",
"ls *.go | wc -l", "top -bn1 | head -20".
Examples NOT requiring shell=true: "git", "grep", "uptime", "free", "ps" (with args instead).`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"command":         map[string]any{"type": "string", "description": "Absolute path or PATH-resolvable command name. When shell=true, this is the full shell command string (e.g. \"ps aux | grep nginx\")."},
						"args":            map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Command arguments only -- do NOT include the command name. Must not be provided when shell=true."},
						"shell":           map[string]any{"type": "boolean", "description": "If true, run via bash -c. REQUIRED for any command containing |, &&, ||, >, >>, $(...), or other shell syntax."},
						"cwd":             map[string]any{"type": "string", "description": "Working directory; empty = inherit werkler's cwd"},
						"env":             map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}, "description": "Extra environment variables merged on top of the current environment"},
						"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait before killing the process (default 30, max 600)"},
						"title":           map[string]any{"type": "string", "description": "REQUIRED. Short human-readable phrase describing what is being run. Shown in the approval dialog."},
					},
					"required": []string{"command", "title"},
				},
			},
			handle: m.handleRunCommand,
		},
		// --- File tools ---
		{
			def: ai.ToolDefinition{
				Name: "file_read_multi",
				Description: `Read text file regions, multiple in one call possible. Returns each region labeled with a header line.
Each region may specify start_line and end_line (1-indexed, inclusive); omit both to read the full file.
Partial failures are reported inline — other regions still return their content.
Total output is capped at 1 KiB across all regions.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"regions": map[string]any{
							"type":        "array",
							"description": "List of regions to read",
							"items": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"path":       map[string]any{"type": "string", "description": "Absolute or ~ path to the file"},
									"start_line": map[string]any{"type": "number", "description": "First line to return (1-indexed, default 1)"},
									"end_line":   map[string]any{"type": "number", "description": "Last line to return (1-indexed, default: end of file)"},
								},
								"required": []string{"path"},
							},
						},
					},
					"required": []string{"regions"},
				},
			},
			handle: m.handleFileReadMulti,
		},
		{
			def: ai.ToolDefinition{
				Name:        "read_image",
				Description: "Load a local image file so you can see its visual content. ONLY use for the following supported formats: PNG, JPEG, GIF, WebP.",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{"type": "string", "description": "Absolute or ~ path to the image file"},
					},
					"required": []string{"path"},
				},
			},
			handleWithParts: m.handleReadImage,
		},
		{
			def: ai.ToolDefinition{
				Name:        "file_list",
				Description: "List the contents of a directory. Returns a JSON array of {name, type, size} objects.",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{"type": "string", "description": "Absolute or ~ path to the directory"},
					},
					"required": []string{"path"},
				},
			},
			handle: m.handleFileList,
		},
		{
			def: ai.ToolDefinition{
				Name: "file_write",
				Description: `Create a new file or overwrite an existing file with the given text content.
This is the correct tool to use whenever you need to write a file whole.
To create a new file including its parent directories, set create_parents to true.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path":           map[string]any{"type": "string", "description": "Absolute or ~ path to the file"},
						"content":        map[string]any{"type": "string", "description": "File contents (UTF-8)"},
						"create_parents": map[string]any{"type": "boolean", "description": "Create parent directories if they don't exist (default false)"},
					},
					"required": []string{"path", "content"},
				},
			},
			handle: m.handleFileWrite,
		},
		{
			def: ai.ToolDefinition{
				Name: "file_edit",
				Description: `Replace one or more exact-string occurrences in a file.

Single-hunk form (most common):
  path, old_str, new_str — replace one occurrence of old_str with new_str.

Multi-hunk form (use when making several changes to the same file in one call):
  path, edits: [{old_str, new_str}, …] — apply each replacement in order.
  All hunks are validated before any writes; the call fails atomically if any
  old_str is not found or matches more than once.

In both forms: the match must be exact (including whitespace). Returns an error
if old_str appears zero times (not found) or more than once (ambiguous — include
more surrounding context).
On success, returns the line number(s) where replacements were made.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path":    map[string]any{"type": "string", "description": "Absolute or ~ path to the file"},
						"old_str": map[string]any{"type": "string", "description": "Exact text to find and replace (single-hunk form)"},
						"new_str": map[string]any{"type": "string", "description": "Replacement text (single-hunk form)"},
						"edits": map[string]any{
							"type":        "array",
							"description": "List of {old_str, new_str} pairs applied in order (multi-hunk form). Mutually exclusive with top-level old_str/new_str.",
							"items": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"old_str": map[string]any{"type": "string"},
									"new_str": map[string]any{"type": "string"},
								},
								"required": []string{"old_str", "new_str"},
							},
						},
					},
					"required": []string{"path"},
				},
			},
			handle: m.handleFileEdit,
		},
		{
			def: ai.ToolDefinition{
				Name:        "file_delete",
				Description: "Delete a file (not a directory).",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path": map[string]any{"type": "string", "description": "Absolute or ~ path to the file"},
					},
					"required": []string{"path"},
				},
			},
			handle: m.handleFileDelete,
		},
		// --- Interaction tools ---
		{
			def: ai.ToolDefinition{
				Name: "ask_user",
				Description: `Ask the user a direct question and wait for their answer.
Use this when the task requires a decision or information that only the user can provide.

REQUIRED: When there are 2–4 known valid answers, you MUST put them in the "choices" array — do NOT embed numbered options in the question string itself.
WRONG: question="What should I do?\n\n1. Deploy now\n2. Review first", choices=[]
RIGHT: question="What should I do?", choices=["Deploy now", "Review first"]

Set allow_freeform to false to restrict the user strictly to those choices.
Set recommended_choice to highlight a suggested option.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"question": map[string]any{
							"type":        "string",
							"description": "The question to present to the user. Must be a plain sentence ending in '?'. Do NOT include numbered or bulleted options — put those in the 'choices' array.",
						},
						"choices": map[string]any{
							"type":        "array",
							"items":       map[string]any{"type": "string"},
							"description": "Predefined answer choices (e.g. [\"Yes\", \"No\"] or [\"Option A\", \"Option B\", \"Option C\"]). Always populate this when there are 2–4 known valid answers.",
						},
						"recommended_choice": map[string]any{
							"type":        "string",
							"description": "The choice you recommend; must exactly match one of the choices",
						},
						"allow_freeform": map[string]any{
							"type":        "boolean",
							"description": "Whether to also accept a custom typed answer (default: true)",
						},
					},
					"required": []string{"question"},
				},
			},
			handle: m.handleAskUser,
		},
	}

	for i := range m.subagents {
		sa := &m.subagents[i]
		builtins = append(builtins, builtin{
			def: ai.ToolDefinition{
				Name:        sa.name,
				Description: sa.description,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"context": map[string]any{
							"type":        "string",
							"description": "The content to submit for review",
						},
						"focus": map[string]any{
							"type":        "string",
							"description": `Optional: specific aspects to focus on (e.g. "clarity", "accuracy")`,
						},
					},
					"required": []string{"context"},
				},
			},
			handle: func(ctx context.Context, args map[string]any) (string, error) {
				return m.runSubagent(ctx, sa, args)
			},
		})
	}

	if len(m.skills) > 0 {
		names := make([]any, len(m.skills))
		for i, s := range m.skills {
			names[i] = s.Name
		}
		builtins = append(builtins, builtin{
			def: ai.ToolDefinition{
				Name:        "use_skill",
				Description: "Load the instructions for a named skill into the conversation.",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{
							"type":        "string",
							"enum":        names,
							"description": "Name of the skill to load",
						},
					},
					"required": []string{"name"},
				},
			},
			handle: m.handleUseSkill,
		})
	}

	if len(m.agents) > 0 {
		names := make([]any, len(m.agents))
		for i, a := range m.agents {
			names[i] = a.Name
		}
		builtins = append(builtins, builtin{
			def: ai.ToolDefinition{
				Name:        "use_agent",
				Description: "Activate a named agent persona for the current session.",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{
							"type":        "string",
							"enum":        names,
							"description": "Agent name to activate",
						},
					},
					"required": []string{"name"},
				},
			},
			handle: m.handleUseAgent,
		})
	}

	if m.todoStore != nil {
		builtins = append(builtins,
			builtin{
				def: ai.ToolDefinition{
					Name: "todo_add",
					Description: `Add a single todo item to the session task list.
Always supply a short kebab-case id (e.g. "write-readme", "fix-login-bug") so you can
reference the todo later. Use proactively at the start of multi-step tasks.
To add several todos at once, use todo_add_many instead.
IMPORTANT: Duplicate titles are not allowed. If a todo with the same title already exists
you will receive the existing item back — do NOT call todo_add again with the same title.
IMPORTANT: Duplicate IDs are not allowed. Choose a unique id; if the id already exists
you will receive the existing item back.
Only rephrase/re-id if this is genuinely a distinct task.`,
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"id":          map[string]any{"type": "string", "description": "Short kebab-case identifier (e.g. write-readme). Must be unique in this session."},
							"title":       map[string]any{"type": "string", "description": "Short one-line title"},
							"description": map[string]any{"type": "string", "description": "Optional detail or acceptance criteria"},
						},
						"required": []string{"title"},
					},
				},
				handle: m.handleTodoAdd,
			},
			builtin{
				def: ai.ToolDefinition{
					Name: "todo_add_many",
					Description: `Add multiple todo items to the session task list in a single call.
Prefer this over repeated todo_add calls whenever you know the full list of tasks upfront.
Duplicate titles or IDs are skipped with a note in the result — do not retry them.`,
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"items": map[string]any{
								"type":        "array",
								"description": "List of todo items to add",
								"items": map[string]any{
									"type": "object",
									"properties": map[string]any{
										"id":          map[string]any{"type": "string", "description": "Short kebab-case identifier. Must be unique in this session."},
										"title":       map[string]any{"type": "string", "description": "Short one-line title"},
										"description": map[string]any{"type": "string", "description": "Optional detail or acceptance criteria"},
									},
									"required": []string{"title"},
								},
							},
						},
						"required": []string{"items"},
					},
				},
				handle: m.handleTodoAddMany,
			},
			builtin{
				def: ai.ToolDefinition{
					Name:        "todo_update",
					Description: `Update the status (or title/description) of an existing todo item.`,
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"id":          map[string]any{"type": "string", "description": "Todo ID returned by todo_add"},
							"status":      map[string]any{"type": "string", "enum": []string{"pending", "in_progress", "done", "blocked"}, "description": "New status"},
							"title":       map[string]any{"type": "string", "description": "Replace title"},
							"description": map[string]any{"type": "string", "description": "Replace description"},
						},
						"required": []string{"id"},
					},
				},
				handle: m.handleTodoUpdate,
			},
			builtin{
				def: ai.ToolDefinition{
					Name:        "todo_list",
					Description: `Return the current todo list as text so you can review progress.`,
					InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
				},
				handle: m.handleTodoList,
			},
		)
	}

	if m.memoryStore != nil {
		builtins = append(builtins,
			builtin{
				def: ai.ToolDefinition{
					Name:        "memory_list",
					Description: `List all named project memory files for the current directory, with their sizes.`,
					InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
				},
				handle: m.handleMemoryList,
			},
			builtin{
				def: ai.ToolDefinition{
					Name: "memory_read",
					Description: `Read a named project memory file.
All memories are injected into your system prompt at session start; call this
only to re-read a specific memory mid-session (e.g. one that was too large to inject).`,
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type":        "string",
								"description": `Memory name (slug: lowercase letters, digits, hyphens; e.g. "general", "api-notes")`,
							},
						},
						"required": []string{"name"},
					},
				},
				handle: m.handleMemoryRead,
			},
			builtin{
				def: ai.ToolDefinition{
					Name:        "lookup_werkler_docs",
					Description: "Look up information from werkler's built-in documentation. Call this when the user asks about werkler features, keybindings, configuration options, modes, skills, autopilot, memory, or agents. Returns the relevant documentation sections.",
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"topic": map[string]any{
								"type":        "string",
								"description": `The topic or question to look up (e.g. "keybindings", "how to configure providers", "autopilot cycle limit")`,
							},
						},
						"required": []string{"topic"},
					},
				},
				handle: func(_ context.Context, args map[string]any) (string, error) {
					topic, _ := args["topic"].(string)
					return docs.Search(topic), nil
				},
			},
			builtin{
				def: ai.ToolDefinition{
					Name: "memory_write",
					Description: `Write (replace) a named project memory file.
Use named files to keep different concerns separate (e.g. "general", "conventions", "architecture").
Maximum ` + fmt.Sprintf("%d", memorystore.MaxBytesPerFile) + ` bytes per file; up to ` + fmt.Sprintf("%d", memorystore.MaxFiles) + ` files per project.
Use this to persist project knowledge across sessions: conventions, architecture decisions,
known issues, preferred patterns, important file locations.`,
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type":        "string",
								"description": `Memory name (slug: lowercase letters, digits, hyphens; e.g. "general", "api-notes")`,
							},
							"content": map[string]any{
								"type":        "string",
								"description": "Markdown content to store (replaces the named file's previous content)",
							},
						},
						"required": []string{"name", "content"},
					},
				},
				handle: m.handleMemoryWrite,
			},
			builtin{
				def: ai.ToolDefinition{
					Name: "memory_delete",
					Description: `Delete a named project memory file.
Use only when the memory is fully obsolete. This cannot be undone without rewriting from scratch.`,
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type":        "string",
								"description": "Name of the memory file to delete",
							},
						},
						"required": []string{"name"},
					},
				},
				handle: m.handleMemoryDelete,
			},
			builtin{
				def: ai.ToolDefinition{
					Name: "memory_promote",
					Description: `Move a named memory file to a parent directory's store, making it available to all sub-projects.
Use this when a note turns out to be relevant across the whole project or workspace (e.g. a monorepo root),
not just the current directory. The memory is deleted from the current directory after being moved.
Call memory_list first if you are unsure which memories exist.
target_directory accepts a relative path (e.g. ".." or "../..") or an absolute path; it must be a parent of the current project directory.`,
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type":        "string",
								"description": "Name of the memory to promote",
							},
							"target_directory": map[string]any{
								"type":        "string",
								"description": `Relative path to the target parent directory, e.g. ".." or "../.."`,
							},
						},
						"required": []string{"name", "target_directory"},
					},
				},
				handle: m.handleMemoryPromote,
			},
		)
	}

	if m.mcpMgr != nil {
		configured := m.mcpMgr.ConfiguredServers()
		if len(configured) > 0 {
			nameList := make([]string, len(configured))
			for i, srv := range configured {
				nameList[i] = srv.Name
			}
			builtins = append(builtins, builtin{
				def: ai.ToolDefinition{
					Name: "connect_server",
					Description: "Connect to a configured MCP server to make its tools available. " +
						"Call this immediately when the user's request requires tools from that server — " +
						"do not ask for permission first and do not connect servers unrelated to the current task. " +
						"After connecting, the server's tools will be listed in the result.",
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type":        "string",
								"description": "Name of the server to connect",
								"enum":        nameList,
							},
						},
						"required": []string{"name"},
					},
				},
				handle: m.handleConnectServer,
			})
		}
	}

	// calculate, and sleep are always registered.
	builtins = append(builtins,
		builtin{
			def: ai.ToolDefinition{
				Name: "calculate",
				Description: `Evaluate a mathematical expression and return the result.
Use this for arithmetic, unit conversions, and any calculation you would normally
estimate. Supports: +, -, *, /, % (remainder), bitwise &/|/^/<</>>/&^.
Unary minus and plus are supported. Integer literals: 0xff, 0b1010, 0o17.
Constants: pi, e, phi, sqrt2, ln2.
Functions: sqrt, cbrt, abs, floor, ceil, round, trunc, exp, exp2, log, log2,
log10, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, pow, hypot,
mod, min, max. Note: ^ is bitwise XOR; use pow(x,y) for exponentiation.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"expression": map[string]any{
							"type":        "string",
							"description": "Mathematical expression to evaluate (e.g. \"sqrt(2) * pi\", \"2**8\" is invalid — use pow(2,8))",
						},
					},
					"required": []string{"expression"},
				},
			},
			handle: m.handleCalculate,
		},
		builtin{
			def: ai.ToolDefinition{
				Name: "sleep",
				Description: `Pause execution for a duration or until a specific time.
Use this ONLY when you genuinely need to wait (e.g. polling for a file,
waiting for a background process, or an explicit user request to delay).
Avoid calling this unless strictly necessary.
Specify either "seconds" (float, max 600) OR "until" (RFC3339 timestamp, max 600s ahead).`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"seconds": map[string]any{
							"type":        "number",
							"description": "Seconds to sleep (max 600)",
						},
						"until": map[string]any{
							"type":        "string",
							"description": "RFC3339 timestamp to sleep until (max 600 s in the future)",
						},
					},
				},
			},
			handle: m.handleSleep,
		},
	)
	builtins = append(builtins, builtin{
		def: ai.ToolDefinition{
			Name: "task_start",
			Description: `Set the title of the task you are currently working on.
Call this whenever you begin a new sub-task or phase of work so the user can see
what you are doing in the status bar. You can call it multiple times to update
the title as work progresses. The title should be a short, human-readable phrase
such as "Implementing OAuth callback" or "Writing tests for parser".
Do NOT call this for every small step — only when starting a meaningful new phase.`,
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"title": map[string]any{
						"type":        "string",
						"description": "Short description of the current task or phase",
					},
				},
				"required": []string{"title"},
			},
		},
		handle: m.handleTaskStart,
	})
	// task_complete is always registered — autopilot and manual use both benefit.
	builtins = append(builtins, builtin{
		def: ai.ToolDefinition{
			Name: "task_complete",
			Description: `Signal that the assigned task is fully complete. Call this when all work is done and no further action is needed.
In autopilot mode this stops the autonomous loop. Outside autopilot mode it marks the task done and returns to idle.
Provide a concise summary of what was accomplished.`,
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"summary": map[string]any{
						"type":        "string",
						"description": "Concise summary of what was accomplished",
					},
				},
				"required": []string{"summary"},
			},
		},
		handle: m.handleTaskComplete,
	})
	builtins = append(builtins, builtin{
		def: ai.ToolDefinition{
			Name: "confirm_plan",
			Description: `Present the finalised plan to the user and ask whether to proceed with implementation.
Call this ONLY after writing the plan file and completing any review cycles.
Provide a brief 2-3 sentence summary of the plan for the user to review.
The user will choose one of: implement now, implement with autopilot, or reject.
The return value tells you how to proceed:
- "approved": implement immediately in the current conversation turn.
- "approved_with_autopilot": autopilot has been enabled — STOP your response; the autonomous loop will continue.
- "rejected: <reason>": acknowledge the reason and stop; do NOT start implementing.`,
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"summary": map[string]any{
						"type":        "string",
						"description": "Brief 2-3 sentence summary of the plan for the user",
					},
				},
				"required": []string{"summary"},
			},
		},
		handle: m.handleConfirmPlan,
	})

	return builtins
}

// --- Built-in handlers ---

func stringArg(args map[string]any, key string) string {
	v, _ := args[key].(string)
	return v
}

func boolArg(args map[string]any, key string) bool {
	v, _ := args[key].(bool)
	return v
}

func float64Arg(args map[string]any, key string, def float64) float64 {
	if v, ok := args[key].(float64); ok {
		return v
	}
	return def
}

func stringSliceArg(args map[string]any, key string) []string {
	raw, _ := args[key].([]any)
	out := make([]string, 0, len(raw))
	for _, v := range raw {
		if s, ok := v.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func stringMapArg(args map[string]any, key string) map[string]string {
	raw, _ := args[key].(map[string]any)
	out := make(map[string]string, len(raw))
	for k, v := range raw {
		if s, ok := v.(string); ok {
			out[k] = s
		}
	}
	return out
}

func jsonResult(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(b)
}

func (m *Manager) handleConnectServer(ctx context.Context, args map[string]any) (string, error) {
	name := stringArg(args, "name")
	if name == "" {
		return "", fmt.Errorf("connect_server: name is required")
	}
	already, err := m.mcpMgr.ConnectByName(ctx, name)
	if err != nil {
		return "", err
	}
	if already {
		return fmt.Sprintf("Server %q is already connected.", name), nil
	}
	toolNames, err := m.mcpMgr.ToolNamesForServer(ctx, name)
	if err != nil || len(toolNames) == 0 {
		return fmt.Sprintf("Connected to server %q. New tools from this server are now available.", name), nil
	}
	return fmt.Sprintf(
		"Connected to server %q. You can now call these tools directly:\n- %s",
		name, strings.Join(toolNames, "\n- "),
	), nil
}

func (m *Manager) handleProcessStart(ctx context.Context, args map[string]any) (string, error) {
	command := stringArg(args, "command")
	if command == "" {
		return "", fmt.Errorf("process_start: command is required")
	}
	cmdArgs := stringSliceArg(args, "args")
	title := stringArg(args, "title")
	if title == "" {
		// Fallback: derive a title from the command name + first arg so the TUI
		// always shows something meaningful even if the model skipped the field.
		base := filepath.Base(command)
		if len(cmdArgs) > 0 {
			title = base + " " + cmdArgs[0]
		} else {
			title = base
		}
		args["title"] = title
	}
	cwd := stringArg(args, "cwd")
	usePTY := boolArg(args, "pty")
	env := stringMapArg(args, "env")
	timeoutSecs := float64Arg(args, "timeout_seconds", 5)

	// Check path permissions before starting.
	if unapproved := m.checkWritePaths(ctx, command, cmdArgs, cwd); len(unapproved) > 0 {
		return "", &UnapprovedPathsError{Requests: unapproved}
	}

	handle, err := m.processes.Start(ctx, command, cmdArgs, cwd, env, usePTY)
	if err != nil {
		return "", fmt.Errorf("process_start: %w", err)
	}

	timeout := time.Duration(timeoutSecs * float64(time.Second))
	output, _, _, err := m.processes.ReadOutput(handle, timeout)
	if err != nil {
		output = ""
	}

	return jsonResult(map[string]any{
		"handle": handle,
		"output": output,
	}), nil
}

func (m *Manager) handleProcessSend(ctx context.Context, args map[string]any) (string, error) {
	handle := stringArg(args, "handle")
	text := stringArg(args, "text")
	timeoutSecs := float64Arg(args, "timeout_seconds", 2)

	if err := m.processes.Send(handle, text); err != nil {
		return "", err
	}

	timeout := time.Duration(timeoutSecs * float64(time.Second))
	output, running, exitCode, err := m.processes.ReadOutput(handle, timeout)
	if err != nil {
		return "", err
	}
	return jsonResult(map[string]any{
		"output":    output,
		"running":   running,
		"exit_code": exitCode,
	}), nil
}

func (m *Manager) handleProcessSendKey(ctx context.Context, args map[string]any) (string, error) {
	handle := stringArg(args, "handle")
	key := stringArg(args, "key")
	timeoutSecs := float64Arg(args, "timeout_seconds", 2)

	if err := m.processes.SendKey(handle, key); err != nil {
		return "", err
	}

	timeout := time.Duration(timeoutSecs * float64(time.Second))
	output, running, exitCode, err := m.processes.ReadOutput(handle, timeout)
	if err != nil {
		return "", err
	}
	return jsonResult(map[string]any{
		"output":    output,
		"running":   running,
		"exit_code": exitCode,
	}), nil
}

func (m *Manager) handleProcessRead(ctx context.Context, args map[string]any) (string, error) {
	handle := stringArg(args, "handle")
	timeoutSecs := float64Arg(args, "timeout_seconds", 5)

	timeout := time.Duration(timeoutSecs * float64(time.Second))
	output, running, exitCode, err := m.processes.ReadOutput(handle, timeout)
	if err != nil {
		return "", err
	}
	return jsonResult(map[string]any{
		"output":    output,
		"running":   running,
		"exit_code": exitCode,
	}), nil
}

func (m *Manager) handleProcessStop(_ context.Context, args map[string]any) (string, error) {
	handle := stringArg(args, "handle")
	force := boolArg(args, "force")

	exitCode, output, err := m.processes.Stop(handle, force)
	if err != nil {
		return "", err
	}
	return jsonResult(map[string]any{
		"exit_code": exitCode,
		"output":    output,
	}), nil
}

// runCommandOutputCap is the maximum bytes captured per stream (stdout / stderr / combined).
const runCommandOutputCap = 512 * 1024

// runCommandMaxTimeout is the maximum allowed value for timeout_seconds.
const runCommandMaxTimeout = 600.0

// capBuffer is a size-limited buffer. Writes beyond the cap are silently discarded;
// the Truncated field records the total number of bytes dropped.
type capBuffer struct {
	mu        sync.Mutex
	buf       bytes.Buffer
	cap       int
	written   int
	Truncated int
}

func newCapBuffer(cap int) *capBuffer { return &capBuffer{cap: cap} }

func (b *capBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	remaining := b.cap - b.written
	if remaining <= 0 {
		b.Truncated += len(p)
		return len(p), nil
	}
	if len(p) > remaining {
		b.buf.Write(p[:remaining])
		b.written += remaining
		b.Truncated += len(p) - remaining
		return len(p), nil
	}
	b.buf.Write(p)
	b.written += len(p)
	return len(p), nil
}

// String returns the captured bytes as a UTF-8 string, appending a truncation
// notice if any bytes were dropped.
func (b *capBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	raw := b.buf.Bytes()
	// Snap to last complete UTF-8 rune boundary.
	for len(raw) > 0 && !utf8.Valid(raw) {
		raw = raw[:len(raw)-1]
	}
	s := string(raw)
	if b.Truncated > 0 {
		s += fmt.Sprintf("\n[truncated: %d bytes omitted]", b.Truncated)
	}
	return s
}

// combinedWriter multiplexes writes to both a stream-specific buffer and a
// combined buffer. The combined buffer is protected by a separate mutex so
// writes from concurrent goroutines are serialised globally.
type combinedWriter struct {
	stream   *capBuffer
	combined *capBuffer
	mu       *sync.Mutex // shared mutex that serialises combined writes
}

func (w *combinedWriter) Write(p []byte) (int, error) {
	_, _ = w.stream.Write(p)
	w.mu.Lock()
	_, _ = w.combined.Write(p)
	w.mu.Unlock()
	return len(p), nil
}

func (m *Manager) handleRunCommand(ctx context.Context, args map[string]any) (string, error) {
	command := stringArg(args, "command")
	if command == "" {
		return jsonResult(map[string]any{"error": "run_command: command is required"}), nil
	}
	title := stringArg(args, "title")
	if title == "" {
		title = command
		args["title"] = title
	}

	useShell := boolArg(args, "shell")
	cmdArgs := stringSliceArg(args, "args")
	// When shell=true, args must be empty — the full command goes in "command".
	// Weaker models sometimes pass both; silently drop args rather than erroring
	// so the call still succeeds.
	if useShell && len(cmdArgs) > 0 {
		cmdArgs = nil
	}
	// When shell=false but command contains spaces, the model most likely intended
	// a shell command (e.g. "ls -la"). Auto-promote to shell mode so the call
	// succeeds rather than failing with "executable not found".
	if !useShell && strings.ContainsRune(command, ' ') {
		useShell = true
		cmdArgs = nil
	}

	cwd := stringArg(args, "cwd")
	if cwd != "" {
		info, err := os.Stat(cwd)
		if err != nil {
			return jsonResult(map[string]any{"error": fmt.Sprintf("run_command: cwd does not exist: %s", cwd)}), nil
		}
		if !info.IsDir() {
			return jsonResult(map[string]any{"error": fmt.Sprintf("run_command: cwd is not a directory: %s", cwd)}), nil
		}
	}

	timeoutSecs := float64Arg(args, "timeout_seconds", 30)
	if timeoutSecs <= 0 {
		timeoutSecs = 30
	}
	if timeoutSecs > runCommandMaxTimeout {
		timeoutSecs = runCommandMaxTimeout
	}

	// Build env: start from current environment, apply overrides.
	// A null JSON value for a key (which arrives as nil in map[string]any) means unset.
	baseEnv := os.Environ()
	var envOverrides map[string]any
	if raw, ok := args["env"].(map[string]any); ok {
		envOverrides = raw
	}
	var finalEnv []string
	if len(envOverrides) > 0 {
		// Build a map of key->value from os.Environ() then apply overrides.
		envMap := make(map[string]string, len(baseEnv))
		for _, kv := range baseEnv {
			if idx := strings.IndexByte(kv, '='); idx >= 0 {
				envMap[kv[:idx]] = kv[idx+1:]
			}
		}
		for k, v := range envOverrides {
			if v == nil {
				delete(envMap, k)
			} else if s, ok := v.(string); ok {
				envMap[k] = s
			}
		}
		finalEnv = make([]string, 0, len(envMap))
		for k, v := range envMap {
			finalEnv = append(finalEnv, k+"="+v)
		}
	} else {
		finalEnv = baseEnv
	}

	// Build the exec.Cmd.
	var cmd *exec.Cmd
	if useShell {
		cmd = exec.Command("bash", "-c", command)
	} else {
		cmd = exec.Command(command, cmdArgs...)
	}
	cmd.Env = finalEnv
	if cwd != "" {
		cmd.Dir = cwd
	}
	// Attach stdin to /dev/null so blocking reads are immediately detected.
	devNull, err := os.Open(os.DevNull)
	if err == nil {
		cmd.Stdin = devNull
		defer func() { _ = devNull.Close() }()
	}
	// Spawn in a new process group so we can kill the whole group on timeout.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}

	// Check path approval before executing.
	effectiveCommand := command
	if useShell {
		effectiveCommand = "bash"
	}
	if unapproved := m.checkWritePaths(ctx, effectiveCommand, cmdArgs, cwd); len(unapproved) > 0 {
		return "", &UnapprovedPathsError{Requests: unapproved}
	}

	// Set up output capture.
	stdoutBuf := newCapBuffer(runCommandOutputCap)
	stderrBuf := newCapBuffer(runCommandOutputCap)
	combinedBuf := newCapBuffer(runCommandOutputCap)
	combinedMu := &sync.Mutex{}

	cmd.Stdout = &combinedWriter{stream: stdoutBuf, combined: combinedBuf, mu: combinedMu}
	cmd.Stderr = &combinedWriter{stream: stderrBuf, combined: combinedBuf, mu: combinedMu}

	if startErr := cmd.Start(); startErr != nil {
		return jsonResult(map[string]any{"error": fmt.Sprintf("run_command: %s", startErr.Error())}), nil
	}

	// Wait for process in a goroutine so we can enforce the timeout.
	type waitResult struct {
		exitCode int
	}
	done := make(chan waitResult, 1)
	go func() {
		err := cmd.Wait()
		code := 0
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				code = exitErr.ExitCode()
			} else {
				code = -1
			}
		}
		done <- waitResult{code}
	}()

	timeout := time.Duration(timeoutSecs * float64(time.Second))
	timedOut := false
	var exitCode int

	select {
	case result := <-done:
		exitCode = result.exitCode
	case <-time.After(timeout):
		timedOut = true
		// Kill the process group: SIGTERM first, then SIGKILL after 2s grace.
		if cmd.Process != nil {
			pgid := -cmd.Process.Pid // negative PID signals the whole process group
			_ = syscall.Kill(pgid, syscall.SIGTERM)
			select {
			case result := <-done:
				exitCode = result.exitCode
			case <-time.After(2 * time.Second):
				_ = syscall.Kill(pgid, syscall.SIGKILL)
				result := <-done
				exitCode = result.exitCode
			}
		}
	}

	return jsonResult(map[string]any{
		"exit_code":       exitCode,
		"stdout":          stdoutBuf.String(),
		"stderr":          stderrBuf.String(),
		"combined_output": combinedBuf.String(),
		"timed_out":       timedOut,
	}), nil
}

// --- File tool handlers ---

// maxFileReadMultiBytes is the aggregate output cap for file_read_multi.
const maxFileReadMultiBytes = 8 << 10 // 8 KiB

func (m *Manager) handleFileReadMulti(ctx context.Context, args map[string]any) (string, error) {
	rawRegions, _ := args["regions"].([]any)
	if len(rawRegions) == 0 {
		return "", fmt.Errorf("file_read_multi: regions must be a non-empty array")
	}

	type region struct {
		rawPath   string
		path      string
		startLine int
		endLine   int
		hasStart  bool
		hasEnd    bool
	}

	regions := make([]region, 0, len(rawRegions))
	for i, r := range rawRegions {
		rm, ok := r.(map[string]any)
		if !ok {
			return "", fmt.Errorf("file_read_multi: region[%d] is not an object", i)
		}
		rp := stringArg(rm, "path")
		if rp == "" {
			return "", fmt.Errorf("file_read_multi: region[%d]: path is required", i)
		}
		reg := region{rawPath: rp, path: canonicalizePath(rp)}
		if v, ok := rm["start_line"]; ok {
			if f, ok2 := toFloat64(v); ok2 {
				reg.startLine = int(f)
			}
			reg.hasStart = true
		}
		if v, ok := rm["end_line"]; ok {
			if f, ok2 := toFloat64(v); ok2 {
				reg.endLine = int(f)
			}
			reg.hasEnd = true
		}
		regions = append(regions, reg)
	}

	// Batch path approval: collect all unique unapproved paths.
	approver := m.activeApprover(ctx)
	if approver != nil {
		seen := make(map[string]bool)
		var unapproved []chat.PathAccessRequest
		for _, reg := range regions {
			if !seen[reg.path] && !approver.IsPathReadApproved(reg.path) {
				unapproved = append(unapproved, chat.PathAccessRequest{Path: reg.path, Write: false})
				seen[reg.path] = true
			}
		}
		if len(unapproved) > 0 {
			return "", &UnapprovedPathsError{Requests: unapproved}
		}
	}

	var out strings.Builder
	totalBytes := 0

	for i, reg := range regions {
		if i > 0 {
			out.WriteString("\n")
		}

		info, err := os.Stat(reg.path)
		if err != nil {
			fmt.Fprintf(&out, "=== %s [ERROR] ===\n%s\n", reg.rawPath, err.Error())
			continue
		}
		if info.IsDir() {
			fmt.Fprintf(&out, "=== %s [ERROR] ===\ndirectory; use file_list\n", reg.rawPath)
			continue
		}

		data, err := os.ReadFile(reg.path)
		if err != nil {
			fmt.Fprintf(&out, "=== %s [ERROR] ===\n%s\n", reg.rawPath, err.Error())
			continue
		}
		if !utf8.Valid(data) {
			fmt.Fprintf(&out, "=== %s [ERROR] ===\nbinary file; use process_start\n", reg.rawPath)
			continue
		}

		lines := strings.Split(string(data), "\n")
		totalLines := len(lines)

		startLine := 1
		endLine := totalLines
		if reg.hasStart {
			startLine = reg.startLine
		}
		if reg.hasEnd {
			endLine = reg.endLine
		}
		if startLine < 1 {
			startLine = 1
		}
		if endLine > totalLines {
			endLine = totalLines
		}
		if startLine > endLine {
			fmt.Fprintf(&out, "=== %s [ERROR] ===\nstart_line (%d) > end_line (%d)\n", reg.rawPath, startLine, endLine)
			continue
		}

		selected := lines[startLine-1 : endLine]

		rangeLabel := fmt.Sprintf("L%d-L%d", startLine, startLine+len(selected)-1)
		header := fmt.Sprintf("=== %s [%s of %d] ===\n", reg.rawPath, rangeLabel, totalLines)
		out.WriteString(header)
		totalBytes += len(header)

		var sectionBuf strings.Builder
		for i, l := range selected {
			fmt.Fprintf(&sectionBuf, "%4d│%s\n", startLine+i, l)
		}
		section := sectionBuf.String()

		remaining := maxFileReadMultiBytes - totalBytes
		if remaining <= 0 {
			out.WriteString("[output cap reached — omitted]\n")
			break
		}
		if len(section) > remaining {
			out.WriteString(section[:remaining])
			out.WriteString("\n[output cap reached — truncated]\n")
			break
		}
		out.WriteString(section)
		totalBytes += len(section)
	}

	return out.String(), nil
}

func (m *Manager) handleReadImage(ctx context.Context, args map[string]any) (string, []ai.ImagePart, error) {
	rawPath := stringArg(args, "path")
	if rawPath == "" {
		return "", nil, fmt.Errorf("read_image: path is required")
	}
	path := canonicalizePath(rawPath)

	if err := m.checkSingleRead(ctx, path); err != nil {
		return "", nil, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", nil, fmt.Errorf("read_image: %w", err)
	}

	mime := http.DetectContentType(data)
	if mime == "application/octet-stream" {
		ext := strings.ToLower(filepath.Ext(path))
		switch ext {
		case ".png":
			mime = "image/png"
		case ".jpg", ".jpeg":
			mime = "image/jpeg"
		case ".gif":
			mime = "image/gif"
		case ".webp":
			mime = "image/webp"
		default:
			return "", nil, fmt.Errorf("read_image: unsupported image format (extension %s)", ext)
		}
	}
	if !strings.HasPrefix(mime, "image/") {
		return "", nil, fmt.Errorf("read_image: %s is not an image file (detected content type: %s)", path, mime)
	}

	part := ai.ImagePart{
		Data:     data,
		MIMEType: mime,
		Name:     filepath.Base(path),
	}
	return fmt.Sprintf("Image loaded: %s (%s, %d bytes)", filepath.Base(path), mime, len(data)), []ai.ImagePart{part}, nil
}

func (m *Manager) handleFileList(ctx context.Context, args map[string]any) (string, error) {
	rawPath := stringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_list: path is required")
	}
	path := canonicalizePath(rawPath)

	if err := m.checkSingleRead(ctx, path); err != nil {
		return "", err
	}

	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("file_list: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("file_list: %s is not a directory; use file_read to read files", path)
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return "", fmt.Errorf("file_list: %w", err)
	}

	type entry struct {
		Name string `json:"name"`
		Type string `json:"type"`
		Size *int64 `json:"size,omitempty"`
	}
	result := make([]entry, 0, len(entries))
	for _, e := range entries {
		var kind string
		var size *int64
		switch {
		case e.IsDir():
			kind = "directory"
		case e.Type()&fs.ModeSymlink != 0:
			kind = "symlink"
		default:
			kind = "file"
			if fi, err2 := e.Info(); err2 == nil {
				s := fi.Size()
				size = &s
			}
		}
		result = append(result, entry{Name: e.Name(), Type: kind, Size: size})
	}
	return jsonResult(result), nil
}

func (m *Manager) handleFileWrite(ctx context.Context, args map[string]any) (string, error) {
	rawPath := stringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_write: path is required")
	}
	path := canonicalizePath(rawPath)
	content := stringArg(args, "content")
	createParents := boolArg(args, "create_parents")

	if err := m.checkSingleWrite(ctx, path); err != nil {
		return "", err
	}

	if createParents {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return "", fmt.Errorf("file_write: creating parent directories: %w", err)
		}
	}

	oldBytes, _ := os.ReadFile(path)
	oldContent := string(oldBytes)

	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("file_write: %w", err)
	}

	return jsonResult(map[string]any{
		"path":  path,
		"bytes": len(content),
		"diff":  computeUnifiedDiff(oldContent, content, path),
	}), nil
}

func (m *Manager) handleFileEdit(ctx context.Context, args map[string]any) (string, error) {
	rawPath := stringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_edit: path is required")
	}
	path := canonicalizePath(rawPath)

	// Build the list of {old_str, new_str} pairs.
	type hunk struct{ old, new string }
	var hunks []hunk

	if rawEdits, ok := args["edits"].([]any); ok && len(rawEdits) > 0 {
		// Multi-hunk form: edits: [{old_str, new_str}, …]
		for i, item := range rawEdits {
			m, ok := item.(map[string]any)
			if !ok {
				return "", fmt.Errorf("file_edit: edits[%d] must be an object", i)
			}
			old, _ := m["old_str"].(string)
			if old == "" {
				return "", fmt.Errorf("file_edit: edits[%d].old_str must not be empty", i)
			}
			newVal, _ := m["new_str"].(string)
			hunks = append(hunks, hunk{old, newVal})
		}
	} else {
		// Single-hunk form: top-level old_str / new_str.
		oldStr := stringArg(args, "old_str")
		newStr := stringArg(args, "new_str")
		if oldStr == "" {
			return "", fmt.Errorf("file_edit: old_str must not be empty; use file_write to overwrite entire files")
		}
		hunks = []hunk{{oldStr, newStr}}
	}

	if err := m.checkSingleWrite(ctx, path); err != nil {
		return "", err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("file_edit: %w", err)
	}
	if !utf8.Valid(data) {
		return "", fmt.Errorf("file_edit: %s appears to be a binary file", path)
	}

	content := string(data)

	// Validate all hunks before writing anything (atomic failure).
	for i, h := range hunks {
		count := strings.Count(content, h.old)
		switch {
		case count == 0:
			if len(hunks) == 1 {
				return "", fmt.Errorf("file_edit: old_str not found in %s — call file_read on the file first and use the exact text from the file as old_str (watch for whitespace and indentation)", path)
			}
			return "", fmt.Errorf("file_edit: edits[%d] old_str not found in %s", i, path)
		case count > 1:
			lineNums := findMatchLines(content, h.old)
			if len(hunks) == 1 {
				return "", fmt.Errorf("file_edit: old_str matches %d times in %s (at lines %v); include more surrounding context to make it unique",
					count, path, lineNums)
			}
			return "", fmt.Errorf("file_edit: edits[%d] old_str matches %d times in %s (at lines %v); include more surrounding context",
				i, count, path, lineNums)
		}
	}

	// Apply all hunks in order.
	type result struct {
		line    int
		added   int
		removed int
	}
	results := make([]result, len(hunks))
	for i, h := range hunks {
		idx := strings.Index(content, h.old)
		line := strings.Count(content[:idx], "\n") + 1
		removed := strings.Count(h.old, "\n") + 1
		added := strings.Count(h.new, "\n") + 1
		results[i] = result{line, added, removed}
		content = strings.Replace(content, h.old, h.new, 1)
	}

	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("file_edit: writing %s: %w", path, err)
	}

	newContent, _ := os.ReadFile(path)
	diff := computeUnifiedDiff(string(data), string(newContent), path)

	if len(hunks) == 1 {
		return jsonResult(map[string]any{
			"path":    path,
			"line":    results[0].line,
			"added":   results[0].added,
			"removed": results[0].removed,
			"diff":    diff,
		}), nil
	}
	editsOut := make([]map[string]any, len(results))
	for i, r := range results {
		editsOut[i] = map[string]any{"line": r.line, "added": r.added, "removed": r.removed}
	}
	return jsonResult(map[string]any{"path": path, "edits": editsOut, "diff": diff}), nil
}

// findMatchLines returns the starting line numbers for all occurrences of substr in text.
func findMatchLines(text, substr string) []int {
	var lines []int
	offset := 0
	for {
		idx := strings.Index(text[offset:], substr)
		if idx < 0 {
			break
		}
		abs := offset + idx
		lines = append(lines, strings.Count(text[:abs], "\n")+1)
		offset = abs + len(substr)
	}
	return lines
}

func (m *Manager) handleFileDelete(ctx context.Context, args map[string]any) (string, error) {
	rawPath := stringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_delete: path is required")
	}
	path := canonicalizePath(rawPath)

	if err := m.checkSingleWrite(ctx, path); err != nil {
		return "", err
	}

	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("file_delete: %w", err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("file_delete: %s is a directory; use process_start with rm -r to delete directories", path)
	}

	oldBytes, _ := os.ReadFile(path)

	if err := os.Remove(path); err != nil {
		return "", fmt.Errorf("file_delete: %w", err)
	}
	return jsonResult(map[string]any{
		"deleted": path,
		"diff":    computeUnifiedDiff(string(oldBytes), "", path),
	}), nil
}

// ensure Manager implements chat.ToolManager.
var _ chat.ToolManager = (*Manager)(nil)

func (m *Manager) handleAskUser(ctx context.Context, args map[string]any) (string, error) {
	question := stringArg(args, "question")
	if question == "" {
		return "error: ask_user requires a non-empty question", nil
	}
	choices := stringSliceArg(args, "choices")
	recommended := stringArg(args, "recommended_choice")
	allowFreeform := true
	if v, ok := args["allow_freeform"].(bool); ok {
		allowFreeform = v
	}
	if !allowFreeform && len(choices) == 0 {
		return "error: ask_user requires at least one choice when allow_freeform is false", nil
	}
	if recommended != "" {
		found := false
		for _, c := range choices {
			if c == recommended {
				found = true
				break
			}
		}
		if !found {
			return fmt.Sprintf("error: recommended_choice %q does not match any provided choice", recommended), nil
		}
	}
	if m.userAsker == nil {
		return "(ask_user requires interactive mode — run werkler interactively to provide an answer)", nil
	}
	return m.userAsker(ctx, question, choices, recommended, allowFreeform)
}

// --- Subagent framework ---

// subagentTools returns the built-in tool definitions a subagent is allowed to call.
func (m *Manager) subagentTools(sa *subagentDef) []ai.ToolDefinition {
	var out []ai.ToolDefinition
	for _, b := range m.builtins {
		if sa.allowedTools[b.def.Name] && b.handle != nil {
			out = append(out, b.def)
		}
	}
	return out
}

// callBuiltinAsSubagent dispatches a built-in tool call on behalf of a subagent.
// Uses withReviewMode so path checks use the permissive reviewerApprover.
func (m *Manager) callBuiltinAsSubagent(ctx context.Context, sa *subagentDef, name string, args map[string]any) (string, error) {
	if !sa.allowedTools[name] {
		return "", fmt.Errorf("tool %q is not available to subagent %q", name, sa.name)
	}
	subCtx := withReviewMode(ctx)
	for _, b := range m.builtins {
		if b.def.Name == name && b.handle != nil {
			return b.handle(subCtx, args)
		}
	}
	return "", fmt.Errorf("tool %q is not available to subagent %q", name, sa.name)
}

// runSubagent runs the agentic loop for sa, feeding userContent as the first
// user message. Returns the subagent's final text response.
func (m *Manager) runSubagent(ctx context.Context, sa *subagentDef, args map[string]any) (string, error) {
	content := stringArg(args, "context")
	if content == "" {
		return "error: subagent " + sa.name + " requires a non-empty context", nil
	}
	userContent := content
	if focus := stringArg(args, "focus"); focus != "" {
		userContent += "\n\nFocus particularly on: " + focus
	}

	messages := []ai.Message{
		{Role: "system", Content: sa.systemPrompt},
		{Role: "user", Content: userContent},
	}
	tools := m.subagentTools(sa)

	for range sa.maxSteps {
		msg, err := sa.completer.Complete(ctx, messages, tools)
		if err != nil {
			return "", fmt.Errorf("subagent %s failed: %w", sa.name, err)
		}
		messages = append(messages, msg)

		if len(msg.ToolCalls) == 0 {
			if msg.Content == "" {
				return "", fmt.Errorf("subagent %s produced no response", sa.name)
			}
			return msg.Content, nil
		}

		for _, tc := range msg.ToolCalls {
			result, callErr := m.callBuiltinAsSubagent(ctx, sa, tc.Name, tc.Arguments)
			if callErr != nil {
				result = "error: " + callErr.Error()
			}
			messages = append(messages, ai.Message{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			})
		}
	}
	return "", fmt.Errorf("subagent %s exceeded %d steps without producing a response", sa.name, sa.maxSteps)
}

// makeRubberDuckSubagent returns the built-in code-reviewer subagent definition.
func makeRubberDuckSubagent(c ai.Completer, label string) subagentDef {
	return subagentDef{
		name: "rubber_duck_review",
		description: fmt.Sprintf(
			"Submit your own pre-drafted plan or implementation to a separate reviewer AI for critical feedback.\n"+
				"Reviewer: %s\n"+
				"You MUST have a concrete plan of your own before calling this tool — do NOT use it to brainstorm or ask the reviewer to generate a plan.\n"+
				"Use it to catch bugs, logic errors, design flaws, or missing edge cases before implementing.\n"+
				"Provide complete context so the reviewer can give useful feedback.",
			label,
		),
		systemPrompt: `You are a technical reviewer. Critically evaluate the plan, code, or reasoning you are given.
Identify: correctness issues, bugs, logic errors, edge cases, security concerns, design flaws.
Be concise. Only surface issues that genuinely matter.
If you find no significant issues, say so briefly.
Do NOT comment on style, formatting, naming conventions, or other minor matters.

RESEARCH REQUIREMENT: Before writing your review you MUST use your tools to verify the plan
against the actual codebase. Do not trust the description alone — read the real files.
Specifically:
- Use file_read_multi to read every source file that the plan touches or references.
- Use run_command (shell=true) with grep/rg to find related functions, types, and patterns.
- Use file_list to discover files when paths are not given explicitly.
- Check that types, field names, function signatures, and interfaces cited in the plan actually
  exist and match what the plan says.
- Look for existing code that conflicts with or duplicates the proposed changes.
Only after this investigation should you write your final review.

IMPORTANT: Your every response must be either a tool call or your complete final review.
Never output "I will now...", "Let me read...", or any other intermediate narration — that would
be treated as your final answer. Call tools to gather context, then produce the full review.`,
		allowedTools: map[string]bool{
			"file_read_multi": true,
			"file_list":       true,
			"run_command":     true,
		},
		maxSteps:  20,
		completer: c,
		label:     label,
	}
}

// makeDocReviewSubagent returns the built-in documentation clarity reviewer subagent.
func makeDocReviewSubagent(c ai.Completer, label string) subagentDef {
	return subagentDef{
		name: "doc_review",
		description: fmt.Sprintf(
			"Submit documentation to a non-technical end-user proxy for plain-language and clarity review.\n"+
				"Reviewer: %s\n"+
				"Use this to check that docs are easy to understand for someone unfamiliar with the codebase.\n"+
				"The reviewer will flag jargon, confusing phrasing, missing context, and unclear instructions.",
			label,
		),
		systemPrompt: `You are a non-technical end user reviewing documentation.
Evaluate whether the content is clear, easy to understand, and free of unnecessary jargon.
Flag: confusing phrasing, unexplained technical terms, missing context, unclear steps, or anything a new user would find hard to follow.
Be concise. Only surface issues that genuinely affect clarity or usability.
Do NOT comment on technical correctness, code style, or implementation details.

IMPORTANT: Your every response must be either a tool call or your complete final review.
Never output "I will now...", "Let me read...", or any other intermediate narration — that would
be treated as your final answer.`,
		allowedTools: map[string]bool{
			"file_read_multi": true,
			"file_list":       true,
		},
		maxSteps:  10,
		completer: c,
		label:     label,
	}
}

func (m *Manager) handleUseSkill(_ context.Context, args map[string]any) (string, error) {
	name := stringArg(args, "name")
	for _, s := range m.skills {
		if s.Name == name {
			return s.Content, nil
		}
	}
	return "skill not found: " + name, nil
}

func (m *Manager) handleUseAgent(_ context.Context, args map[string]any) (string, error) {
	name := stringArg(args, "name")
	for _, a := range m.agents {
		if a.Name == name {
			if m.agentActivate != nil {
				m.agentActivate(name)
			}
			return fmt.Sprintf("Agent %q activated.", name), nil
		}
	}
	return fmt.Sprintf("agent not found: %q", name), nil
}

func (m *Manager) handleTodoAdd(_ context.Context, args map[string]any) (string, error) {
	title := stringArg(args, "title")
	if title == "" {
		return "error: title is required", nil
	}
	requestedID := stringArg(args, "id")
	// Reject exact-duplicate titles or IDs so the AI doesn't create ghost copies.
	for _, t := range m.todoStore.List() {
		if t.Title == title {
			return fmt.Sprintf("duplicate: todo %q already exists with id=%s status=%s — use todo_update to change it", title, t.ID, t.Status), nil
		}
		if requestedID != "" && t.ID == requestedID {
			return fmt.Sprintf("duplicate: id %q already exists (title=%q status=%s) — choose a different id or use todo_update", requestedID, t.Title, t.Status), nil
		}
	}
	id := m.todoStore.Add(requestedID, title, stringArg(args, "description"))
	return fmt.Sprintf("added: id=%s status=pending title=%q", id, title), nil
}

func (m *Manager) handleTodoAddMany(_ context.Context, args map[string]any) (string, error) {
	rawItems, _ := args["items"].([]any)
	if len(rawItems) == 0 {
		return "error: items array is required and must not be empty", nil
	}
	var sb strings.Builder
	existing := m.todoStore.List()
	for i, raw := range rawItems {
		item, ok := raw.(map[string]any)
		if !ok {
			fmt.Fprintf(&sb, "item %d: skipped (invalid format)\n", i+1)
			continue
		}
		title := stringArg(item, "title")
		if title == "" {
			fmt.Fprintf(&sb, "item %d: skipped (title is required)\n", i+1)
			continue
		}
		requestedID := stringArg(item, "id")
		skipped := false
		for _, t := range existing {
			if t.Title == title {
				fmt.Fprintf(&sb, "duplicate: %q already exists id=%s status=%s\n", title, t.ID, t.Status)
				skipped = true
				break
			}
			if requestedID != "" && t.ID == requestedID {
				fmt.Fprintf(&sb, "duplicate id %q (title=%q status=%s) — choose a different id\n", requestedID, t.Title, t.Status)
				skipped = true
				break
			}
		}
		if skipped {
			continue
		}
		id := m.todoStore.Add(requestedID, title, stringArg(item, "description"))
		// Keep existing list in sync so later items in the same batch are checked correctly.
		existing = m.todoStore.List()
		fmt.Fprintf(&sb, "added: id=%s title=%q\n", id, title)
	}
	return strings.TrimSpace(sb.String()), nil
}

func (m *Manager) handleTodoUpdate(_ context.Context, args map[string]any) (string, error) {
	id := stringArg(args, "id")
	if id == "" {
		return "error: id is required", nil
	}
	var f todostore.UpdateFields
	if v := stringArg(args, "status"); v != "" {
		f.Status = &v
	}
	if v := stringArg(args, "title"); v != "" {
		f.Title = &v
	}
	if v := stringArg(args, "description"); v != "" {
		f.Description = &v
	}
	if err := m.todoStore.Update(id, f); err != nil {
		return "error: " + err.Error(), nil
	}
	// Echo the updated todo so the AI knows exactly what changed.
	for _, t := range m.todoStore.List() {
		if t.ID == id {
			return fmt.Sprintf("updated: id=%s status=%s title=%q", t.ID, t.Status, t.Title), nil
		}
	}
	return "updated: " + id, nil
}

func (m *Manager) handleTodoList(_ context.Context, _ map[string]any) (string, error) {
	todos := m.todoStore.List()
	if len(todos) == 0 {
		return "no todos", nil
	}
	var sb strings.Builder
	icons := map[string]string{
		todostore.StatusPending:    "○",
		todostore.StatusInProgress: "▶",
		todostore.StatusDone:       "✓",
		todostore.StatusBlocked:    "✗",
	}
	for _, t := range todos {
		icon := icons[t.Status]
		if icon == "" {
			icon = "?"
		}
		fmt.Fprintf(&sb, "%s [%s] %s: %s\n", icon, t.ID, t.Status, t.Title)
		if t.Description != "" {
			fmt.Fprintf(&sb, "    %s\n", t.Description)
		}
	}
	return strings.TrimRight(sb.String(), "\n"), nil
}

func (m *Manager) handleTaskStart(_ context.Context, args map[string]any) (string, error) {
	title := stringArg(args, "title")
	if m.taskTitle != nil {
		m.taskTitle(title)
	}
	return "ok", nil
}

func (m *Manager) handleTaskComplete(_ context.Context, args map[string]any) (string, error) {
	summary := stringArg(args, "summary")
	if summary == "" {
		summary = "Task complete."
	}
	return summary, nil
}

func (m *Manager) handleConfirmPlan(ctx context.Context, args map[string]any) (string, error) {
	summary := stringArg(args, "summary")
	if m.planConfirmer == nil {
		// Non-interactive (prompt mode or no TUI): auto-approve without a dialog.
		return "approved: proceed with implementation", nil
	}
	return m.planConfirmer(ctx, summary)
}

func (m *Manager) handleCalculate(_ context.Context, args map[string]any) (string, error) {
	expr := stringArg(args, "expression")
	if expr == "" {
		return "error: expression is required", nil
	}
	v, err := evalExpression(expr)
	if err != nil {
		return fmt.Sprintf("error: %s", err), nil
	}
	return fmt.Sprintf("%s = %s", expr, formatResult(v)), nil
}

const maxSleepSeconds = 600.0

func (m *Manager) handleSleep(ctx context.Context, args map[string]any) (string, error) {
	var dur time.Duration

	if until := stringArg(args, "until"); until != "" {
		t, err := time.Parse(time.RFC3339, until)
		if err != nil {
			return fmt.Sprintf("error: invalid 'until' timestamp %q: %s", until, err), nil
		}
		dur = time.Until(t)
	} else if v, ok := args["seconds"]; ok {
		secs, ok2 := toFloat64(v)
		if !ok2 || secs < 0 {
			return "error: 'seconds' must be a non-negative number", nil
		}
		dur = time.Duration(secs * float64(time.Second))
	} else {
		return "error: specify either 'seconds' or 'until'", nil
	}

	if dur <= 0 {
		return "0s elapsed (target time already passed)", nil
	}
	if dur > maxSleepSeconds*time.Second {
		dur = maxSleepSeconds * time.Second
	}

	start := time.Now()
	select {
	case <-ctx.Done():
		return fmt.Sprintf("sleep cancelled after %s", time.Since(start).Round(time.Millisecond)), nil
	case <-time.After(dur):
		return fmt.Sprintf("slept %s", dur.Round(time.Millisecond)), nil
	}
}

// toFloat64 coerces a JSON-decoded number value to float64.
func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	}
	return 0, false
}

func (m *Manager) handleMemoryList(_ context.Context, _ map[string]any) (string, error) {
	entries := m.memoryStore.List()
	if len(entries) == 0 {
		return "(no project memories stored yet)", nil
	}
	var sb strings.Builder
	for _, e := range entries {
		fmt.Fprintf(&sb, "- %s (%d bytes)\n", e.Name, e.Size)
	}
	return strings.TrimSpace(sb.String()), nil
}

func (m *Manager) handleMemoryRead(_ context.Context, args map[string]any) (string, error) {
	name := stringArg(args, "name")
	if name == "" {
		return "error: name is required", nil
	}
	content, err := m.memoryStore.ReadNamed(name)
	if err != nil {
		return "error reading project memory: " + err.Error(), nil
	}
	if content == "" {
		return fmt.Sprintf("(no memory named %q exists)", name), nil
	}
	return content, nil
}

func (m *Manager) handleMemoryWrite(_ context.Context, args map[string]any) (string, error) {
	name := stringArg(args, "name")
	content := stringArg(args, "content")
	if name == "" {
		return "error: name is required", nil
	}
	if err := m.memoryStore.WriteNamed(name, content); err != nil {
		return "error writing project memory: " + err.Error(), nil
	}
	return fmt.Sprintf("memory %q saved (%d bytes)", name, len(strings.TrimSpace(content))), nil
}

func (m *Manager) handleMemoryDelete(_ context.Context, args map[string]any) (string, error) {
	name := stringArg(args, "name")
	if name == "" {
		return "error: name is required", nil
	}
	if err := m.memoryStore.DeleteNamed(name); err != nil {
		return "error deleting project memory: " + err.Error(), nil
	}
	return fmt.Sprintf("memory %q deleted", name), nil
}

func (m *Manager) handleMemoryPromote(_ context.Context, args map[string]any) (string, error) {
	name := stringArg(args, "name")
	targetDir := stringArg(args, "target_directory")
	if name == "" {
		return "error: name is required", nil
	}
	if targetDir == "" {
		return "error: target_directory is required", nil
	}
	if err := m.memoryStore.Promote(name, targetDir); err != nil {
		return "error promoting memory: " + err.Error(), nil
	}
	return fmt.Sprintf("memory %q promoted to %s", name, targetDir), nil
}

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
	"context"
	"fmt"
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/icedream/werkler/docs"
	"github.com/icedream/werkler/internal/agents"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/config"
	"github.com/icedream/werkler/internal/memorystore"
	"github.com/icedream/werkler/internal/process"
	"github.com/icedream/werkler/internal/skills"
	"github.com/icedream/werkler/internal/todostore"

	agenttools "github.com/icedream/werkler/internal/tools/agent"
	filetools "github.com/icedream/werkler/internal/tools/file"
	memorytools "github.com/icedream/werkler/internal/tools/memory"
	processtools "github.com/icedream/werkler/internal/tools/process"
	servertools "github.com/icedream/werkler/internal/tools/server"
	tasktools "github.com/icedream/werkler/internal/tools/task"
	"github.com/icedream/werkler/internal/tools/toolutil"
	usertools "github.com/icedream/werkler/internal/tools/user"
	utiltools "github.com/icedream/werkler/internal/tools/util"
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

	// Subpackage handlers — each encapsulates one category of built-in tools.
	fileHandler    *filetools.Handler
	processHandler *processtools.Handler
	serverHandler  *servertools.Handler
	userHandler    *usertools.Handler
	taskHandler    *tasktools.Handler
	memoryHandler  *memorytools.Handler
	agentHandler   *agenttools.Handler
	utilHandler    *utiltools.Handler
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
	m.fileHandler = filetools.NewHandler(m)
	m.processHandler = processtools.NewHandler(m)
	m.serverHandler = servertools.NewHandler(m.mcpMgr)
	m.userHandler = usertools.NewHandler(m)
	m.taskHandler = tasktools.NewHandler(m)
	m.memoryHandler = memorytools.NewHandler(m)
	m.agentHandler = agenttools.NewHandler(m)
	m.utilHandler = utiltools.NewHandler()
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

// checkShellCommandWritePaths is like checkWritePaths for shell=true commands.
// The shell interpreter (bash/sh) is always trusted and skipped from approval;
// the TOOL approval dialog already shows the user the full command.
func (m *Manager) checkShellCommandWritePaths(ctx context.Context, shell, command, cwd string) []chat.PathAccessRequest {
	approver := m.activeApprover(ctx)
	if approver == nil {
		return nil
	}
	// Pass the command as a -c argument so ExtractPaths can scan it for paths.
	// Index 0 is the interpreter itself — always skip it.
	paths := ExtractPaths(shell, []string{"-c", command}, cwd)
	var unapproved []chat.PathAccessRequest
	for i, p := range paths {
		if i == 0 {
			continue // interpreter is auto-approved
		}
		if !approver.IsPathWriteApproved(p) {
			unapproved = append(unapproved, chat.PathAccessRequest{Path: p, Write: true})
		}
	}
	return unapproved
}

// --- Built-in tool definitions ---

func (m *Manager) makeBuiltins() []builtin {
	// Lazy-initialise handlers so tests that construct Manager directly (not via
	// New()) still get a working set of tools.
	if m.fileHandler == nil {
		m.fileHandler = filetools.NewHandler(m)
	}
	if m.processHandler == nil {
		m.processHandler = processtools.NewHandler(m)
	}
	if m.serverHandler == nil {
		m.serverHandler = servertools.NewHandler(m.mcpMgr)
	}
	if m.userHandler == nil {
		m.userHandler = usertools.NewHandler(m)
	}
	if m.taskHandler == nil {
		m.taskHandler = tasktools.NewHandler(m)
	}
	if m.memoryHandler == nil {
		m.memoryHandler = memorytools.NewHandler(m)
	}
	if m.agentHandler == nil {
		m.agentHandler = agenttools.NewHandler(m)
	}
	if m.utilHandler == nil {
		m.utilHandler = utiltools.NewHandler()
	}

	// Collect Builtin definitions from all subpackage handlers, then convert
	// them to the internal builtin type used by the dispatcher.
	var all []toolutil.Builtin

	// Process and command tools.
	all = append(all, m.processHandler.Tools()...)

	// File system tools.
	all = append(all, m.fileHandler.Tools()...)

	// User interaction.
	all = append(all, m.userHandler.Tools()...)

	// Task management and todo list.
	all = append(all, m.taskHandler.Tools()...)

	// Project memory.
	if m.memoryStore != nil {
		all = append(all, m.memoryHandler.Tools()...)
	}

	// Skill and agent activation.
	all = append(all, m.agentHandler.Tools()...)

	// Utility (calculate, sleep).
	all = append(all, m.utilHandler.Tools()...)

	// Subagents: each is registered as its own named tool that dispatches via runSubagent.
	for i := range m.subagents {
		sa := m.subagents[i] // local copy for closure capture
		all = append(all, toolutil.Builtin{
			Def: ai.ToolDefinition{
				Name:        sa.name,
				Description: sa.description,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"context": map[string]any{
							"type":        "string",
							"description": "The plan, code, or content to review — provide full context so the reviewer can give useful feedback.",
						},
					},
					"required": []string{"context"},
				},
			},
			Handle: func(ctx context.Context, args map[string]any) (string, error) {
				return m.runSubagent(ctx, &sa, args)
			},
		})
	}

	// MCP server connection (only when servers are configured).
	// Rebuild the server handler so it picks up the current mcpMgr state.
	m.serverHandler = servertools.NewHandler(m.mcpMgr)
	all = append(all, m.serverHandler.Tools()...)

	// Documentation lookup (built-in, always available).
	all = append(all, toolutil.Builtin{
		Def: ai.ToolDefinition{
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
		Handle: func(_ context.Context, args map[string]any) (string, error) {
			topic, _ := args["topic"].(string)
			return docs.Search(topic), nil
		},
	})

	// Convert toolutil.Builtin → internal builtin.
	builtins := make([]builtin, len(all))
	for i, b := range all {
		builtins[i] = builtin{
			def:             b.Def,
			handle:          b.Handle,
			handleWithParts: b.HandleWithParts,
		}
	}
	return builtins
}

// --- Built-in handlers ---

// combinedWriter multiplexes writes to both a stream-specific buffer and a
// combined buffer. The combined buffer is protected by a separate mutex so

// --- File tool handlers ---

// ensure Manager implements chat.ToolManager.
var _ chat.ToolManager = (*Manager)(nil)

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
	content := toolutil.StringArg(args, "context")
	if content == "" {
		return "error: subagent " + sa.name + " requires a non-empty context", nil
	}
	userContent := content
	if focus := toolutil.StringArg(args, "focus"); focus != "" {
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

// toFloat64 coerces a JSON-decoded number value to float64.

// ─── Subpackage Context interface implementations ─────────────────────────────
// Manager implements the narrow Context interfaces required by each handler
// subpackage.  These methods delegate to the Manager's internal state, keeping
// subpackages decoupled from the concrete Manager type.

// filetools.Context
func (m *Manager) ActiveApprover(ctx context.Context) filetools.PathApprover {
	return m.activeApprover(ctx)
}

// processtools.Context
func (m *Manager) ProcessRegistry() *process.Manager { return m.processes }
func (m *Manager) CheckWritePaths(ctx context.Context, command string, args []string, cwd string) []toolutil.PathAccessRequest {
	return m.checkWritePaths(ctx, command, args, cwd)
}
func (m *Manager) CheckShellCommandWritePaths(ctx context.Context, shell, command, cwd string) []toolutil.PathAccessRequest {
	return m.checkShellCommandWritePaths(ctx, shell, command, cwd)
}
func (m *Manager) LiveOutputCh(ctx context.Context) chan<- string { return liveOutputFromCtx(ctx) }

// usertools.Context
func (m *Manager) UserAsker() usertools.Asker { return usertools.Asker(m.userAsker) }
func (m *Manager) PlanConfirmer() usertools.PlanConfirmer {
	return usertools.PlanConfirmer(m.planConfirmer)
}

// tasktools.Context
func (m *Manager) TodoStore() *todostore.Store { return m.todoStore }
func (m *Manager) NotifyTaskTitle(title string) {
	if m.taskTitle != nil {
		m.taskTitle(title)
	}
}

// memorytools.Context
func (m *Manager) MemoryStore() *memorystore.MemoryStore { return m.memoryStore }

// agenttools.Context
func (m *Manager) Skills() []skills.Skill { return m.skills }
func (m *Manager) Agents() []agents.Agent { return m.agents }
func (m *Manager) NotifyAgentActivate(name string) {
	if m.agentActivate != nil {
		m.agentActivate(name)
	}
}

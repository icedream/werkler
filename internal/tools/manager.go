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
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
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

// UserAsker is implemented by interactive callers (e.g. the TUI) to suspend a
// tool goroutine and wait for user input. It is called from a tool execution
// goroutine and blocks until the user responds or ctx is cancelled.
type UserAsker func(ctx context.Context, question string, choices []string, recommendedChoice string, allowFreeform bool) (string, error)

// Manager wraps a chat.ToolManager and adds built-in tools.
type Manager struct {
	wrapped       chat.ToolManager
	processes     *process.Manager
	pathApprover  PathApprover
	builtins      []builtin
	userAsker     UserAsker
	reviewer      ai.Completer
	reviewerLabel string
	skills        []skills.Skill
	todoStore     *todostore.Store
	memoryStore   *memorystore.MemoryStore
}

type builtin struct {
	def    ai.ToolDefinition
	handle func(ctx context.Context, args map[string]any) (string, error)
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
	m.processes = process.New(fn)
}

// SetSkills provides the list of skills available via the use_skill built-in.
// Rebuilds the built-in tool list. Must be called at setup time only (not concurrency-safe).
func (m *Manager) SetSkills(s []skills.Skill) {
	m.skills = s
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

// Session (which wraps this Manager) to serve as its own path approver without
// a circular dependency at construction time.
func (m *Manager) SetPathApprover(pa PathApprover) {
	m.pathApprover = pa
}

// SetUserAsker provides the callback used by the ask_user built-in to request
// user input interactively. When nil (the default), ask_user returns a static
// "not available in non-interactive mode" message.
func (m *Manager) SetUserAsker(fn UserAsker) { m.userAsker = fn }

// SetReviewer provides an optional secondary AI model for rubber duck reviews.
// Calling this rebuilds the built-in tool list to include rubber_duck_review
// when c is non-nil, or remove it when nil. Must be called before the tool
// manager is in use (setup time only — not concurrency-safe).
func (m *Manager) SetReviewer(c ai.Completer, label string) {
	m.reviewer = c
	m.reviewerLabel = label
	m.builtins = m.makeBuiltins()
}

// Processes returns the underlying process.Manager for inspection by the TUI.
func (m *Manager) Processes() *process.Manager { return m.processes }

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

// Tools returns the union of MCP tools and built-in tool definitions.
func (m *Manager) Tools(ctx context.Context) ([]ai.ToolDefinition, error) {
	mcpTools, err := m.wrapped.Tools(ctx)
	if err != nil {
		return nil, err
	}
	defs := make([]ai.ToolDefinition, 0, len(mcpTools)+len(m.builtins))
	defs = append(defs, mcpTools...)
	for _, b := range m.builtins {
		defs = append(defs, b.def)
	}
	return defs, nil
}

// CallTool dispatches to a built-in or the wrapped MCP manager.
// For process-starting tools, path permissions are checked first and
// *UnapprovedPathsError is returned if any paths lack approval.
func (m *Manager) CallTool(ctx context.Context, name string, args map[string]any) (string, error) {
	for _, b := range m.builtins {
		if b.def.Name == name {
			return b.handle(ctx, args)
		}
	}
	return m.wrapped.CallTool(ctx, name, args)
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

	// 2. cwd itself.
	if cwd != "" {
		add(cwd)
	}

	// 3 & 4. Each arg: resolve if relative, scan for embedded paths.
	for _, arg := range args {
		// Resolve relative paths against cwd.
		if isPathLike(arg) {
			add(resolvePath(arg, cwd))
		}
		// Heuristic scan for embedded paths (covers cases like "-I/usr/include").
		for _, m := range pathHeuristic.FindAllStringSubmatch(arg, -1) {
			if len(m) >= 2 {
				p := strings.Trim(m[1], `"'`)
				if !containsEllipsis(p) {
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
				for _, m := range pathHeuristic.FindAllStringSubmatch(script, -1) {
					if len(m) >= 2 {
						p := strings.Trim(m[1], `"'`)
						if !containsEllipsis(p) {
							add(resolvePath(p, cwd))
						}
					}
				}
			}
		}
	}

	return paths
}

// isPathLike reports whether s looks like a standalone filesystem path.
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

// maxFileReadBytes is the maximum file size returned by file_read without a range.
const maxFileReadBytes = 1 << 20 // 1 MiB

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
// paths (cwd, file arguments) require write-level approval.
func (m *Manager) checkWritePaths(command string, args []string, cwd string) []chat.PathAccessRequest {
	if m.pathApprover == nil {
		return nil
	}
	paths := ExtractPaths(command, args, cwd)
	var unapproved []chat.PathAccessRequest
	for i, p := range paths {
		if !m.pathApprover.IsPathWriteApproved(p) {
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
func (m *Manager) checkSingleRead(path string) *UnapprovedPathsError {
	if m.pathApprover == nil {
		return nil
	}
	if !m.pathApprover.IsPathReadApproved(path) {
		return &UnapprovedPathsError{Requests: []chat.PathAccessRequest{{Path: path, Write: false}}}
	}
	return nil
}

// checkSingleWrite returns a non-nil error if path lacks write approval.
func (m *Manager) checkSingleWrite(path string) *UnapprovedPathsError {
	if m.pathApprover == nil {
		return nil
	}
	if !m.pathApprover.IsPathWriteApproved(path) {
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
Use pty=false (default) for non-interactive commands (builds, scripts) — output is clean plain text.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"command":         map[string]any{"type": "string", "description": "Absolute path or PATH-resolvable command name"},
						"args":            map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Command arguments (not including the command itself)"},
						"title":           map[string]any{"type": "string", "description": "Short one-line description of what this command does and why, shown to the user"},
						"cwd":             map[string]any{"type": "string", "description": "Working directory; empty = inherit werkler's cwd"},
						"pty":             map[string]any{"type": "boolean", "description": "Allocate a PTY (pseudo-terminal); required for interactive programs"},
						"env":             map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}, "description": "Extra environment variables merged on top of the current environment"},
						"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait for initial output before returning (0 = don't wait, default 5)"},
					},
					"required": []string{"command"},
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
		// --- File tools ---
		{
			def: ai.ToolDefinition{
				Name: "file_read",
				Description: `Read the contents of a file.
Returns content with line numbers (format: "   1│<line>"), total line count, and the range returned.
Line numbers are decorative — do NOT include them when writing or editing file content.
For large files, use start_line and end_line to read sections (1-indexed, inclusive).
Returns an error for binary files; use process_start to handle those.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path":       map[string]any{"type": "string", "description": "Absolute or ~ path to the file"},
						"start_line": map[string]any{"type": "number", "description": "First line to return (1-indexed, default 1)"},
						"end_line":   map[string]any{"type": "number", "description": "Last line to return (1-indexed, default: end of file or 1 MiB limit)"},
					},
					"required": []string{"path"},
				},
			},
			handle: m.handleFileRead,
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
				Description: `Create a new file or overwrite an existing file with the given content.
This is the correct tool to use whenever you need to write a file.
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
				Description: `Replace exactly one occurrence of old_str with new_str in a file.
The match must be exact (including whitespace). Returns an error if old_str appears
zero times (not found) or more than once (ambiguous — include more context).
On success, returns the line number where the replacement was made.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path":    map[string]any{"type": "string", "description": "Absolute or ~ path to the file"},
						"old_str": map[string]any{"type": "string", "description": "Exact text to find and replace"},
						"new_str": map[string]any{"type": "string", "description": "Replacement text"},
					},
					"required": []string{"path", "old_str", "new_str"},
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
Provide predefined choices when the answer is one of a known set.
Set allow_freeform to false to restrict the user strictly to those choices.
Set recommended_choice to highlight a suggested option.`,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"question": map[string]any{
							"type":        "string",
							"description": "The question to present to the user",
						},
						"choices": map[string]any{
							"type":        "array",
							"items":       map[string]any{"type": "string"},
							"description": "Predefined answer choices",
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

	if m.reviewer != nil {
		reviewerDesc := fmt.Sprintf(
			"Submit a plan, code, or reasoning to a separate reviewer AI for critical feedback.\n"+
				"Reviewer: %s\n"+
				"Use before implementing something non-trivial to catch bugs, logic errors, or design flaws early.\n"+
				"Provide complete context so the reviewer can give useful feedback.",
			m.reviewerLabel,
		)
		builtins = append(builtins, builtin{
			def: ai.ToolDefinition{
				Name:        "rubber_duck_review",
				Description: reviewerDesc,
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"context": map[string]any{
							"type":        "string",
							"description": "The plan, code, or reasoning to review",
						},
						"focus": map[string]any{
							"type":        "string",
							"description": `Optional: specific aspects to focus on (e.g. "concurrency safety", "error handling")`,
						},
					},
					"required": []string{"context"},
				},
			},
			handle: m.handleRubberDuck,
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

	if m.todoStore != nil {
		builtins = append(builtins,
			builtin{
				def: ai.ToolDefinition{
					Name: "todo_add",
					Description: `Add a todo item to the session task list.
Use proactively at the start of multi-step tasks. Returns the todo ID for later updates.`,
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
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
					Name: "memory_read",
					Description: `Read the project memory for the current directory.
Returns notes saved by memory_write in previous sessions about this project.
The current memory is also automatically injected into your system prompt at session start,
so you rarely need to call this — only if you want to re-read it mid-session.`,
					InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
				},
				handle: m.handleMemoryRead,
			},
			builtin{
				def: ai.ToolDefinition{
					Name: "memory_write",
					Description: `Write (replace) the project memory for the current directory.
IMPORTANT: This tool REPLACES the entire memory file. You must include the existing notes
plus any new additions — partial writes will erase earlier content.
Use this to persist project knowledge across sessions: conventions, architecture decisions,
known issues, preferred patterns, important file locations.
Keep entries concise. Maximum ` + fmt.Sprintf("%d", memorystore.MaxBytes) + ` bytes total.`,
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"content": map[string]any{
								"type":        "string",
								"description": "Full markdown content to store as the project memory (replaces previous content)",
							},
						},
						"required": []string{"content"},
					},
				},
				handle: m.handleMemoryWrite,
			},
		)
	}

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

func (m *Manager) handleProcessStart(ctx context.Context, args map[string]any) (string, error) {
	command := stringArg(args, "command")
	if command == "" {
		return "", fmt.Errorf("process_start: command is required")
	}
	cmdArgs := stringSliceArg(args, "args")
	cwd := stringArg(args, "cwd")
	usePTY := boolArg(args, "pty")
	env := stringMapArg(args, "env")
	timeoutSecs := float64Arg(args, "timeout_seconds", 5)

	// Check path permissions before starting.
	if unapproved := m.checkWritePaths(command, cmdArgs, cwd); len(unapproved) > 0 {
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

// --- File tool handlers ---

func (m *Manager) handleFileRead(_ context.Context, args map[string]any) (string, error) {
	rawPath := stringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_read: path is required")
	}
	path := canonicalizePath(rawPath)

	if err := m.checkSingleRead(path); err != nil {
		return "", err
	}

	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("file_read: %w", err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("file_read: %s is a directory; use file_list to list directories", path)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("file_read: %w", err)
	}

	if !utf8.Valid(data) {
		return "", fmt.Errorf("file_read: %s appears to be a binary file; use process_start to handle binary files", path)
	}

	lines := strings.Split(string(data), "\n")
	totalLines := len(lines)

	startLine := int(float64Arg(args, "start_line", 1))
	endLine := int(float64Arg(args, "end_line", float64(totalLines)))
	if startLine < 1 {
		startLine = 1
	}
	if endLine > totalLines {
		endLine = totalLines
	}
	if startLine > endLine {
		return "", fmt.Errorf("file_read: start_line (%d) > end_line (%d)", startLine, endLine)
	}

	selected := lines[startLine-1 : endLine]

	// Enforce size limit when reading without a range (full file).
	if _, hasStart := args["start_line"]; !hasStart {
		if _, hasEnd := args["end_line"]; !hasEnd {
			if len(data) > maxFileReadBytes {
				// Truncate to fit within maxFileReadBytes; find the last safe line.
				size := 0
				for i, l := range selected {
					size += len(l) + 1
					if size > maxFileReadBytes {
						selected = selected[:i]
						break
					}
				}
			}
		}
	}

	// Format with line numbers. Use a │ separator that is visually distinct
	// from file content so models don't copy the line-number prefix verbatim.
	var sb strings.Builder
	for i, l := range selected {
		fmt.Fprintf(&sb, "%4d│%s\n", startLine+i, l)
	}

	return jsonResult(map[string]any{
		"content":     sb.String(),
		"total_lines": totalLines,
		"start_line":  startLine,
		"end_line":    startLine + len(selected) - 1,
		"truncated":   startLine+len(selected)-1 < totalLines && len(data) > maxFileReadBytes,
	}), nil
}

func (m *Manager) handleFileList(_ context.Context, args map[string]any) (string, error) {
	rawPath := stringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_list: path is required")
	}
	path := canonicalizePath(rawPath)

	if err := m.checkSingleRead(path); err != nil {
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

func (m *Manager) handleFileWrite(_ context.Context, args map[string]any) (string, error) {
	rawPath := stringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_write: path is required")
	}
	path := canonicalizePath(rawPath)
	content := stringArg(args, "content")
	createParents := boolArg(args, "create_parents")

	if err := m.checkSingleWrite(path); err != nil {
		return "", err
	}

	if createParents {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return "", fmt.Errorf("file_write: creating parent directories: %w", err)
		}
	}

	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("file_write: %w", err)
	}

	return jsonResult(map[string]any{
		"path":  path,
		"bytes": len(content),
	}), nil
}

func (m *Manager) handleFileEdit(_ context.Context, args map[string]any) (string, error) {
	rawPath := stringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_edit: path is required")
	}
	path := canonicalizePath(rawPath)
	oldStr := stringArg(args, "old_str")
	newStr := stringArg(args, "new_str")

	if oldStr == "" {
		return "", fmt.Errorf("file_edit: old_str must not be empty; use file_write to overwrite entire files")
	}

	if err := m.checkSingleWrite(path); err != nil {
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
	count := strings.Count(content, oldStr)
	switch {
	case count == 0:
		return "", fmt.Errorf("file_edit: old_str not found in %s — call file_read on the file first and use the exact text from the file as old_str (watch for whitespace and indentation)", path)
	case count > 1:
		// Give the AI enough context to narrow down the match.
		lineNums := findMatchLines(content, oldStr)
		return "", fmt.Errorf("file_edit: old_str matches %d times in %s (at lines %v); include more surrounding context to make it unique",
			count, path, lineNums)
	}

	newContent := strings.Replace(content, oldStr, newStr, 1)
	if err := os.WriteFile(path, []byte(newContent), 0o644); err != nil {
		return "", fmt.Errorf("file_edit: writing %s: %w", path, err)
	}

	// Report the line where the replacement occurred and line-count delta.
	idx := strings.Index(content, oldStr)
	line := strings.Count(content[:idx], "\n") + 1
	removed := strings.Count(oldStr, "\n") + 1
	added := strings.Count(newStr, "\n") + 1
	return jsonResult(map[string]any{
		"path":    path,
		"line":    line,
		"added":   added,
		"removed": removed,
	}), nil
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

func (m *Manager) handleFileDelete(_ context.Context, args map[string]any) (string, error) {
	rawPath := stringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_delete: path is required")
	}
	path := canonicalizePath(rawPath)

	if err := m.checkSingleWrite(path); err != nil {
		return "", err
	}

	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("file_delete: %w", err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("file_delete: %s is a directory; use process_start with rm -r to delete directories", path)
	}

	if err := os.Remove(path); err != nil {
		return "", fmt.Errorf("file_delete: %w", err)
	}
	return jsonResult(map[string]any{"deleted": path}), nil
}

// ensure Manager implements chat.ToolManager.
var _ chat.ToolManager = (*Manager)(nil)

// ensure slices is used (imported for future dedup use).
var _ = slices.Contains[[]string, string]

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

// --- Rubber duck handler ---

// rubberDuckSystemPrompt instructs the reviewer AI to give concise, high-signal feedback.
const rubberDuckSystemPrompt = `You are a technical reviewer. Critically evaluate the plan, code, or reasoning you are given.
Identify: correctness issues, bugs, logic errors, edge cases, security concerns, design flaws.
Be concise. Only surface issues that genuinely matter.
If you find no significant issues, say so briefly.
Do NOT comment on style, formatting, naming conventions, or other minor matters.`

func (m *Manager) handleRubberDuck(ctx context.Context, args map[string]any) (string, error) {
	plan := stringArg(args, "context")
	if plan == "" {
		return "error: rubber_duck_review requires a non-empty context", nil
	}
	userContent := plan
	if focus := stringArg(args, "focus"); focus != "" {
		userContent += "\n\nFocus particularly on: " + focus
	}
	messages := []ai.Message{
		{Role: "system", Content: rubberDuckSystemPrompt},
		{Role: "user", Content: userContent},
	}
	msg, err := m.reviewer.Complete(ctx, messages, nil)
	if err != nil {
		return "", fmt.Errorf("rubber duck review failed: %w", err)
	}
	return msg.Content, nil
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

func (m *Manager) handleTodoAdd(_ context.Context, args map[string]any) (string, error) {
	title := stringArg(args, "title")
	if title == "" {
		return "error: title is required", nil
	}
	id := m.todoStore.Add(title, stringArg(args, "description"))
	return fmt.Sprintf("todo %s added", id), nil
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
	return "todo " + id + " updated", nil
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

func (m *Manager) handleTaskComplete(_ context.Context, args map[string]any) (string, error) {
	summary := stringArg(args, "summary")
	if summary == "" {
		summary = "Task complete."
	}
	return summary, nil
}

func (m *Manager) handleMemoryRead(_ context.Context, _ map[string]any) (string, error) {
	content, err := m.memoryStore.Read()
	if err != nil {
		return "error reading project memory: " + err.Error(), nil
	}
	if content == "" {
		return "(no project memory stored yet)", nil
	}
	return content, nil
}

func (m *Manager) handleMemoryWrite(_ context.Context, args map[string]any) (string, error) {
	content := stringArg(args, "content")
	if err := m.memoryStore.Write(content); err != nil {
		return "error writing project memory: " + err.Error(), nil
	}
	return fmt.Sprintf("project memory saved (%d bytes)", len(content)), nil
}

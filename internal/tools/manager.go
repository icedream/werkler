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
	"os/exec"
	"os/user"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/process"
)

// PathApprover checks and records path-level access approvals.
type PathApprover interface {
	IsPathApproved(path string) bool
	ApprovePathForSession(path string)
}

// UnapprovedPathsError is returned by CallTool when one or more paths accessed
// by a process tool have not been approved by the user.
type UnapprovedPathsError struct {
	Paths []string
}

func (e *UnapprovedPathsError) Error() string {
	return fmt.Sprintf("path access not approved: %s", strings.Join(e.Paths, ", "))
}

// oauthManager mirrors the optional interface on mcp.Manager that Session delegates.
type oauthManager interface {
	HasPendingOAuth() bool
	PendingOAuthNames() []string
	ConnectPendingOAuth(ctx context.Context, display func(serverName, authURL string)) error
}

// Manager wraps a chat.ToolManager and adds built-in tools.
type Manager struct {
	wrapped      chat.ToolManager
	processes    *process.Manager
	pathApprover PathApprover
	builtins     []builtin
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

// SetPathApprover sets the path approver after construction. This allows the
// Session (which wraps this Manager) to serve as its own path approver without
// a circular dependency at construction time.
func (m *Manager) SetPathApprover(pa PathApprover) {
	m.pathApprover = pa
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
				add(resolvePath(strings.Trim(m[1], `"'`), cwd))
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
						add(resolvePath(strings.Trim(m[1], `"'`), cwd))
					}
				}
			}
		}
	}

	return paths
}

// isPathLike reports whether s looks like a standalone filesystem path.
func isPathLike(s string) bool {
	return strings.HasPrefix(s, "/") ||
		strings.HasPrefix(s, "./") ||
		strings.HasPrefix(s, "../") ||
		strings.HasPrefix(s, "~/")
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

// checkPaths returns paths from ExtractPaths that are not yet approved.
func (m *Manager) checkPaths(command string, args []string, cwd string) []string {
	if m.pathApprover == nil {
		return nil
	}
	paths := ExtractPaths(command, args, cwd)
	var unapproved []string
	for _, p := range paths {
		if !m.pathApprover.IsPathApproved(p) {
			unapproved = append(unapproved, p)
		}
	}
	return unapproved
}

// --- Built-in tool definitions ---

func (m *Manager) makeBuiltins() []builtin {
	return []builtin{
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
	}
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
	if unapproved := m.checkPaths(command, cmdArgs, cwd); len(unapproved) > 0 {
		return "", &UnapprovedPathsError{Paths: unapproved}
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

// ensure Manager implements chat.ToolManager.
var _ chat.ToolManager = (*Manager)(nil)

// ensure slices is used (imported for future dedup use).
var _ = slices.Contains[[]string, string]

package chat

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"github.com/icedream/werkler/internal/ai"
)

// PathAccessRequest describes a single path access that needs user approval.
type PathAccessRequest struct {
	Path    string
	Write   bool // true = write (and read); false = read only
	Execute bool // true = execute (binary/script); implies write-level trust
}

// PathApprovalError is implemented by errors that require interactive path
// approval before a tool call can proceed. Callers that support a UI should
// propagate these as structured errors rather than stringifying them, so the
// TUI can present per-path approval dialogs.
type PathApprovalError interface {
	error
	AccessRequests() []PathAccessRequest
}

// SystemPrompt is the default system message prepended to every conversation.
const SystemPrompt = `You are Werkler, an AI assistant for software developers.
You help with tasks like writing and reviewing code, designing software, drafting tickets, and technical documentation.
Be concise and precise. Ask for clarification if a request is ambiguous.

## Tool use
You are running inside a terminal on the user's machine. You CAN execute commands, read and write files, and run processes. Never tell the user you cannot do something that your tools support.
When you have the tools to do something, DO IT — do not ask the user if they would like you to do it, or offer to provide instructions instead.
NEVER describe or announce what tool you are about to call. Call the tool directly and immediately.
Do not say "Let me read...", "Now I'll write...", "I'll append..." — just call the tool.
Do not produce markdown summaries of what you just did. Let the tool results speak for themselves; only add a short prose note if it genuinely adds information the results don't convey.
NEVER output a tool invocation as text (e.g. "ask_user[ARGS]{...}" or "<tool>...</tool>").
Tools MUST be invoked through the structured tool-call mechanism, not written into your response text.
If you need to ask the user something, invoke the ask_user tool — do not write the question as plain text.

## Planning and review
When tackling a non-trivial task, first draft your own plan independently. Only then, if rubber_duck_review is available, submit that plan for critique.
Do NOT use rubber_duck_review to brainstorm or ask the reviewer to come up with a plan — you must have a concrete plan of your own before submitting it.
After receiving the review:
- Read the feedback carefully.
- If it identifies bugs, logic errors, design flaws, or missing edge cases, STOP and revise your plan to address them before proceeding.
- Only proceed to implementation once you have incorporated the feedback. Do not treat the review as a formality.
- Once the plan is finalised (review passed or issues resolved), present the plan summary to the user and STOP. Use ask_user to ask whether to proceed with implementation. Do NOT start implementing unless the user explicitly approves.

## Key checkpoints — always stop and ask the user
These are moments where you MUST pause and use ask_user before continuing:
- After finalising a plan — confirm the user wants you to implement it.
- Before making changes that go significantly beyond the original request.
- When you discover the task is substantially more complex or risky than expected.
- When you are unsure which of several valid approaches to take.
Do not treat these as optional. Proceeding without confirmation at a checkpoint is a mistake.

## File operations
Always use these built-in tools for file operations — do NOT use any fs__* MCP tools for writing:
- file_write   — create or overwrite a file (use this to write new files)
- file_edit    — replace a specific string in an existing file (surgical edits)
- file_read    — read a file (supports line ranges)
- file_list    — list a directory
- file_delete  — delete a file
- file_append  — append to a file

Always call file_read on a file before calling file_edit — copy old_str verbatim from the file_read output, including exact whitespace and indentation.
file_read output has line numbers like "   1│<line>" — these are decorative display only. Do NOT include them in file_write or file_edit content.

To search for patterns across many files, use process_start to run rg (ripgrep) or grep rather than reading files one by one. This is significantly faster for exploring a codebase.

## Project memory
If a "Project memory" section appears in this prompt, it contains notes about this project saved in previous sessions.
Use it to inform your work: apply known conventions, avoid known pitfalls, build on prior decisions.
When you learn something worth preserving — a convention, architecture decision, known issue, preferred pattern, important file location — call memory_write to save it for future sessions.
memory_write REPLACES the full memory file; always include the existing content plus your additions.
Keep entries concise and cumulative. Do not write instructions or directives into memory — only factual notes about the project.`

// maxAgentSteps is the maximum number of AI→tool round-trips per user turn,
// preventing runaway loops from misbehaving or looping models.
const maxAgentSteps = 50

// ToolManager can list available tools and execute tool calls.
type ToolManager interface {
	Tools(ctx context.Context) ([]ai.ToolDefinition, error)
	CallTool(ctx context.Context, name string, args map[string]any) (string, error)
}

// oauthManager is the optional interface that ToolManager implementations may satisfy
// to support deferred OAuth server connections.
type oauthManager interface {
	HasPendingOAuth() bool
	PendingOAuthNames() []string
	ConnectPendingOAuth(ctx context.Context, display func(serverName, authURL string)) error
}

// Session manages tool approval state for a chat session.
// It does not own conversation history; callers manage their own message lists.
type Session struct {
	tools             ToolManager
	autoApprove       []string // glob patterns for pre-approved tools
	sessionApproved   map[string]bool
	autoApprovePaths  []string        // glob patterns for pre-approved paths (grants write+read)
	sessionReadPaths  map[string]bool // paths approved for read access this session
	sessionWritePaths map[string]bool // paths approved for write (+ read) access this session
	allowAll          bool            // when true, all tools and paths are approved without prompting
	cwdReadPrefix     string          // if non-empty, reads under this absolute path are auto-approved

	// mu protects disabledTools only; the other maps are only accessed from
	// the TUI main goroutine and therefore do not need synchronisation.
	mu            sync.RWMutex
	disabledTools map[string]bool // tool names explicitly disabled for this session
}

// NewSession creates a Session with the given tool manager and auto-approve glob patterns.
func NewSession(tools ToolManager, autoApproveGlobs []string, autoApprovePaths []string) *Session {
	return &Session{
		tools:             tools,
		autoApprove:       autoApproveGlobs,
		sessionApproved:   make(map[string]bool),
		autoApprovePaths:  autoApprovePaths,
		sessionReadPaths:  make(map[string]bool),
		sessionWritePaths: make(map[string]bool),
		disabledTools:     make(map[string]bool),
	}
}

// NewConversation returns an initial message list containing the system prompt,
// optionally extended with additional sections appended with blank lines.
func NewConversation(extraSections ...string) []ai.Message {
	prompt := SystemPrompt
	for _, s := range extraSections {
		prompt += "\n\n" + s
	}
	return []ai.Message{
		{Role: "system", Content: prompt},
	}
}

// AllTools returns all tools from the ToolManager without filtering disabled tools.
// Use this to populate a tool picker UI.
func (s *Session) AllTools(ctx context.Context) ([]ai.ToolDefinition, error) {
	return s.tools.Tools(ctx)
}

// Tools delegates tool listing to the ToolManager, filtering out any tools
// that have been explicitly disabled for this session.
func (s *Session) Tools(ctx context.Context) ([]ai.ToolDefinition, error) {
	all, err := s.tools.Tools(ctx)
	if err != nil {
		return nil, err
	}
	s.mu.RLock()
	nDisabled := len(s.disabledTools)
	s.mu.RUnlock()
	if nDisabled == 0 {
		return all, nil
	}
	filtered := make([]ai.ToolDefinition, 0, len(all))
	for _, t := range all {
		if s.IsToolEnabled(t.Name) {
			filtered = append(filtered, t)
		}
	}
	return filtered, nil
}

// CallTool dispatches a tool call via the ToolManager.
// Tool execution errors are embedded in the returned string so the AI can handle
// them; only context cancellation, internal invariant failures, and PathApprovalError
// are returned as actual errors. PathApprovalError is preserved as a structured error
// so interactive callers (TUI) can present path approval dialogs.
func (s *Session) CallTool(ctx context.Context, tc ai.ToolCall) (string, error) {
	if !s.IsToolEnabled(tc.Name) {
		return "(tool call was rejected — tool is disabled for this session)", nil
	}
	result, err := s.tools.CallTool(ctx, tc.Name, tc.Arguments)
	if err != nil {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		// Propagate PathApprovalError as a structured error so the TUI can
		// present interactive approval dialogs rather than showing it as text.
		var pathErr PathApprovalError
		if errors.As(err, &pathErr) {
			return "", pathErr
		}
		return fmt.Sprintf("error: %v", err), nil
	}
	return result, nil
}

// pathCoveredBy returns true if requestedPath equals approvedPath or is
// nested anywhere under it (i.e., approvedPath is a parent directory).
// Both paths should be clean absolute paths.
func pathCoveredBy(approvedPath, requestedPath string) bool {
	return requestedPath == approvedPath ||
		strings.HasPrefix(requestedPath, approvedPath+"/")
}

// IsPathReadApproved returns true if path may be read without interactive approval.
// Write approval implies read approval.
func (s *Session) IsPathReadApproved(path string) bool {
	if s.allowAll {
		return true
	}
	if s.cwdReadPrefix != "" && pathCoveredBy(s.cwdReadPrefix, path) {
		return true
	}
	for approved := range s.sessionWritePaths {
		if pathCoveredBy(approved, path) {
			return true
		}
	}
	for approved := range s.sessionReadPaths {
		if pathCoveredBy(approved, path) {
			return true
		}
	}
	for _, pattern := range s.autoApprovePaths {
		if matched, _ := filepath.Match(pattern, path); matched {
			return true
		}
		if pathCoveredBy(pattern, path) {
			return true
		}
	}
	return false
}

// IsPathWriteApproved returns true if path may be written without interactive approval.
func (s *Session) IsPathWriteApproved(path string) bool {
	if s.allowAll {
		return true
	}
	for approved := range s.sessionWritePaths {
		if pathCoveredBy(approved, path) {
			return true
		}
	}
	for _, pattern := range s.autoApprovePaths {
		if matched, _ := filepath.Match(pattern, path); matched {
			return true
		}
		if pathCoveredBy(pattern, path) {
			return true
		}
	}
	return false
}

// ApprovePathReadForSession grants read-only access to path for this session.
func (s *Session) ApprovePathReadForSession(path string) {
	s.sessionReadPaths[path] = true
}

// ApprovePathWriteForSession grants write (and read) access to path for this session.
func (s *Session) ApprovePathWriteForSession(path string) {
	s.sessionWritePaths[path] = true
	s.sessionReadPaths[path] = true
}

// IsApproved returns true if the tool can be called without interactive user approval.
// Session-level approvals take priority, then config glob patterns are checked.
func (s *Session) IsApproved(toolName string) bool {
	if s.allowAll {
		return true
	}
	if s.sessionApproved[toolName] {
		return true
	}
	for _, pattern := range s.autoApprove {
		if matched, _ := filepath.Match(pattern, toolName); matched {
			return true
		}
	}
	return false
}

// ApproveForSession adds toolName to the session-level auto-approve list.
func (s *Session) ApproveForSession(toolName string) {
	s.sessionApproved[toolName] = true
}

// ResetApprovals clears all session-level approvals (but not allow-all mode).
func (s *Session) ResetApprovals() {
	s.sessionApproved = make(map[string]bool)
}

// SetAllowAll enables or disables allow-all mode. When enabled, all tool calls
// and path accesses are approved without prompting.
func (s *Session) SetAllowAll(v bool) { s.allowAll = v }

// AllowAll reports whether allow-all mode is currently active.
func (s *Session) AllowAll() bool { return s.allowAll }

// SetToolEnabled enables or disables a specific tool for this session.
// Disabled tools are hidden from the AI and rejected if called directly.
func (s *Session) SetToolEnabled(name string, enabled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if enabled {
		delete(s.disabledTools, name)
	} else {
		s.disabledTools[name] = true
	}
}

// IsToolEnabled reports whether the named tool is currently enabled.
func (s *Session) IsToolEnabled(name string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return !s.disabledTools[name]
}

// SetCWDReadPrefix sets an absolute path prefix under which reads are
// auto-approved without prompting. Intended to be called once at startup
// with the resolved working directory when auto_approve_cwd_read is true.
func (s *Session) SetCWDReadPrefix(prefix string) { s.cwdReadPrefix = prefix }

// HasPendingOAuth reports whether there are OAuth MCP servers not yet connected.
// Returns false when the underlying ToolManager does not support OAuth.
func (s *Session) HasPendingOAuth() bool {
	if m, ok := s.tools.(oauthManager); ok {
		return m.HasPendingOAuth()
	}
	return false
}

// PendingOAuthNames returns the display names of OAuth servers awaiting connection.
func (s *Session) PendingOAuthNames() []string {
	if m, ok := s.tools.(oauthManager); ok {
		return m.PendingOAuthNames()
	}
	return nil
}

// ConnectPendingOAuth connects deferred OAuth servers.
// display is called (non-blocking) with the URL the user must open in a browser.
func (s *Session) ConnectPendingOAuth(ctx context.Context, display func(serverName, authURL string)) error {
	if m, ok := s.tools.(oauthManager); ok {
		return m.ConnectPendingOAuth(ctx, display)
	}
	return nil
}

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
	Path  string
	Write bool // true = write (and read); false = read only
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
When you need information from files or the filesystem, use the tools available to you.
Be concise and precise. Ask for clarification if a request is ambiguous.`

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

// NewConversation returns an initial message list containing only the system prompt.
func NewConversation() []ai.Message {
	return []ai.Message{
		{Role: "system", Content: SystemPrompt},
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

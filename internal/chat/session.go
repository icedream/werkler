package chat

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/icedream/werkler/internal/ai"
)

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
	tools                ToolManager
	autoApprove          []string // glob patterns for pre-approved tools
	sessionApproved      map[string]bool
	autoApprovePaths     []string // glob patterns for pre-approved paths
	sessionApprovedPaths map[string]bool
}

// NewSession creates a Session with the given tool manager and auto-approve glob patterns.
func NewSession(tools ToolManager, autoApproveGlobs []string, autoApprovePaths []string) *Session {
	return &Session{
		tools:                tools,
		autoApprove:          autoApproveGlobs,
		sessionApproved:      make(map[string]bool),
		autoApprovePaths:     autoApprovePaths,
		sessionApprovedPaths: make(map[string]bool),
	}
}

// NewConversation returns an initial message list containing only the system prompt.
func NewConversation() []ai.Message {
	return []ai.Message{
		{Role: "system", Content: SystemPrompt},
	}
}

// Tools delegates tool listing to the ToolManager.
func (s *Session) Tools(ctx context.Context) ([]ai.ToolDefinition, error) {
	return s.tools.Tools(ctx)
}

// CallTool dispatches a tool call via the ToolManager.
// Tool execution errors are embedded in the returned string so the AI can handle
// them; only context cancellation or internal invariant failures are returned as error.
func (s *Session) CallTool(ctx context.Context, tc ai.ToolCall) (string, error) {
	result, err := s.tools.CallTool(ctx, tc.Name, tc.Arguments)
	if err != nil {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		return fmt.Sprintf("error: %v", err), nil
	}
	return result, nil
}

// IsPathApproved returns true if the path may be accessed without interactive approval.
func (s *Session) IsPathApproved(path string) bool {
	if s.sessionApprovedPaths[path] {
		return true
	}
	for _, pattern := range s.autoApprovePaths {
		if matched, _ := filepath.Match(pattern, path); matched {
			return true
		}
	}
	return false
}

// ApprovePathForSession adds path to the session-level approved set.
func (s *Session) ApprovePathForSession(path string) {
	s.sessionApprovedPaths[path] = true
}

// IsApproved returns true if the tool can be called without interactive user approval.
// Session-level approvals take priority, then config glob patterns are checked.
func (s *Session) IsApproved(toolName string) bool {
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

// ResetApprovals clears all session-level approvals.
func (s *Session) ResetApprovals() {
	s.sessionApproved = make(map[string]bool)
}

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

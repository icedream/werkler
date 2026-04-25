package chat_test

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
)

// --- Mock ToolManager ---

type mockToolManager struct {
	mock.Mock
}

func (m *mockToolManager) Tools(ctx context.Context) ([]ai.ToolDefinition, error) {
	args := m.Called(ctx)
	return args.Get(0).([]ai.ToolDefinition), args.Error(1)
}

func (m *mockToolManager) CallTool(ctx context.Context, name string, toolArgs map[string]any) (string, error) {
	args := m.Called(ctx, name, toolArgs)
	return args.String(0), args.Error(1)
}

// --- IsApproved ---

func TestIsApproved_ExactGlob(t *testing.T) {
	s := chat.NewSession(&mockToolManager{}, []string{"fs__read"}, nil)
	assert.True(t, s.IsApproved("fs__read"))
	assert.False(t, s.IsApproved("fs__write"))
}

func TestIsApproved_WildcardGlob(t *testing.T) {
	s := chat.NewSession(&mockToolManager{}, []string{"fs__*"}, nil)
	assert.True(t, s.IsApproved("fs__read"))
	assert.True(t, s.IsApproved("fs__write_file"))
	assert.False(t, s.IsApproved("git__commit"))
}

func TestIsApproved_MultiplePatterns(t *testing.T) {
	s := chat.NewSession(&mockToolManager{}, []string{"fs__read", "git__*"}, nil)
	assert.True(t, s.IsApproved("fs__read"))
	assert.True(t, s.IsApproved("git__log"))
	assert.False(t, s.IsApproved("fs__write"))
}

func TestIsApproved_NotApproved(t *testing.T) {
	s := chat.NewSession(&mockToolManager{}, nil, nil)
	assert.False(t, s.IsApproved("any_tool"))
}

// --- ApproveForSession ---

func TestApproveForSession_ApprovesExactName(t *testing.T) {
	s := chat.NewSession(&mockToolManager{}, nil, nil)
	assert.False(t, s.IsApproved("my_tool"))
	s.ApproveForSession("my_tool")
	assert.True(t, s.IsApproved("my_tool"))
}

func TestApproveForSession_DoesNotApproveOtherTools(t *testing.T) {
	s := chat.NewSession(&mockToolManager{}, nil, nil)
	s.ApproveForSession("tool_a")
	assert.False(t, s.IsApproved("tool_b"))
}

// --- ResetApprovals ---

func TestResetApprovals_ClearsSessionApprovals(t *testing.T) {
	s := chat.NewSession(&mockToolManager{}, nil, nil)
	s.ApproveForSession("my_tool")
	require.True(t, s.IsApproved("my_tool"))
	s.ResetApprovals()
	assert.False(t, s.IsApproved("my_tool"))
}

func TestResetApprovals_PreservesAutoApproveGlobs(t *testing.T) {
	s := chat.NewSession(&mockToolManager{}, []string{"fs__*"}, nil)
	s.ResetApprovals()
	// Glob patterns should still work after reset.
	assert.True(t, s.IsApproved("fs__read"))
}

// --- CallTool ---

func TestCallTool_Success(t *testing.T) {
	tm := &mockToolManager{}
	tm.On("CallTool", mock.Anything, "fs__read", map[string]any{"path": "/tmp"}).
		Return("file contents", nil)

	s := chat.NewSession(tm, nil, nil)
	result, err := s.CallTool(context.Background(), ai.ToolCall{
		ID:        "c1",
		Name:      "fs__read",
		Arguments: map[string]any{"path": "/tmp"},
	})
	require.NoError(t, err)
	assert.Equal(t, "file contents", result)
	tm.AssertExpectations(t)
}

func TestCallTool_ToolError_EmbeddedAsString(t *testing.T) {
	// Tool errors should be returned as a string (not a Go error) so the AI can handle them.
	tm := &mockToolManager{}
	tm.On("CallTool", mock.Anything, "bad_tool", mock.Anything).
		Return("", errors.New("permission denied"))

	s := chat.NewSession(tm, nil, nil)
	result, err := s.CallTool(context.Background(), ai.ToolCall{
		ID:   "c1",
		Name: "bad_tool",
	})
	require.NoError(t, err)
	assert.Contains(t, result, "permission denied")
	tm.AssertExpectations(t)
}

func TestCallTool_ContextCancel_ReturnsGoError(t *testing.T) {
	// When the context is cancelled, a Go error must be returned (not embedded).
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled

	tm := &mockToolManager{}
	tm.On("CallTool", mock.Anything, "slow_tool", mock.Anything).
		Return("", context.Canceled)

	s := chat.NewSession(tm, nil, nil)
	_, err := s.CallTool(ctx, ai.ToolCall{ID: "c1", Name: "slow_tool"})
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	tm.AssertExpectations(t)
}

// --- NewConversation ---

func TestNewConversation_StartsWithSystem(t *testing.T) {
	msgs := chat.NewConversation()
	require.Len(t, msgs, 1)
	assert.Equal(t, "system", msgs[0].Role)
	assert.NotEmpty(t, msgs[0].Content)
}

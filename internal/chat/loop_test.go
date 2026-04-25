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

// --- Mock Completer ---

type mockCompleter struct {
	mock.Mock
}

func (m *mockCompleter) Complete(ctx context.Context, messages []ai.Message, tools []ai.ToolDefinition) (ai.Message, error) {
	args := m.Called(ctx, messages, tools)
	return args.Get(0).(ai.Message), args.Error(1)
}

// newTestSession returns a Session with the given tool manager and no auto-approved tools,
// paired with the same mockToolManager instance for assertion.
func newTestSession(tm *mockToolManager, autoApprove []string) *chat.Session {
	return chat.NewSession(tm, autoApprove)
}

// --- RunPrompt ---

func TestRunPrompt_SimpleResponse(t *testing.T) {
	tm := &mockToolManager{}
	tm.On("Tools", mock.Anything).Return([]ai.ToolDefinition{}, nil)

	c := &mockCompleter{}
	c.On("Complete", mock.Anything, mock.MatchedBy(func(msgs []ai.Message) bool {
		// Should contain system + user message.
		return len(msgs) == 2 && msgs[1].Role == "user"
	}), mock.Anything).Return(ai.Message{Role: "assistant", Content: "pong"}, nil)

	s := newTestSession(tm, nil)
	result, err := chat.RunPrompt(context.Background(), c, s, "ping", nil)
	require.NoError(t, err)
	assert.Equal(t, "pong", result)
	c.AssertExpectations(t)
}

func TestRunPrompt_AIError(t *testing.T) {
	tm := &mockToolManager{}
	tm.On("Tools", mock.Anything).Return([]ai.ToolDefinition{}, nil)

	c := &mockCompleter{}
	c.On("Complete", mock.Anything, mock.Anything, mock.Anything).
		Return(ai.Message{}, errors.New("api error"))

	s := newTestSession(tm, nil)
	_, err := chat.RunPrompt(context.Background(), c, s, "hello", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "api error")
}

func TestRunPrompt_ApprovedToolCall(t *testing.T) {
	tm := &mockToolManager{}
	tm.On("Tools", mock.Anything).Return([]ai.ToolDefinition{}, nil)
	tm.On("CallTool", mock.Anything, "fs__read", map[string]any{"path": "/tmp/f"}).
		Return("file contents", nil)

	c := &mockCompleter{}
	// First call: returns a tool call.
	c.On("Complete", mock.Anything, mock.MatchedBy(func(msgs []ai.Message) bool {
		return len(msgs) == 2
	}), mock.Anything).Return(ai.Message{
		Role: "assistant",
		ToolCalls: []ai.ToolCall{
			{ID: "c1", Name: "fs__read", Arguments: map[string]any{"path": "/tmp/f"}},
		},
	}, nil).Once()
	// Second call (after tool result): returns final text.
	c.On("Complete", mock.Anything, mock.MatchedBy(func(msgs []ai.Message) bool {
		return len(msgs) == 4 // system + user + assistant + tool result
	}), mock.Anything).Return(ai.Message{Role: "assistant", Content: "done"}, nil).Once()

	s := newTestSession(tm, []string{"fs__read"})
	result, err := chat.RunPrompt(context.Background(), c, s, "read the file", nil)
	require.NoError(t, err)
	assert.Equal(t, "done", result)
	c.AssertExpectations(t)
	tm.AssertExpectations(t)
}

func TestRunPrompt_UnapprovedToolCall_DenialFedBack(t *testing.T) {
	tm := &mockToolManager{}
	tm.On("Tools", mock.Anything).Return([]ai.ToolDefinition{}, nil)
	// CallTool must NOT be called for unapproved tools.

	c := &mockCompleter{}
	// First call: requests an unapproved tool.
	c.On("Complete", mock.Anything, mock.MatchedBy(func(msgs []ai.Message) bool {
		return len(msgs) == 2
	}), mock.Anything).Return(ai.Message{
		Role:      "assistant",
		ToolCalls: []ai.ToolCall{{ID: "c1", Name: "dangerous_tool"}},
	}, nil).Once()
	// Second call: receives the denial message and produces a final response.
	c.On("Complete", mock.Anything, mock.MatchedBy(func(msgs []ai.Message) bool {
		// The last message should be the denial tool result.
		last := msgs[len(msgs)-1]
		return last.Role == "tool" && last.ToolCallID == "c1"
	}), mock.Anything).Return(ai.Message{Role: "assistant", Content: "ok sorry"}, nil).Once()

	s := newTestSession(tm, nil) // no auto-approve
	result, err := chat.RunPrompt(context.Background(), c, s, "do something dangerous", nil)
	require.NoError(t, err)
	assert.Equal(t, "ok sorry", result)
	c.AssertExpectations(t)
	tm.AssertNumberOfCalls(t, "CallTool", 0)
}

func TestRunPrompt_MaxStepsExceeded(t *testing.T) {
	tm := &mockToolManager{}
	tm.On("Tools", mock.Anything).Return([]ai.ToolDefinition{}, nil)
	tm.On("CallTool", mock.Anything, "loop_tool", mock.Anything).
		Return("looping", nil)

	c := &mockCompleter{}
	// Always returns a tool call to simulate a looping agent.
	c.On("Complete", mock.Anything, mock.Anything, mock.Anything).
		Return(ai.Message{
			Role:      "assistant",
			ToolCalls: []ai.ToolCall{{ID: "c1", Name: "loop_tool"}},
		}, nil)

	s := newTestSession(tm, []string{"loop_tool"})
	_, err := chat.RunPrompt(context.Background(), c, s, "loop forever", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exceeded")
}

func TestRunPrompt_ContextCancel(t *testing.T) {
	tm := &mockToolManager{}
	tm.On("Tools", mock.Anything).Return([]ai.ToolDefinition{}, nil)

	ctx, cancel := context.WithCancel(context.Background())

	c := &mockCompleter{}
	c.On("Complete", mock.Anything, mock.Anything, mock.Anything).
		Run(func(_ mock.Arguments) { cancel() }).
		Return(ai.Message{}, context.Canceled)

	s := newTestSession(tm, nil)
	_, err := chat.RunPrompt(ctx, c, s, "cancel me", nil)
	require.Error(t, err)
}

func TestRunPrompt_ToolsError(t *testing.T) {
	tm := &mockToolManager{}
	tm.On("Tools", mock.Anything).Return([]ai.ToolDefinition{}, errors.New("server down"))

	s := newTestSession(tm, nil)
	_, err := chat.RunPrompt(context.Background(), &mockCompleter{}, s, "hi", nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "server down")
}

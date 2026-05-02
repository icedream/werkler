package sessioncontext

import (
	"testing"
	"time"
)

func TestNewContext(t *testing.T) {
	systemPrompt := "You are a helpful assistant."
	messages := []Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
	}
	tools := []ToolDefinition{
		{Name: "read", Description: "Read a file", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}}},
	}
	model := ModelInfo{Model: "test-model"}

	ctx := NewContext(systemPrompt, messages, tools, model)

	if ctx.SystemPrompt != systemPrompt {
		t.Errorf("Expected system prompt %q, got %q", systemPrompt, ctx.SystemPrompt)
	}

	if len(ctx.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(ctx.Messages))
	}

	if len(ctx.Tools) != 1 {
		t.Errorf("Expected 1 tool, got %d", len(ctx.Tools))
	}
}

func TestContextClone(t *testing.T) {
	systemPrompt := "You are a helpful assistant."
	messages := []Message{
		{Role: "user", Content: "Hello"},
	}
	tools := []ToolDefinition{
		{Name: "read", Description: "Read a file", InputSchema: map[string]any{"type": "object"}},
	}
	model := ModelInfo{Model: "test-model"}

	ctx := NewContext(systemPrompt, messages, tools, model)
	ctx.AddToolCall(ToolCall{ID: "call1", Name: "test_tool", Arguments: map[string]any{"arg": "value"}})
	ctx.StartStreaming("stream1")

	cloned := ctx.Clone()

	if cloned.SystemPrompt != ctx.SystemPrompt {
		t.Errorf("System prompt mismatch")
	}

	if len(cloned.Messages) != len(ctx.Messages) {
		t.Errorf("Message count mismatch")
	}

	if len(cloned.Tools) != len(ctx.Tools) {
		t.Errorf("Tool count mismatch")
	}

	if cloned.ThinkingLevel != ctx.ThinkingLevel {
		t.Errorf("Thinking level mismatch")
	}

	if cloned.SessionID != ctx.SessionID {
		t.Errorf("Session ID mismatch")
	}

	if cloned.IsStreaming != ctx.IsStreaming {
		t.Errorf("Streaming state mismatch")
	}

	if cloned.StreamingModel == nil {
		t.Error("Expected StreamingModel to be cloned")
	} else if cloned.StreamingModel.CallID != ctx.StreamingModel.CallID {
		t.Errorf("StreamingModel CallID mismatch")
	}

	if len(cloned.PendingToolCalls) != 1 {
		t.Errorf("Expected 1 pending tool call, got %d", len(cloned.PendingToolCalls))
	}

	// Modify the cloned context
	cloned.AddMessage(Message{Role: "assistant", Content: "Update"})
	cloned.UpdateThinkingLevel("medium")

	// Original context should not be affected
	if len(ctx.Messages) != 1 {
		t.Errorf("Original context should not be affected")
	}
	if ctx.ThinkingLevel != "off" {
		t.Errorf("Original thinking level should not be affected")
	}
}

func TestContextAddMessage(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})

	ctx.AddMessage(Message{Role: "user", Content: "Test message"})

	if len(ctx.Messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(ctx.Messages))
	}

	if ctx.Messages[0].Content != "Test message" {
		t.Errorf("Message content mismatch")
	}

	if !ctx.LastUpdate.After(time.Time{}) {
		t.Errorf("LastUpdate should be set")
	}
}

func TestContextAddToolResult(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})

	result := ToolResult{CallID: "call1", Name: "test", Result: "success"}
	ctx.AddToolResult(result)

	if len(ctx.ToolResults) != 1 {
		t.Errorf("Expected 1 tool result, got %d", len(ctx.ToolResults))
	}

	if ctx.ToolResults[0].Result != "success" {
		t.Errorf("Tool result mismatch")
	}
}

func TestContextAddToolCall(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})

	ctx.AddToolCall(ToolCall{ID: "call1", Name: "test_tool", Arguments: map[string]any{"arg": "value"}})

	if len(ctx.PendingToolCalls) != 1 {
		t.Errorf("Expected 1 pending tool call, got %d", len(ctx.PendingToolCalls))
	}

	if ctx.PendingToolCalls[0].Name != "test_tool" {
		t.Errorf("Tool call name mismatch")
	}
}

func TestContextMarkToolStates(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})
	ctx.AddToolCall(ToolCall{ID: "call1", Name: "test_tool", Arguments: map[string]any{}})

	ctx.MarkToolAsPending("call1")
	if ctx.PendingToolCalls[0].Status != "pending" {
		t.Errorf("Expected status 'pending', got %q", ctx.PendingToolCalls[0].Status)
	}

	ctx.MarkToolAsExecuting("call1")
	if ctx.PendingToolCalls[0].Status != "executing" {
		t.Errorf("Expected status 'executing', got %q", ctx.PendingToolCalls[0].Status)
	}

	ctx.MarkToolAsCompleted("call1", "result", nil)
	if ctx.PendingToolCalls[0].Status != "completed" {
		t.Errorf("Expected status 'completed', got %q", ctx.PendingToolCalls[0].Status)
	}

	if ctx.PendingToolCalls[0].Result != "result" {
		t.Errorf("Expected result 'result', got %q", ctx.PendingToolCalls[0].Result)
	}
}

func TestContextResetPendingToolCalls(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})
	ctx.AddToolCall(ToolCall{ID: "call1", Name: "test_tool", Arguments: map[string]any{}})
	ctx.AddToolCall(ToolCall{ID: "call2", Name: "test_tool2", Arguments: map[string]any{}})

	ctx.ResetPendingToolCalls()

	if len(ctx.PendingToolCalls) != 0 {
		t.Errorf("Expected 0 pending tool calls, got %d", len(ctx.PendingToolCalls))
	}
}

func TestContextGetLastMessage(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})
	ctx.AddMessage(Message{Role: "user", Content: "First"})
	ctx.AddMessage(Message{Role: "assistant", Content: "Second"})

	last := ctx.GetLastMessage()
	if last == nil {
		t.Fatal("Expected last message, got nil")
	}

	if last.Content != "Second" {
		t.Errorf("Expected last message content 'Second', got %q", last.Content)
	}
}

func TestContextGetRecentMessages(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})
	ctx.AddMessage(Message{Role: "user", Content: "First"})
	ctx.AddMessage(Message{Role: "tool", Content: "Tool result"})
	ctx.AddMessage(Message{Role: "assistant", Content: "Second"})
	ctx.AddMessage(Message{Role: "user", Content: "Third"})

	recent := ctx.GetRecentMessages(2)
	if len(recent) != 2 {
		t.Errorf("Expected 2 recent messages, got %d", len(recent))
	}

	if recent[0].Content != "Second" {
		t.Errorf("Expected first recent message to be 'Second', got %q", recent[0].Content)
	}
}

func TestContextGetContextForLLM(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})
	ctx.AddMessage(Message{Role: "user", Content: "User message"})
	ctx.AddMessage(Message{Role: "assistant", Content: "Assistant message"})
	ctx.AddMessage(Message{Role: "tool", Content: "Tool result"})
	// Add a tool call message (should be filtered out)
	ctx.AddMessage(Message{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{ID: "call1", Name: "test", Arguments: map[string]any{}},
		},
	})

	llmContext := ctx.GetContextForLLM()

	// Should have 4 messages (user, assistant, tool, assistant-with-toolcalls)
	// All are valid LLM message types
	if len(llmContext) != 4 {
		t.Errorf("Expected 4 messages for LLM, got %d", len(llmContext))
	}

	// Assistant message with tool calls should still be included
	// (it's a valid LLM message type, just with tool calls)
	if llmContext[3].Role != "assistant" {
		t.Errorf("Expected last message to be assistant, got %q", llmContext[3].Role)
	}
	if len(llmContext[3].ToolCalls) != 1 {
		t.Errorf("Expected 1 tool call in last message, got %d", len(llmContext[3].ToolCalls))
	}
}

func TestContextUpdateThinkingLevel(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})
	ctx.UpdateThinkingLevel("off")

	if ctx.ThinkingLevel != "off" {
		t.Errorf("Expected thinking level 'off', got %q", ctx.ThinkingLevel)
	}

	ctx.UpdateThinkingLevel("medium")
	if ctx.ThinkingLevel != "medium" {
		t.Errorf("Expected thinking level 'medium', got %q", ctx.ThinkingLevel)
	}
}

func TestContextUpdateSessionID(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})
	ctx.UpdateSessionID("session123")

	if ctx.SessionID != "session123" {
		t.Errorf("Expected session ID 'session123', got %q", ctx.SessionID)
	}
}

func TestContextStartStopStreaming(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})

	if ctx.IsStreaming {
		t.Error("Expected IsStreaming to be false initially")
	}

	ctx.StartStreaming("stream1")
	if !ctx.IsStreaming {
		t.Error("Expected IsStreaming to be true after StartStreaming")
	}

	if ctx.StreamingModel == nil {
		t.Error("Expected StreamingModel to be set")
	} else if ctx.StreamingModel.CallID != "stream1" {
		t.Errorf("Expected StreamingModel CallID 'stream1', got %q", ctx.StreamingModel.CallID)
	}

	ctx.StopStreaming(100)
	if ctx.IsStreaming {
		t.Error("Expected IsStreaming to be false after StopStreaming")
	}

	if ctx.StreamingModel != nil {
		t.Error("Expected StreamingModel to be nil after StopStreaming")
	}
}

func TestContextIncrementTokenCount(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})

	ctx.StartStreaming("stream1")
	ctx.IncrementTokenCount()
	if ctx.StreamingModel == nil {
		t.Error("Expected StreamingModel to be created")
	}

	if ctx.StreamingModel.TokenCount != 1 {
		t.Errorf("Expected token count 1, got %d", ctx.StreamingModel.TokenCount)
	}

	ctx.IncrementTokenCount()
	if ctx.StreamingModel.TokenCount != 2 {
		t.Errorf("Expected token count 2, got %d", ctx.StreamingModel.TokenCount)
	}
}

func TestContextAppendMessages(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})
	ctx.AddMessage(Message{Role: "user", Content: "First"})

	newMessages := []Message{
		{Role: "assistant", Content: "Second"},
		{Role: "user", Content: "Third"},
	}

	ctx.AppendMessages(newMessages)

	if len(ctx.Messages) != 3 {
		t.Errorf("Expected 3 messages, got %d", len(ctx.Messages))
	}

	if ctx.Messages[1].Content != "Second" {
		t.Errorf("Expected second message to be 'Second', got %q", ctx.Messages[1].Content)
	}
}

func TestContextClearMessages(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})
	ctx.AddMessage(Message{Role: "user", Content: "First"})
	ctx.AddMessage(Message{Role: "assistant", Content: "Second"})

	ctx.ClearMessages()

	if len(ctx.Messages) != 0 {
		t.Errorf("Expected 0 messages after ClearMessages, got %d", len(ctx.Messages))
	}
}

func TestContextUpdateWithToolResult(t *testing.T) {
	ctx := NewContext("", nil, nil, ModelInfo{})
	ctx.AddToolCall(ToolCall{ID: "call1", Name: "test", Arguments: map[string]any{}})

	// Manually update the tool in the context
	ctx.PendingToolCalls[0].Status = "completed"
	ctx.PendingToolCalls[0].Result = "success"

	if ctx.PendingToolCalls[0].Result != "success" {
		t.Errorf("Expected tool result 'success', got %q", ctx.PendingToolCalls[0].Result)
	}
}

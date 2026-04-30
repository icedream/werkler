package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// sseBody builds a minimal SSE response body from a slice of (event, data) pairs.
// Pass event="" to omit the "event:" line (inline type in the JSON payload).
func sseBody(events [][2]string) string {
	var b strings.Builder
	for _, ev := range events {
		if ev[0] != "" {
			fmt.Fprintf(&b, "event: %s\n", ev[0])
		}
		fmt.Fprintf(&b, "data: %s\n\n", ev[1])
	}
	return b.String()
}

// newTestResponsesClient wires a ResponsesClient to talk to srv.
func newTestResponsesClient(srv *httptest.Server) *ResponsesClient {
	return NewResponsesClient(srv.URL, "test-key", "gpt-4o", srv.Client())
}

// --- toResponsesItems ---

func TestToResponsesItems_SystemBecomesInstructions(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	}
	instructions, items := toResponsesItems(msgs)
	assert.Equal(t, "You are helpful.", instructions)
	require.Len(t, items, 1)
	assert.Equal(t, "message", items[0].Type)
	assert.Equal(t, "user", items[0].Role)
}

func TestToResponsesItems_MultipleSystemsMerged(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "First."},
		{Role: "system", Content: "Second."},
		{Role: "user", Content: "Hi"},
	}
	instructions, items := toResponsesItems(msgs)
	assert.Equal(t, "First.\n\nSecond.", instructions)
	require.Len(t, items, 1)
}

func TestToResponsesItems_AssistantTextAndToolCall(t *testing.T) {
	msgs := []Message{
		{
			Role:    "assistant",
			Content: "Let me check.",
			ToolCalls: []ToolCall{
				{ID: "call_abc", Name: "fs__read", Arguments: map[string]any{"path": "/etc/hosts"}},
			},
		},
	}
	_, items := toResponsesItems(msgs)
	// Expect one message item (text) + one function_call item.
	require.Len(t, items, 2)
	assert.Equal(t, "message", items[0].Type)
	assert.Equal(t, "function_call", items[1].Type)
	assert.Equal(t, "call_abc", items[1].CallID)
	assert.Equal(t, "fs__read", items[1].Name)
	assert.Contains(t, items[1].Arguments, "/etc/hosts")
}

func TestToResponsesItems_ToolResultBecomesOutput(t *testing.T) {
	msgs := []Message{
		{Role: "tool", Content: "file contents here", ToolCallID: "call_abc"},
	}
	_, items := toResponsesItems(msgs)
	require.Len(t, items, 1)
	assert.Equal(t, "function_call_output", items[0].Type)
	assert.Equal(t, "call_abc", items[0].CallID)
	assert.Equal(t, "file contents here", items[0].Output)
}

func TestToResponsesItems_UserWithImageURL(t *testing.T) {
	msgs := []Message{
		{
			Role:    "user",
			Content: "what is this?",
			Parts:   []ImagePart{{URL: "https://example.com/img.png"}},
		},
	}
	_, items := toResponsesItems(msgs)
	require.Len(t, items, 1)
	var parts []respContentPart
	require.NoError(t, json.Unmarshal(items[0].Content, &parts))
	require.Len(t, parts, 2)
	assert.Equal(t, "input_text", parts[0].Type)
	assert.Equal(t, "input_image", parts[1].Type)
	assert.Equal(t, "url", parts[1].Source.Type)
	assert.Equal(t, "https://example.com/img.png", parts[1].Source.URL)
}

func TestToResponsesItems_UserWithBase64Image(t *testing.T) {
	msgs := []Message{
		{
			Role:  "user",
			Parts: []ImagePart{{Data: []byte{0x89, 0x50}, MIMEType: "image/png"}},
		},
	}
	_, items := toResponsesItems(msgs)
	require.Len(t, items, 1)
	var parts []respContentPart
	require.NoError(t, json.Unmarshal(items[0].Content, &parts))
	require.Len(t, parts, 1)
	assert.Equal(t, "base64", parts[0].Source.Type)
	assert.Equal(t, "image/png", parts[0].Source.MediaType)
	assert.NotEmpty(t, parts[0].Source.Data)
}

// --- CompleteStream / Complete via httptest ---

// completedEvent builds the response.completed SSE payload with given status and token counts.
func completedEvent(status string, inputTok, outputTok int) string {
	b, _ := json.Marshal(map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"status": status,
			"output": []any{},
			"usage": map[string]any{
				"input_tokens":  inputTok,
				"output_tokens": outputTok,
			},
		},
	})
	return string(b)
}

func TestCompleteStream_SimpleTextResponse(t *testing.T) {
	// Simulates a real Copilot /responses SSE stream for a plain text reply.
	// Trimmed from a captured trace: two text deltas then response.completed.
	body := sseBody([][2]string{
		{"response.output_text.delta", `{"type":"response.output_text.delta","delta":"Hello"}`},
		{"response.output_text.delta", `{"type":"response.output_text.delta","delta":", world!"}`},
		{"response.completed", completedEvent("completed", 10, 5)},
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, body)
	}))
	defer srv.Close()

	msg, err := newTestResponsesClient(srv).Complete(context.Background(), []Message{
		{Role: "user", Content: "Say hello"},
	}, nil)

	require.NoError(t, err)
	assert.Equal(t, "assistant", msg.Role)
	assert.Equal(t, "Hello, world!", msg.Content)
	assert.Empty(t, msg.ToolCalls)
}

func TestCompleteStream_DeltasAccumulatedIntoContent(t *testing.T) {
	// Verifies that individual delta chunks are streamed AND the final Done
	// message has Content == concatenated deltas (regression for the empty-content bug).
	deltas := []string{"one", " two", " three"}
	var evs [][2]string
	for _, d := range deltas {
		payload, _ := json.Marshal(map[string]string{"type": "response.output_text.delta", "delta": d})
		evs = append(evs, [2]string{"response.output_text.delta", string(payload)})
	}
	evs = append(evs, [2]string{"response.completed", completedEvent("completed", 5, 3)})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, sseBody(evs))
	}))
	defer srv.Close()

	cl := newTestResponsesClient(srv)
	ch := cl.CompleteStream(context.Background(), []Message{{Role: "user", Content: "count"}}, nil)

	var streamedDeltas []string
	var finalMsg Message
	for chunk := range ch {
		require.NoError(t, chunk.Err)
		if chunk.Done {
			finalMsg = chunk.Msg
		} else if chunk.Delta != "" {
			streamedDeltas = append(streamedDeltas, chunk.Delta)
		}
	}

	assert.Equal(t, deltas, streamedDeltas)
	assert.Equal(t, "one two three", finalMsg.Content)
}

func TestCompleteStream_ToolCallReconstruction(t *testing.T) {
	// Simulates the Copilot pattern where function_call items appear in
	// response.output_item.done (fully formed, not streamed piecemeal).
	itemDone, _ := json.Marshal(map[string]any{
		"type": "response.output_item.done",
		"item": map[string]any{
			"type":      "function_call",
			"call_id":   "call_1",
			"name":      "fs__read",
			"arguments": `{"path":"/etc/hosts"}`,
		},
	})
	body := sseBody([][2]string{
		{"response.output_item.done", string(itemDone)},
		{"response.completed", completedEvent("completed", 20, 10)},
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, body)
	}))
	defer srv.Close()

	msg, err := newTestResponsesClient(srv).Complete(context.Background(), []Message{
		{Role: "user", Content: "read /etc/hosts"},
	}, nil)

	require.NoError(t, err)
	require.Len(t, msg.ToolCalls, 1)
	assert.Equal(t, "call_1", msg.ToolCalls[0].ID)
	assert.Equal(t, "fs__read", msg.ToolCalls[0].Name)
	assert.Equal(t, "/etc/hosts", msg.ToolCalls[0].Arguments["path"])
}

func TestCompleteStream_ToolCallFinishReason(t *testing.T) {
	itemDone, _ := json.Marshal(map[string]any{
		"type": "response.output_item.done",
		"item": map[string]any{
			"type":      "function_call",
			"call_id":   "call_1",
			"name":      "my_tool",
			"arguments": `{}`,
		},
	})
	body := sseBody([][2]string{
		{"response.output_item.done", string(itemDone)},
		{"response.completed", completedEvent("completed", 5, 2)},
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	ch := newTestResponsesClient(srv).CompleteStream(context.Background(), []Message{
		{Role: "user", Content: "call something"},
	}, nil)

	var finalChunk StreamChunk
	for c := range ch {
		if c.Done {
			finalChunk = c
		}
	}
	assert.Equal(t, "tool_calls", finalChunk.FinishReason)
}

func TestCompleteStream_IncompleteBecomesLengthFinishReason(t *testing.T) {
	body := sseBody([][2]string{
		{"response.output_text.delta", `{"type":"response.output_text.delta","delta":"truncated"}`},
		{"response.completed", completedEvent("incomplete", 100, 50)},
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	ch := newTestResponsesClient(srv).CompleteStream(context.Background(), []Message{
		{Role: "user", Content: "write me a novel"},
	}, nil)

	var finalChunk StreamChunk
	for c := range ch {
		require.NoError(t, c.Err)
		if c.Done {
			finalChunk = c
		}
	}
	assert.Equal(t, "length", finalChunk.FinishReason)
}

func TestCompleteStream_UsageTokens(t *testing.T) {
	body := sseBody([][2]string{
		{"response.output_text.delta", `{"type":"response.output_text.delta","delta":"hi"}`},
		{"response.completed", completedEvent("completed", 42, 7)},
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	ch := newTestResponsesClient(srv).CompleteStream(context.Background(), []Message{
		{Role: "user", Content: "hi"},
	}, nil)

	var finalChunk StreamChunk
	for c := range ch {
		require.NoError(t, c.Err)
		if c.Done {
			finalChunk = c
		}
	}
	assert.Equal(t, 42, finalChunk.Usage.PromptTokens)
	assert.Equal(t, 7, finalChunk.Usage.CompletionTokens)
	assert.Equal(t, 49, finalChunk.Usage.TotalTokens)
}

func TestCompleteStream_APIErrorEvent(t *testing.T) {
	body := sseBody([][2]string{
		{"error", `{"type":"error","message":"rate limit exceeded"}`},
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	_, err := newTestResponsesClient(srv).Complete(context.Background(), []Message{
		{Role: "user", Content: "hello"},
	}, nil)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "rate limit exceeded")
}

func TestCompleteStream_HTTP4xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	defer srv.Close()

	_, err := newTestResponsesClient(srv).Complete(context.Background(), []Message{
		{Role: "user", Content: "hello"},
	}, nil)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "401")
}

func TestCompleteStream_InlineTypeField(t *testing.T) {
	// When the SSE stream omits "event:" lines and puts "type" inside the JSON
	// (the alternative format some endpoints use), the parser should still work.
	body := sseBody([][2]string{
		{"", `{"type":"response.output_text.delta","delta":"inline"}`},
		{"", completedEvent("completed", 1, 1)},
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	msg, err := newTestResponsesClient(srv).Complete(context.Background(), []Message{
		{Role: "user", Content: "test"},
	}, nil)

	require.NoError(t, err)
	assert.Equal(t, "inline", msg.Content)
}

func TestCompleteStream_MultipleToolCallsOrdering(t *testing.T) {
	// Two function_call items arriving out-of-insertion-order should be
	// returned sorted by the order they were first seen in the stream.
	item1, _ := json.Marshal(map[string]any{
		"type": "response.output_item.done",
		"item": map[string]any{"type": "function_call", "call_id": "first", "name": "tool_a", "arguments": `{}`},
	})
	item2, _ := json.Marshal(map[string]any{
		"type": "response.output_item.done",
		"item": map[string]any{"type": "function_call", "call_id": "second", "name": "tool_b", "arguments": `{}`},
	})
	body := sseBody([][2]string{
		{"response.output_item.done", string(item1)},
		{"response.output_item.done", string(item2)},
		{"response.completed", completedEvent("completed", 10, 5)},
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	defer srv.Close()

	msg, err := newTestResponsesClient(srv).Complete(context.Background(), []Message{
		{Role: "user", Content: "do both"},
	}, nil)

	require.NoError(t, err)
	require.Len(t, msg.ToolCalls, 2)
	assert.Equal(t, "first", msg.ToolCalls[0].ID)
	assert.Equal(t, "second", msg.ToolCalls[1].ID)
}

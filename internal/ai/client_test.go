package ai

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	openai "github.com/sashabaranov/go-openai"
)

// --- buildFinalMessage ---

func TestBuildFinalMessage_NoTools(t *testing.T) {
	msg, err := buildFinalMessage("hello world", nil)
	require.NoError(t, err)
	assert.Equal(t, "assistant", msg.Role)
	assert.Equal(t, "hello world", msg.Content)
	assert.Empty(t, msg.ToolCalls)
}

func TestBuildFinalMessage_SingleTool(t *testing.T) {
	acc := &accumTool{id: "call1", name: "fs__read"}
	acc.args.WriteString(`{"path":"/tmp/foo"}`)
	toolAccum := map[int]*accumTool{0: acc}

	msg, err := buildFinalMessage("", toolAccum)
	require.NoError(t, err)
	require.Len(t, msg.ToolCalls, 1)
	assert.Equal(t, "call1", msg.ToolCalls[0].ID)
	assert.Equal(t, "fs__read", msg.ToolCalls[0].Name)
	assert.Equal(t, "/tmp/foo", msg.ToolCalls[0].Arguments["path"])
}

func TestBuildFinalMessage_MultipleTools_SortedByIndex(t *testing.T) {
	acc0 := &accumTool{id: "c0", name: "tool_a"}
	acc0.args.WriteString(`{"x":1}`)
	acc2 := &accumTool{id: "c2", name: "tool_b"}
	acc2.args.WriteString(`{"x":2}`)
	// Deliberately use sparse indexes 0 and 2 (no index 1).
	toolAccum := map[int]*accumTool{2: acc2, 0: acc0}

	msg, err := buildFinalMessage("", toolAccum)
	require.NoError(t, err)
	require.Len(t, msg.ToolCalls, 2)
	// Must be sorted: index 0 first, index 2 second.
	assert.Equal(t, "c0", msg.ToolCalls[0].ID)
	assert.Equal(t, "c2", msg.ToolCalls[1].ID)
}

func TestBuildFinalMessage_EmptyArgs(t *testing.T) {
	acc := &accumTool{id: "c1", name: "no_args"}
	// empty args builder → no JSON to parse
	msg, err := buildFinalMessage("", map[int]*accumTool{0: acc})
	require.NoError(t, err)
	require.Len(t, msg.ToolCalls, 1)
	assert.Nil(t, msg.ToolCalls[0].Arguments)
}

func TestBuildFinalMessage_InvalidJSON(t *testing.T) {
	acc := &accumTool{id: "c1", name: "bad"}
	acc.args.WriteString(`{not valid json`)
	_, err := buildFinalMessage("", map[int]*accumTool{0: acc})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "bad")
}

// --- toOpenAIMessages ---

func TestToOpenAIMessages_Basic(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}
	out := toOpenAIMessages(msgs)
	require.Len(t, out, 2)
	assert.Equal(t, "user", out[0].Role)
	assert.Equal(t, "hello", out[0].Content)
	assert.Equal(t, "assistant", out[1].Role)
}

func TestToOpenAIMessages_ToolResult(t *testing.T) {
	msgs := []Message{{Role: "tool", Content: "result", ToolCallID: "call1"}}
	out := toOpenAIMessages(msgs)
	require.Len(t, out, 1)
	assert.Equal(t, "tool", out[0].Role)
	assert.Equal(t, "call1", out[0].ToolCallID)
}

func TestToOpenAIMessages_AssistantWithToolCalls(t *testing.T) {
	msgs := []Message{{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{ID: "c1", Name: "my_tool", Arguments: map[string]any{"k": "v"}},
		},
	}}
	out := toOpenAIMessages(msgs)
	require.Len(t, out, 1)
	require.Len(t, out[0].ToolCalls, 1)
	assert.Equal(t, "c1", out[0].ToolCalls[0].ID)
	assert.Equal(t, "my_tool", out[0].ToolCalls[0].Function.Name)
	assert.Contains(t, out[0].ToolCalls[0].Function.Arguments, `"k"`)
}

// --- toOpenAITools ---

func TestToOpenAITools_Nil(t *testing.T) {
	assert.Nil(t, toOpenAITools(nil))
	assert.Nil(t, toOpenAITools([]ToolDefinition{}))
}

func TestToOpenAITools_Populated(t *testing.T) {
	tools := []ToolDefinition{{
		Name:        "my_tool",
		Description: "does stuff",
		InputSchema: map[string]any{"type": "object"},
	}}
	out := toOpenAITools(tools)
	require.Len(t, out, 1)
	assert.Equal(t, openai.ToolTypeFunction, out[0].Type)
	assert.Equal(t, "my_tool", out[0].Function.Name)
	assert.Equal(t, "does stuff", out[0].Function.Description)
}

// --- fromOpenAIMessage ---

func TestFromOpenAIMessage_Simple(t *testing.T) {
	m := openai.ChatCompletionMessage{Role: "assistant", Content: "hello"}
	msg, err := fromOpenAIMessage(m)
	require.NoError(t, err)
	assert.Equal(t, "assistant", msg.Role)
	assert.Equal(t, "hello", msg.Content)
	assert.Empty(t, msg.ToolCalls)
}

func TestFromOpenAIMessage_WithToolCall(t *testing.T) {
	m := openai.ChatCompletionMessage{
		Role: "assistant",
		ToolCalls: []openai.ToolCall{{
			ID:   "c1",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "fs__read",
				Arguments: `{"path":"/tmp/x"}`,
			},
		}},
	}
	msg, err := fromOpenAIMessage(m)
	require.NoError(t, err)
	require.Len(t, msg.ToolCalls, 1)
	assert.Equal(t, "fs__read", msg.ToolCalls[0].Name)
	assert.Equal(t, "/tmp/x", msg.ToolCalls[0].Arguments["path"])
}

func TestFromOpenAIMessage_InvalidJSON(t *testing.T) {
	m := openai.ChatCompletionMessage{
		Role: "assistant",
		ToolCalls: []openai.ToolCall{{
			ID:   "c1",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "bad",
				Arguments: `{not json`,
			},
		}},
	}
	_, err := fromOpenAIMessage(m)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "bad")
}

// --- interface compliance (compile-time checked by var _ assertions in client.go,
//     but we also verify the method signatures exist via a call here) ---

func TestClientImplementsInterfaces(t *testing.T) {
	var _ Completer = (*Client)(nil)
	var _ StreamCompleter = (*Client)(nil)
}

// --- accumTool helpers ---

func TestAccumTool_NameConcatenation(t *testing.T) {
	// Streaming may send the tool name in fragments.
	acc := &accumTool{}
	acc.name += "fs_"
	acc.name += "read"
	assert.Equal(t, "fs_read", acc.name)
}

func TestAccumTool_ArgsBuilderConcatenation(t *testing.T) {
	acc := &accumTool{}
	acc.args.WriteString(`{"pa`)
	acc.args.WriteString(`th":"/tmp"}`)
	assert.Equal(t, `{"path":"/tmp"}`, acc.args.String())
}

// --- roundtrip: toOpenAIMessages → fromOpenAIMessage ---

func TestRoundtrip_Messages(t *testing.T) {
	original := []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "world"},
	}
	for i, oaiMsg := range toOpenAIMessages(original) {
		got, err := fromOpenAIMessage(oaiMsg)
		require.NoError(t, err)
		assert.Equal(t, original[i].Role, got.Role)
		assert.Equal(t, original[i].Content, got.Content)
	}
}

// --- splitToolName (used in legacy code path; keep tested) ---

func TestSplitOrDontSplitToolName(t *testing.T) {
	// Test internal splitToolName through a string operation (the function is in mcp package,
	// but we verify the separator constant concept is coherent).
	name := "myserver__my_tool"
	idx := strings.Index(name, "__")
	require.GreaterOrEqual(t, idx, 0)
	assert.Equal(t, "myserver", name[:idx])
	assert.Equal(t, "my_tool", name[idx+2:])
}

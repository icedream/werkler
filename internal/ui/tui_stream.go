package ui

import (
	"context"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/icedream/werkler/internal/ai"
)

// the first chunk as a streamChunkMsg (carrying the channel for further reads).
func doStartStream(ctx context.Context, client ai.StreamCompleter, messages []ai.Message, tools []ai.ToolDefinition) tea.Cmd {
	snapshot := make([]ai.Message, len(messages))
	copy(snapshot, messages)
	return func() tea.Msg {
		ch := client.CompleteStream(ctx, snapshot, tools)
		return readNextChunk(ch)()
	}
}

// doStream picks the incremental path (using previous_response_id to send only new
// messages) when the incremental client is enabled, falling back to the full-context
// doStartStream otherwise.  All callers should prefer this over calling doStartStream
// directly so the incremental optimisation is applied consistently.
func (m *Model) doStream(ctx context.Context, messages []ai.Message, tools []ai.ToolDefinition) tea.Cmd {
	if m.incrementalModeEnabled && m.incrementalClient != nil {
		// Snapshot messages so the goroutine inside CompleteStreamIncremental is
		// not affected by future appends to m.messages.
		snapshot := make([]ai.Message, len(messages))
		copy(snapshot, messages)
		ch := m.incrementalClient.CompleteStreamIncremental(ctx, snapshot, tools)
		return readNextChunk(ch)
	}
	return doStartStream(ctx, m.client, messages, tools)
}

// recentContextMessages returns the last n user/assistant messages from msgs,
// stripping tool calls and image parts for a clean context bundle.
func recentContextMessages(msgs []ai.Message, n int) []ai.Message {
	var filtered []ai.Message
	for _, m := range msgs {
		if m.Role != "user" && m.Role != "assistant" {
			continue
		}
		filtered = append(filtered, ai.Message{Role: m.Role, Content: m.Content})
	}
	if len(filtered) > n {
		filtered = filtered[len(filtered)-n:]
	}
	return filtered
}

// doThinkTool starts a focused sub-completion for the think tool and returns
// a command that reads the first chunk. Subsequent chunks are read via
// thinkChunkMsg dispatch in Update. Reasoning tokens are displayed live as an
// itemReasoning block; the final text answer is returned as a toolResultMsg.
func doThinkTool(ctx context.Context, callID string, client ai.StreamCompleter, question string, recentMsgs []ai.Message, effort string) tea.Cmd {
	msgs := []ai.Message{
		{
			Role: "system",
			Content: "Reason carefully and step by step. Provide a thorough, accurate analysis. " +
				"Focus only on the question asked — be concise but complete.",
		},
	}
	msgs = append(msgs, recentMsgs...)
	msgs = append(msgs, ai.Message{Role: "user", Content: question})
	if effort != "" {
		ctx = ai.WithReasoningEffortCtx(ctx, effort)
	}
	ch := client.CompleteStream(ctx, msgs, nil)
	return readNextThinkChunk(callID, ch)
}

// readNextChunk returns a Cmd that reads one chunk from ch and wraps it in a
// streamChunkMsg (which carries ch so Update can dispatch the next read).
func readNextChunk(ch <-chan ai.StreamChunk) tea.Cmd {
	return func() tea.Msg {
		chunk, ok := <-ch
		if !ok {
			// Channel closed without a Done chunk — treat as done with empty message.
			return streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Done: true}}
		}
		return streamChunkMsg{ch: ch, chunk: chunk}
	}
}

// readNextThinkChunk returns a Cmd that reads one chunk from a think sub-completion
// channel and wraps it in a thinkChunkMsg (which carries ch and callID so Update
// can dispatch the next read and associate the result with the correct tool call).
func readNextThinkChunk(callID string, ch <-chan ai.StreamChunk) tea.Cmd {
	return func() tea.Msg {
		chunk, ok := <-ch
		if !ok {
			return thinkChunkMsg{callID: callID, ch: ch, chunk: ai.StreamChunk{Done: true}}
		}
		return thinkChunkMsg{callID: callID, ch: ch, chunk: chunk}
	}
}

func readNextCompactChunk(ch <-chan ai.StreamChunk, snap []ai.Message, modelName string, toolTokens, maxTokens int) tea.Cmd {
	return func() tea.Msg {
		chunk, ok := <-ch
		if !ok {
			return compactChunkMsg{ch: ch, chunk: ai.StreamChunk{Done: true}, snap: snap, modelName: modelName, toolTokens: toolTokens, maxTokens: maxTokens}
		}
		return compactChunkMsg{ch: ch, chunk: chunk, snap: snap, modelName: modelName, toolTokens: toolTokens, maxTokens: maxTokens}
	}
}

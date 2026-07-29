package ui

import (
	"context"
	"fmt"
	"io"

	tea "charm.land/bubbletea/v2"
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

// readNextChunk returns a Cmd that reads one chunk from ch and wraps it in a
// streamChunkMsg (which carries ch so Update can dispatch the next read).
func readNextChunk(ch <-chan ai.StreamChunk) tea.Cmd {
	return func() tea.Msg {
		chunk, ok := <-ch
		if !ok {
			return streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Done: true}}
		}
		return streamChunkMsg{ch: ch, chunk: chunk}
	}
}

func readNextCompactChunk(ch <-chan ai.StreamChunk, snap []ai.Message, modelName string, toolTokens, maxTokens int) tea.Cmd {
	return func() tea.Msg {
		chunk, ok := <-ch
		if !ok {
			// Channel closed without a terminal Done/Err chunk — the server was
			// killed or the goroutine panicked.  Treat as an error so the
			// compaction handler does not mistake this for a successful completion
			// and proceed to apply a partial (or empty) summary.
			return compactChunkMsg{
				ch:         ch,
				chunk:      ai.StreamChunk{Err: fmt.Errorf("compaction stream closed unexpectedly: %w", io.ErrUnexpectedEOF)},
				snap:       snap,
				modelName:  modelName,
				toolTokens: toolTokens,
				maxTokens:  maxTokens,
			}
		}
		return compactChunkMsg{ch: ch, chunk: chunk, snap: snap, modelName: modelName, toolTokens: toolTokens, maxTokens: maxTokens}
	}
}

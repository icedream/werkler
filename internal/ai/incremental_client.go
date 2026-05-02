package ai

import "context"

// IncrementalClient wraps a StreamCompleter and adds incremental streaming
// via the Responses API's previous_response_id mechanism.
//
// On the first call CompleteStreamIncremental behaves identically to a plain
// CompleteStream (full message history is sent, store:true is included so the
// server persists the response).  Every subsequent call captures the response
// ID from the server's Done chunk and, on the next invocation, sends only the
// messages that appeared after the last assistant turn together with
// previous_response_id.  The server reconstructs the full context from its
// stored state, so only the new tokens (tool results, next user turn) travel
// over the wire.
//
// If the server does not return a response ID (self-hosted models that ignore
// store:true) lastResponseID stays "" and every call falls back to full-context
// mode transparently.
type IncrementalClient struct {
	client         StreamCompleter
	lastResponseID string
}

// NewIncrementalClient creates a new IncrementalClient wrapping client.
func NewIncrementalClient(client StreamCompleter) *IncrementalClient {
	return &IncrementalClient{client: client}
}

// CompleteStreamIncremental starts a streaming request using previous_response_id
// when a prior response ID is available, sending only new messages rather than
// the full conversation history.
func (ic *IncrementalClient) CompleteStreamIncremental(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
) <-chan StreamChunk {
	// Snapshot and immediately clear so a failed or aborted call can never
	// poison the next one with a stale ID.
	lastResponseID := ic.lastResponseID
	ic.lastResponseID = ""

	var streamCtx context.Context
	var streamMessages []Message

	if lastResponseID != "" {
		// Find the LAST assistant message — everything after it is new.
		lastAssistantIdx := -1
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == "assistant" {
				lastAssistantIdx = i
				break
			}
		}
		if lastAssistantIdx >= 0 && lastAssistantIdx < len(messages)-1 {
			streamMessages = messages[lastAssistantIdx+1:]
			streamCtx = WithPreviousResponseID(ctx, lastResponseID)
		} else {
			// Nothing new after the last assistant message; fall back to full context.
			streamMessages = messages
			streamCtx = ctx
		}
	} else {
		streamMessages = messages
		streamCtx = ctx
	}

	ch := make(chan StreamChunk, 16)
	go func() {
		defer close(ch)
		for chunk := range ic.client.CompleteStream(streamCtx, streamMessages, tools) {
			if chunk.Done {
				// Write BEFORE the channel send so the caller goroutine is
				// guaranteed to observe the updated value on its next call
				// (the channel send provides the happens-before edge).
				ic.lastResponseID = chunk.ResponseID
			}
			select {
			case ch <- chunk:
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch
}

// SetLastResponseID overrides the stored response ID.  Call with "" to force
// the next CompleteStreamIncremental to send full context (e.g. after context
// compaction rewrites the message history).
func (ic *IncrementalClient) SetLastResponseID(id string) {
	ic.lastResponseID = id
}

// GetLastResponseID returns the response ID captured from the last response.
func (ic *IncrementalClient) GetLastResponseID() string {
	return ic.lastResponseID
}

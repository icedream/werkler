package ai

import (
	"context"
	"fmt"
)

// StreamCompleterAdapter wraps a StreamCompleter to implement Completer
// by collecting the final Done chunk from the stream.
type StreamCompleterAdapter struct {
	sc StreamCompleter
}

// NewStreamCompleterAdapter wraps sc to implement Completer.
func NewStreamCompleterAdapter(sc StreamCompleter) *StreamCompleterAdapter {
	return &StreamCompleterAdapter{sc: sc}
}

func (a *StreamCompleterAdapter) Complete(ctx context.Context, messages []Message, tools []ToolDefinition) (Message, error) {
	ch := a.sc.CompleteStream(ctx, messages, tools)
	for chunk := range ch {
		if chunk.Err != nil {
			return Message{}, chunk.Err
		}
		if chunk.Done {
			return chunk.Msg, nil
		}
	}
	return Message{}, fmt.Errorf("stream ended without a final message")
}

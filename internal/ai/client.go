package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// Client wraps the OpenAI-compatible API for chat completions with tool use.
type Client struct {
	inner openai.Client
	model string
}

// New creates a Client using the given base URL, API key and model name.
func New(endpoint, apiKey, model string) *Client {
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = endpoint
	return &Client{
		inner: *openai.NewClientWithConfig(cfg),
		model: model,
	}
}

// ToolDefinition describes a callable tool for the AI.
type ToolDefinition struct {
	Name        string
	Description string
	// InputSchema is a JSON Schema object (map[string]any) describing the input.
	InputSchema any
}

// Message represents a single entry in the conversation history.
type Message struct {
	Role       string
	Content    string
	ToolCallID string     // set for tool result messages
	ToolCalls  []ToolCall // set for assistant messages that invoke tools
}

// ToolCall is a tool invocation requested by the assistant.
type ToolCall struct {
	ID        string
	Name      string
	Arguments map[string]any
}

// StreamChunk is one event from a streaming completion.
// Either Delta (incremental text) or Done (final message + tool calls) or Err is set.
type StreamChunk struct {
	Delta string  // non-empty for incremental text chunks
	Done  bool    // true on the final chunk; Msg is valid
	Msg   Message // valid only when Done && Err == nil
	Err   error   // non-nil on error; stream is terminated
}

// Complete sends the conversation to the AI and returns the next message.
// The returned message may contain ToolCalls that the caller must execute.
func (c *Client) Complete(ctx context.Context, messages []Message, tools []ToolDefinition) (Message, error) {
	req := openai.ChatCompletionRequest{
		Model:    c.model,
		Messages: toOpenAIMessages(messages),
		Tools:    toOpenAITools(tools),
	}

	resp, err := c.inner.CreateChatCompletion(ctx, req)
	if err != nil {
		return Message{}, fmt.Errorf("chat completion: %w", err)
	}
	if len(resp.Choices) == 0 {
		return Message{}, fmt.Errorf("chat completion: no choices returned")
	}

	return fromOpenAIMessage(resp.Choices[0].Message)
}

// CompleteStream starts a streaming chat completion and returns a channel of
// StreamChunk events. The channel is closed after the final Done chunk (or an
// Err chunk). The caller must drain the channel to avoid blocking the sender.
// Context cancellation stops the stream; the goroutine will send an Err chunk
// and exit cleanly.
func (c *Client) CompleteStream(ctx context.Context, messages []Message, tools []ToolDefinition) <-chan StreamChunk {
	ch := make(chan StreamChunk, 16)
	go func() {
		defer close(ch)

		send := func(sc StreamChunk) bool {
			select {
			case ch <- sc:
				return true
			case <-ctx.Done():
				return false
			}
		}

		req := openai.ChatCompletionRequest{
			Model:    c.model,
			Messages: toOpenAIMessages(messages),
			Tools:    toOpenAITools(tools),
		}

		stream, err := c.inner.CreateChatCompletionStream(ctx, req)
		if err != nil {
			send(StreamChunk{Err: fmt.Errorf("chat completion stream: %w", err)})
			return
		}
		defer func() { _ = stream.Close() }()

		// Accumulate the full response across chunks.
		var contentBuf strings.Builder
		// toolAccum maps tool-call index → accumulated data.
		type accumTool struct {
			id   string
			name string
			args strings.Builder
		}
		toolAccum := map[int]*accumTool{}

		for {
			resp, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				send(StreamChunk{Err: fmt.Errorf("stream recv: %w", err)})
				return
			}
			if len(resp.Choices) == 0 {
				continue
			}
			delta := resp.Choices[0].Delta

			if delta.Content != "" {
				contentBuf.WriteString(delta.Content)
				if !send(StreamChunk{Delta: delta.Content}) {
					return
				}
			}

			// Accumulate tool calls by index (streamed as fragments).
			for _, tc := range delta.ToolCalls {
				idx := 0
				if tc.Index != nil {
					idx = *tc.Index
				}
				acc, ok := toolAccum[idx]
				if !ok {
					acc = &accumTool{}
					toolAccum[idx] = acc
				}
				if tc.ID != "" {
					acc.id = tc.ID
				}
				if tc.Function.Name != "" {
					acc.name += tc.Function.Name
				}
				acc.args.WriteString(tc.Function.Arguments)
			}
		}

		// Build the final message.
		msg := Message{
			Role:    "assistant",
			Content: contentBuf.String(),
		}
		for i := 0; ; i++ {
			acc, ok := toolAccum[i]
			if !ok {
				break
			}
			var args map[string]any
			if s := acc.args.String(); s != "" {
				if err := json.Unmarshal([]byte(s), &args); err != nil {
					send(StreamChunk{Err: fmt.Errorf("parsing tool call arguments for %q: %w", acc.name, err)})
					return
				}
			}
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:        acc.id,
				Name:      acc.name,
				Arguments: args,
			})
		}

		send(StreamChunk{Done: true, Msg: msg})
	}()
	return ch
}

func toOpenAIMessages(msgs []Message) []openai.ChatCompletionMessage {
	out := make([]openai.ChatCompletionMessage, 0, len(msgs))
	for _, m := range msgs {
		msg := openai.ChatCompletionMessage{
			Role:       m.Role,
			Content:    m.Content,
			ToolCallID: m.ToolCallID,
		}
		for _, tc := range m.ToolCalls {
			raw, _ := json.Marshal(tc.Arguments)
			msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
				ID:   tc.ID,
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      tc.Name,
					Arguments: string(raw),
				},
			})
		}
		out = append(out, msg)
	}
	return out
}

func toOpenAITools(tools []ToolDefinition) []openai.Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]openai.Tool, 0, len(tools))
	for _, t := range tools {
		raw, _ := json.Marshal(t.InputSchema)
		params := json.RawMessage(raw)
		out = append(out, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}
	return out
}

func fromOpenAIMessage(m openai.ChatCompletionMessage) (Message, error) {
	msg := Message{
		Role:    m.Role,
		Content: m.Content,
	}
	for _, tc := range m.ToolCalls {
		var args map[string]any
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return Message{}, fmt.Errorf("parsing tool call arguments for %q: %w", tc.Function.Name, err)
		}
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: args,
		})
	}
	return msg, nil
}

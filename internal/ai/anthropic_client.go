package ai

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// AnthropicClient wraps the Anthropic Messages API.
// It implements Completer, StreamCompleter, ModelManager, ModelInfoGetter, and Modeler.
type AnthropicClient struct {
	inner            anthropic.Client
	model            string
	disableReasoning bool
}

// NewAnthropicClient creates an AnthropicClient.
// endpoint may be empty (uses the default Anthropic API base URL).
func NewAnthropicClient(endpoint, apiKey, model string, opts ...AnthropicClientOption) *AnthropicClient {
	clientOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}
	if endpoint != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(endpoint))
	}
	c := &AnthropicClient{
		inner: anthropic.NewClient(clientOpts...),
		model: model,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// AnthropicClientOption configures an AnthropicClient.
type AnthropicClientOption func(*AnthropicClient)

// WithAnthropicDisableReasoning prevents thinking from being enabled.
func WithAnthropicDisableReasoning() AnthropicClientOption {
	return func(c *AnthropicClient) { c.disableReasoning = true }
}

// Model returns the current model name (implements Modeler).
func (c *AnthropicClient) Model() string { return c.model }

// SetModel changes the active model (implements ModelManager).
func (c *AnthropicClient) SetModel(item ModelItem) { c.model = item.Model }

// GetModelInfo returns empty info (Anthropic API doesn't expose context window sizes).
func (c *AnthropicClient) GetModelInfo(_ context.Context) (ModelInfo, error) {
	return ModelInfo{Model: c.model}, nil
}

func (c *AnthropicClient) ListModels(ctx context.Context) ([]ModelItem, error) {
	page, err := c.inner.Models.List(ctx, anthropic.ModelListParams{})
	if err != nil {
		return nil, fmt.Errorf("listing Anthropic models: %w", err)
	}
	var items []ModelItem
	for _, m := range page.Data {
		items = append(items, ModelItem{Model: m.ID})
	}
	slices.SortFunc(items, func(a, b ModelItem) int { return cmp.Compare(a.Model, b.Model) })
	return items, nil
}

func toAnthropicMessages(msgs []Message) (string, []anthropic.MessageParam, error) {
	var systemParts []string
	var out []anthropic.MessageParam

	for _, m := range msgs {
		switch m.Role {
		case "system":
			systemParts = append(systemParts, m.Content)

		case "user":
			var blocks []anthropic.ContentBlockParamUnion
			if m.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, p := range m.Parts {
				if p.URL != "" {
					blocks = append(blocks, anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: p.URL}))
				} else if len(p.Data) > 0 {
					blocks = append(blocks, anthropic.NewImageBlock(anthropic.Base64ImageSourceParam{
						MediaType: anthropic.Base64ImageSourceMediaType(p.MIMEType),
						Data:      base64.StdEncoding.EncodeToString(p.Data),
					}))
				}
			}
			if len(blocks) == 0 {
				continue // skip empty user messages
			}
			out = append(out, anthropic.NewUserMessage(blocks...))

		case "assistant":
			// Drop empty assistant messages (no content, no tool calls)
			if m.Content == "" && len(m.ToolCalls) == 0 {
				continue
			}
			var blocks []anthropic.ContentBlockParamUnion
			if m.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				raw, _ := json.Marshal(tc.Arguments)
				if string(raw) == "null" {
					raw = []byte("{}")
				}
				blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, json.RawMessage(raw), tc.Name))
			}
			out = append(out, anthropic.NewAssistantMessage(blocks...))

		case "tool":
			out = append(out, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(m.ToolCallID, m.Content, false),
			))
		}
	}

	return strings.Join(systemParts, "\n\n"), out, nil
}

func toAnthropicTools(tools []ToolDefinition) []anthropic.ToolUnionParam {
	if len(tools) == 0 {
		return nil
	}
	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		raw, _ := json.Marshal(t.InputSchema)
		var props map[string]any
		_ = json.Unmarshal(raw, &props)
		tool := anthropic.ToolParam{
			Name:        t.Name,
			Description: anthropic.String(t.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: props,
			},
		}
		out = append(out, anthropic.ToolUnionParam{OfTool: &tool})
	}
	return out
}

func (c *AnthropicClient) Complete(ctx context.Context, messages []Message, tools []ToolDefinition) (Message, error) {
	system, anthropicMsgs, err := toAnthropicMessages(messages)
	if err != nil {
		return Message{}, err
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(c.model),
		MaxTokens: 4096,
		Messages:  anthropicMsgs,
		Tools:     toAnthropicTools(tools),
	}
	if system != "" {
		params.System = []anthropic.TextBlockParam{{Text: system}}
	}

	effort := ReasoningEffortFromCtx(ctx)
	if effort != "" && !c.disableReasoning {
		params.MaxTokens = 8192
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
		}
	}

	resp, err := c.inner.Messages.New(ctx, params)
	if err != nil {
		return Message{}, fmt.Errorf("anthropic completion: %w", err)
	}

	return fromAnthropicMessage(resp), nil
}

func fromAnthropicMessage(msg *anthropic.Message) Message {
	out := Message{Role: "assistant"}
	var reasoning []string
	for _, block := range msg.Content {
		switch v := block.AsAny().(type) {
		case anthropic.TextBlock:
			out.Content += v.Text
		case anthropic.ThinkingBlock:
			reasoning = append(reasoning, v.Thinking)
		case anthropic.ToolUseBlock:
			var args map[string]any
			_ = json.Unmarshal([]byte(v.JSON.Input.Raw()), &args)
			out.ToolCalls = append(out.ToolCalls, ToolCall{
				ID:        v.ID,
				Name:      v.Name,
				Arguments: args,
			})
		}
	}
	out.Reasoning = strings.Join(reasoning, "\n")
	return out
}

func (c *AnthropicClient) CompleteStream(ctx context.Context, messages []Message, tools []ToolDefinition) <-chan StreamChunk {
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

		system, anthropicMsgs, err := toAnthropicMessages(messages)
		if err != nil {
			send(StreamChunk{Err: err})
			return
		}

		params := anthropic.MessageNewParams{
			Model:     anthropic.Model(c.model),
			MaxTokens: 4096,
			Messages:  anthropicMsgs,
			Tools:     toAnthropicTools(tools),
		}
		if system != "" {
			params.System = []anthropic.TextBlockParam{{Text: system}}
		}

		effort := ReasoningEffortFromCtx(ctx)
		if effort != "" && !c.disableReasoning {
			params.MaxTokens = 8192
			params.Thinking = anthropic.ThinkingConfigParamUnion{
				OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
			}
		}

		stream := c.inner.Messages.NewStreaming(ctx, params)
		defer func() { _ = stream.Close() }()

		var (
			contentBuf   strings.Builder
			reasoningBuf strings.Builder
			toolAccum    = map[int64]*accumAnthropicTool{}
			finishReason string
			promptTokens int
			outputTokens int
		)

		for stream.Next() {
			event := stream.Current()
			switch e := event.AsAny().(type) {
			case anthropic.MessageStartEvent:
				promptTokens = int(e.Message.Usage.InputTokens)
			case anthropic.ContentBlockStartEvent:
				idx := e.Index
				switch cb := e.ContentBlock.AsAny().(type) {
				case anthropic.TextBlock:
					_ = cb
				case anthropic.ThinkingBlock:
					_ = cb
				case anthropic.ToolUseBlock:
					toolAccum[idx] = &accumAnthropicTool{id: cb.ID, name: cb.Name}
				}
			case anthropic.ContentBlockDeltaEvent:
				idx := e.Index
				switch d := e.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					contentBuf.WriteString(d.Text)
					if !send(StreamChunk{Delta: d.Text}) {
						return
					}
				case anthropic.ThinkingDelta:
					reasoningBuf.WriteString(d.Thinking)
					if !send(StreamChunk{ReasoningDelta: d.Thinking}) {
						return
					}
				case anthropic.InputJSONDelta:
					if acc, ok := toolAccum[idx]; ok {
						acc.args.WriteString(d.PartialJSON)
					}
				}
			case anthropic.MessageDeltaEvent:
				outputTokens = int(e.Usage.OutputTokens)
				switch string(e.Delta.StopReason) {
				case "end_turn", "stop_sequence":
					finishReason = "stop"
				case "max_tokens":
					finishReason = "length"
				case "tool_use":
					finishReason = "tool_calls"
				default:
					finishReason = string(e.Delta.StopReason)
				}
			}
		}

		if err := stream.Err(); err != nil {
			send(StreamChunk{Err: fmt.Errorf("anthropic stream: %w", err)})
			return
		}

		msg := Message{
			Role:      "assistant",
			Content:   contentBuf.String(),
			Reasoning: reasoningBuf.String(),
		}

		// Collect tool calls in index order.
		idxs := make([]int64, 0, len(toolAccum))
		for idx := range toolAccum {
			idxs = append(idxs, idx)
		}
		slices.Sort(idxs)
		for _, idx := range idxs {
			acc := toolAccum[idx]
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

		usage := Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      promptTokens + outputTokens,
		}
		send(StreamChunk{Done: true, Msg: msg, FinishReason: finishReason, Usage: usage})
	}()
	return ch
}

type accumAnthropicTool struct {
	id   string
	name string
	args strings.Builder
}

// Compile-time assertions.
var _ Completer = (*AnthropicClient)(nil)
var _ StreamCompleter = (*AnthropicClient)(nil)
var _ ModelManager = (*AnthropicClient)(nil)
var _ ModelInfoGetter = (*AnthropicClient)(nil)
var _ Modeler = (*AnthropicClient)(nil)

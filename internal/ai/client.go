package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// Client wraps the OpenAI-compatible API for chat completions with tool use.
type Client struct {
	inner    openai.Client
	model    string
	endpoint string // base URL, stored for provider-specific probing (e.g. Ollama /api/show)
}

// New creates a Client using the given base URL, API key and model name.
func New(endpoint, apiKey, model string) *Client {
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = endpoint
	return &Client{
		inner:    *openai.NewClientWithConfig(cfg),
		model:    model,
		endpoint: endpoint,
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

// Completer can perform a non-streaming chat completion.
type Completer interface {
	Complete(ctx context.Context, messages []Message, tools []ToolDefinition) (Message, error)
}

// StreamCompleter can perform a streaming chat completion.
type StreamCompleter interface {
	CompleteStream(ctx context.Context, messages []Message, tools []ToolDefinition) <-chan StreamChunk
}

// ModelItem identifies a model available from a provider.
// Provider is empty for single-provider clients.
type ModelItem struct {
	Provider string
	Model    string
}

// Display returns the formatted display string for the item.
// When Provider is set, the format is "Provider: Model".
func (m ModelItem) Display() string {
	if m.Provider == "" {
		return m.Model
	}
	return m.Provider + ": " + m.Model
}

// ModelManager can list available chat models and switch the active model.
type ModelManager interface {
	ListModels(ctx context.Context) ([]ModelItem, error)
	SetModel(item ModelItem)
}

// Compile-time assertions: *Client satisfies all interfaces.
var _ Completer = (*Client)(nil)
var _ StreamCompleter = (*Client)(nil)
var _ ModelManager = (*Client)(nil)

// SetModel changes the model used for subsequent completions.
func (c *Client) SetModel(item ModelItem) {
	c.model = item.Model
}

// NewWithHTTPClient creates a Client using the given base URL, model and a
// custom *http.Client (e.g. with a custom transport for token injection).
// The API key is handled by the transport; an empty string is used here.
func NewWithHTTPClient(baseURL, model string, httpClient *http.Client) *Client {
	cfg := openai.DefaultConfig("")
	cfg.BaseURL = baseURL
	cfg.HTTPClient = httpClient
	return &Client{
		inner:    *openai.NewClientWithConfig(cfg),
		model:    model,
		endpoint: baseURL,
	}
}

// nonChatKeywords identifies model IDs that are clearly not chat-completion models.
var nonChatKeywords = []string{
	"embed", "embedding",
	"whisper", "tts", "transcri",
	"dall-e", "dall_e", "image",
	"moderat",
	"rerank",
}

// ListModels returns the IDs of models available for chat completions.
// Models that are clearly non-chat (embeddings, TTS, image generation, etc.)
// are filtered out. The list is sorted alphabetically.
func (c *Client) ListModels(ctx context.Context) ([]ModelItem, error) {
	resp, err := c.inner.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("listing models: %w", err)
	}
	var items []ModelItem
	for _, m := range resp.Models {
		id := strings.ToLower(m.ID)
		skip := false
		for _, kw := range nonChatKeywords {
			if strings.Contains(id, kw) {
				skip = true
				break
			}
		}
		if !skip {
			items = append(items, ModelItem{Model: m.ID})
		}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Model < items[j].Model })
	return items, nil
}

type accumTool struct {
	id   string
	name string
	args strings.Builder
}

// buildFinalMessage assembles the complete assistant Message from accumulated
// streaming fragments. toolAccum maps tool-call index → accumulated data.
// Indexes are iterated in sorted order to handle sparse or out-of-order indexes.
func buildFinalMessage(content string, toolAccum map[int]*accumTool) (Message, error) {
	msg := Message{
		Role:    "assistant",
		Content: content,
	}
	if len(toolAccum) == 0 {
		return msg, nil
	}

	indexes := make([]int, 0, len(toolAccum))
	for idx := range toolAccum {
		indexes = append(indexes, idx)
	}
	sort.Ints(indexes)

	for _, idx := range indexes {
		acc := toolAccum[idx]
		var args map[string]any
		if s := acc.args.String(); s != "" {
			if err := json.Unmarshal([]byte(s), &args); err != nil {
				return Message{}, fmt.Errorf("parsing tool call arguments for %q: %w", acc.name, err)
			}
		}
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID:        acc.id,
			Name:      acc.name,
			Arguments: args,
		})
	}
	return msg, nil
}

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

		msg, err := buildFinalMessage(contentBuf.String(), toolAccum)
		if err != nil {
			send(StreamChunk{Err: err})
			return
		}
		send(StreamChunk{Done: true, Msg: msg})
	}()
	return ch
}

func toOpenAIMessages(msgs []Message) []openai.ChatCompletionMessage {
	out := make([]openai.ChatCompletionMessage, 0, len(msgs))
	for _, m := range msgs {
		// Drop empty assistant messages (no content, no tool calls) — they are
		// produced when the model returns a blank response and some providers
		// (e.g. Ollama) reject them with a 400 "invalid message content type: <nil>".
		if m.Role == "assistant" && m.Content == "" && len(m.ToolCalls) == 0 {
			continue
		}
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

// GetModelInfo probes the provider for model metadata. Currently only Ollama
// is supported: it tries <base>/api/show (stripping a /v1 suffix from the
// configured endpoint). Returns an empty ModelInfo without error for
// non-Ollama providers or when the probe fails.
//
// For Ollama, num_ctx from the Modelfile parameters is preferred over the
// architectural context length because it reflects the actually-configured
// context window.
func (c *Client) GetModelInfo(ctx context.Context) (ModelInfo, error) {
	base := strings.TrimSuffix(strings.TrimSuffix(strings.TrimRight(c.endpoint, "/"), "/v1"), "/")
	if base == "" {
		return ModelInfo{}, nil
	}

	type showRequest struct {
		Model string `json:"model"`
	}
	body, err := json.Marshal(showRequest{Model: c.model})
	if err != nil {
		return ModelInfo{}, nil
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/show", strings.NewReader(string(body)))
	if err != nil {
		return ModelInfo{}, nil
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		return ModelInfo{}, nil
	}
	defer func() { _ = resp.Body.Close() }()

	var payload struct {
		Parameters string         `json:"parameters"`
		ModelInfo  map[string]any `json:"model_info"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return ModelInfo{}, nil
	}

	info := ModelInfo{Model: c.model}

	// Prefer num_ctx from the Modelfile parameters (effective limit) over the
	// architectural maximum reported in model_info.
	if n := parseNumCtx(payload.Parameters); n > 0 {
		info.Context.MaxTokens = n
	} else {
		// Fall back to the architectural context length from model_info.
		for k, v := range payload.ModelInfo {
			if strings.HasSuffix(k, ".context_length") {
				if f, ok := v.(float64); ok {
					info.Context.MaxTokens = int(f)
					break
				}
			}
		}
	}
	return info, nil
}

// parseNumCtx extracts the num_ctx value from an Ollama parameters string.
// The format is one parameter per line: "num_ctx    4096".
// Returns 0 if not present.
func parseNumCtx(params string) int {
	for _, line := range strings.Split(params, "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == "num_ctx" {
			var n int
			if _, err := fmt.Sscanf(fields[1], "%d", &n); err == nil && n > 0 {
				return n
			}
		}
	}
	return 0
}

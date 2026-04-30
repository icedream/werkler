package ai

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"

	openai "github.com/sashabaranov/go-openai"
)

// Client wraps the OpenAI-compatible API for chat completions with tool use.
type Client struct {
	inner              openai.Client
	model              string
	endpoint           string       // base URL, stored for provider-specific probing (e.g. Ollama /api/show)
	probeClient        *http.Client // HTTP client used for model-info probes; nil = use http.DefaultClient
	disableStreamUsage atomic.Bool  // set after a 400 caused by unsupported stream_options
	disableReasoning   bool         // when true, reasoning_effort is never set regardless of context
}

// reasoningEffortKey is the context key for a per-call reasoning effort override.
type reasoningEffortKey struct{}

// WithReasoningEffortCtx returns a copy of ctx annotated with the given
// reasoning effort level ("low", "medium", "high"). Pass the returned ctx to
// CompleteStream or Complete to apply it for that call only.
func WithReasoningEffortCtx(ctx context.Context, effort string) context.Context {
	return context.WithValue(ctx, reasoningEffortKey{}, effort)
}

// ReasoningEffortFromCtx returns the reasoning effort level stored in ctx by
// WithReasoningEffortCtx, or "" if none is set.
func ReasoningEffortFromCtx(ctx context.Context) string {
	if v, ok := ctx.Value(reasoningEffortKey{}).(string); ok {
		return v
	}
	return ""
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

// NewWithTransport creates a Client that sends requests through the given
// transport. If transport is nil, http.DefaultTransport is used. The API key
// is included in the Authorization header as usual.
func NewWithTransport(endpoint, apiKey, model string, transport http.RoundTripper, opts ...ClientOption) *Client {
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = endpoint
	cfg.HTTPClient = &http.Client{Transport: transport}
	c := &Client{
		inner:    *openai.NewClientWithConfig(cfg),
		model:    model,
		endpoint: endpoint,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// ToolDefinition describes a callable tool for the AI.
type ToolDefinition struct {
	Name        string
	Description string
	// InputSchema is a JSON Schema object (map[string]any) describing the input.
	InputSchema any
}

// ImagePart holds an image to include in a user message. Either URL (for
// http/https images) or Data+MIMEType (for locally-loaded files) must be set.
type ImagePart struct {
	// URL is an http/https URL for remote images.
	URL string `json:"url,omitempty"`
	// Data holds the raw image bytes for local files (encoded as base64 in the API).
	Data []byte `json:"data,omitempty"`
	// MIMEType is required when Data is set (e.g. "image/png", "image/jpeg").
	MIMEType string `json:"mime_type,omitempty"`
	// Name is the display name shown in the TUI (e.g. the file basename).
	Name string `json:"name,omitempty"`
}

// Message represents a single entry in the conversation history.
type Message struct {
	Role       string
	Content    string
	Parts      []ImagePart // optional images attached to a user message
	Reasoning  string      // reasoning/thinking content emitted by reasoning models (not sent back to the API)
	ToolCallID string      // set for tool result messages
	ToolCalls  []ToolCall  // set for assistant messages that invoke tools
}

// ToolCall is a tool invocation requested by the assistant.
type ToolCall struct {
	ID        string
	Name      string
	Arguments map[string]any
}

// StreamChunk is one event from a streaming completion.
// Either Delta/ReasoningDelta (incremental text) or Done (final message + tool calls) or Err is set.
type StreamChunk struct {
	Delta          string     // non-empty for incremental content text chunks
	ReasoningDelta string     // non-empty for incremental reasoning/thinking chunks
	Done           bool       // true on the final chunk; Msg is valid
	Msg            Message    // valid only when Done && Err == nil
	Err            error      // non-nil on error; stream is terminated
	RateLimits     RateLimits // populated on Done; zero when provider doesn't report limits
	FinishReason   string     // "stop", "length", "tool_calls", etc.; populated on Done
	Usage          Usage      // token usage; populated on Done when provider reports it
}

// Usage holds token consumption statistics for a single AI turn.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
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

// ClientOption configures a [Client] at construction time.
type ClientOption func(*Client)

// WithNoStreamUsage disables the stream_options.include_usage field in streaming
// requests. Use this for providers (e.g. GitHub Copilot) that reject that field.
func WithNoStreamUsage() ClientOption {
	return func(c *Client) {
		c.disableStreamUsage.Store(true)
	}
}

// WithDisableReasoning prevents reasoning_effort from ever being applied for
// this client, regardless of what the caller passes via context. Use for
// providers or models that do not support the reasoning_effort parameter.
func WithDisableReasoning() ClientOption {
	return func(c *Client) {
		c.disableReasoning = true
	}
}

// NewWithHTTPClient creates a Client using the given base URL, model and a
// custom *http.Client (e.g. with a custom transport for token injection).
// The API key is handled by the transport; an empty string is used here.
func NewWithHTTPClient(baseURL, model string, httpClient *http.Client, opts ...ClientOption) *Client {
	cfg := openai.DefaultConfig("")
	cfg.BaseURL = baseURL
	cfg.HTTPClient = httpClient
	c := &Client{
		inner:       *openai.NewClientWithConfig(cfg),
		model:       model,
		endpoint:    baseURL,
		probeClient: httpClient,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
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
	slices.SortFunc(items, func(a, b ModelItem) int { return cmp.Compare(a.Model, b.Model) })
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
func buildFinalMessage(content, reasoning string, toolAccum map[int]*accumTool) (Message, error) {
	msg := Message{
		Role:      "assistant",
		Content:   content,
		Reasoning: reasoning,
	}
	if len(toolAccum) == 0 {
		return msg, nil
	}

	indexes := make([]int, 0, len(toolAccum))
	for idx := range toolAccum {
		indexes = append(indexes, idx)
	}
	slices.Sort(indexes)

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
	if !c.disableReasoning {
		if effort := ReasoningEffortFromCtx(ctx); effort != "" {
			req.ReasoningEffort = effort
		}
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
		if !c.disableReasoning {
			if effort := ReasoningEffortFromCtx(ctx); effort != "" {
				req.ReasoningEffort = effort
			}
		}
		if !c.disableStreamUsage.Load() {
			req.StreamOptions = &openai.StreamOptions{IncludeUsage: true}
		}

		stream, err := c.inner.CreateChatCompletionStream(ctx, req)
		if err != nil {
			// Some providers (e.g. GitHub Copilot) reject stream_options with a plain
			// 400 "Bad Request".  Retry once without it and remember for future calls.
			if req.StreamOptions != nil && isHTTPStatusError(err, http.StatusBadRequest) {
				c.disableStreamUsage.Store(true)
				req.StreamOptions = nil
				stream, err = c.inner.CreateChatCompletionStream(ctx, req)
			}
			if err != nil && isHTTPStatusError(err, http.StatusBadRequest) {
				// Dump the request body to a temp file so the user can diagnose the rejection.
				if raw, merr := json.Marshal(req); merr == nil {
					if f, ferr := os.CreateTemp("", "werkler-400-*.json"); ferr == nil {
						_, _ = f.Write(raw)
						_ = f.Close()
						_, _ = fmt.Fprintf(os.Stderr, "\n[werkler debug] 400 Bad Request — request body written to: %s\n", f.Name())
					}
				}
			}
		}
		if err != nil {
			send(StreamChunk{Err: fmt.Errorf("chat completion stream: %w", err)})
			return
		}
		defer func() { _ = stream.Close() }()

		// Capture rate limit headers from the HTTP response (available immediately).
		rl := stream.GetRateLimitHeaders()
		rateLimits := RateLimits{
			LimitRequests:     rl.LimitRequests,
			LimitTokens:       rl.LimitTokens,
			RemainingRequests: rl.RemainingRequests,
			RemainingTokens:   rl.RemainingTokens,
		}

		// Accumulate the full response across chunks.
		var contentBuf strings.Builder
		var reasoningBuf strings.Builder
		// toolAccum maps tool-call index → accumulated data.
		toolAccum := map[int]*accumTool{}
		var finishReason string
		var usage Usage

		for {
			resp, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				send(StreamChunk{Err: fmt.Errorf("stream recv: %w", err)})
				return
			}
			// Usage arrives in the final chunk (no Choices) when include_usage is set.
			if resp.Usage != nil {
				usage = Usage{
					PromptTokens:     resp.Usage.PromptTokens,
					CompletionTokens: resp.Usage.CompletionTokens,
					TotalTokens:      resp.Usage.TotalTokens,
				}
			}
			if len(resp.Choices) == 0 {
				continue
			}
			choice := resp.Choices[0]
			delta := choice.Delta
			if choice.FinishReason != "" {
				finishReason = string(choice.FinishReason)
			}

			if delta.ReasoningContent != "" {
				reasoningBuf.WriteString(delta.ReasoningContent)
				if !send(StreamChunk{ReasoningDelta: delta.ReasoningContent}) {
					return
				}
			}

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

		msg, err := buildFinalMessage(contentBuf.String(), reasoningBuf.String(), toolAccum)
		if err != nil {
			send(StreamChunk{Err: err})
			return
		}
		send(StreamChunk{Done: true, Msg: msg, RateLimits: rateLimits, FinishReason: finishReason, Usage: usage})
	}()
	return ch
}

// isHTTPStatusError reports whether err carries the given HTTP status code.
// go-openai wraps HTTP errors as *openai.RequestError (for non-JSON bodies)
// or *openai.APIError (for JSON error bodies); both carry HTTPStatusCode.
func isHTTPStatusError(err error, code int) bool {
	var re *openai.RequestError
	if errors.As(err, &re) {
		return re.HTTPStatusCode == code
	}
	var ae *openai.APIError
	if errors.As(err, &ae) {
		return ae.HTTPStatusCode == code
	}
	return false
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
		// Merge consecutive system messages — some providers (e.g. GitHub Copilot)
		// reject requests with more than one role:"system" message. This can happen
		// with sessions saved before the single-system-message invariant was enforced.
		if m.Role == "system" && len(out) > 0 && out[len(out)-1].Role == "system" {
			out[len(out)-1].Content += "\n\n" + m.Content
			continue
		}
		msg := openai.ChatCompletionMessage{
			Role:       m.Role,
			Content:    m.Content,
			ToolCallID: m.ToolCallID,
		}
		// If the message carries image parts, switch to MultiContent.
		if len(m.Parts) > 0 {
			parts := make([]openai.ChatMessagePart, 0, 1+len(m.Parts))
			if m.Content != "" {
				parts = append(parts, openai.ChatMessagePart{
					Type: openai.ChatMessagePartTypeText,
					Text: m.Content,
				})
			}
			for _, p := range m.Parts {
				var imageURL openai.ChatMessageImageURL
				if p.URL != "" {
					imageURL.URL = p.URL
				} else {
					imageURL.URL = "data:" + p.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(p.Data)
				}
				imageURL.Detail = openai.ImageURLDetailAuto
				parts = append(parts, openai.ChatMessagePart{
					Type:     openai.ChatMessagePartTypeImageURL,
					ImageURL: &imageURL,
				})
			}
			msg.Content = ""
			msg.MultiContent = parts
		}
		for _, tc := range m.ToolCalls {
			raw, _ := json.Marshal(tc.Arguments)
			if string(raw) == "null" {
				raw = []byte("{}")
			}
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
		Role:      m.Role,
		Content:   m.Content,
		Reasoning: m.ReasoningContent,
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

// GetModelInfo probes the provider for model metadata. Tries two strategies:
//
//  1. Ollama-specific: POST <base>/api/show — gives the effective num_ctx.
//  2. OpenAI-compatible GET <base>/models/<model> — used for providers like
//     GitHub Copilot that expose context_window or
//     capabilities.limits.max_context_window_tokens.
//
// The Ollama probe is tried first; if it returns a non-zero MaxTokens that
// result is used. Otherwise the OpenAI models endpoint is tried (only when a
// probeClient is set, i.e. when the client was built with NewWithHTTPClient).
func (c *Client) GetModelInfo(ctx context.Context) (ModelInfo, error) {
	base := strings.TrimSuffix(strings.TrimSuffix(strings.TrimRight(c.endpoint, "/"), "/v1"), "/")
	if base == "" {
		return ModelInfo{}, nil
	}

	// --- Strategy 1: Ollama /api/show ---
	if info := c.probeOllama(ctx, base); info.Context.MaxTokens > 0 {
		return info, nil
	}

	// --- Strategy 2: OpenAI-compatible /models/{id} ---
	if c.probeClient != nil {
		if info := c.probeOpenAIModel(ctx); info.Context.MaxTokens > 0 {
			return info, nil
		}
	}

	return ModelInfo{Model: c.model}, nil
}

// probeOllama tries POST <base>/api/show and returns the context window if found.
func (c *Client) probeOllama(ctx context.Context, base string) ModelInfo {
	type showRequest struct {
		Model string `json:"model"`
	}
	body, err := json.Marshal(showRequest{Model: c.model})
	if err != nil {
		return ModelInfo{}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/show", bytes.NewReader(body))
	if err != nil {
		return ModelInfo{}
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := c.probeClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	resp, err := httpClient.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		return ModelInfo{}
	}
	defer func() { _ = resp.Body.Close() }()

	var payload struct {
		Parameters string         `json:"parameters"`
		ModelInfo  map[string]any `json:"model_info"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return ModelInfo{}
	}

	info := ModelInfo{Model: c.model}
	if n := parseNumCtx(payload.Parameters); n > 0 {
		info.Context.MaxTokens = n
	} else {
		for k, v := range payload.ModelInfo {
			if strings.HasSuffix(k, ".context_length") {
				if f, ok := v.(float64); ok {
					info.Context.MaxTokens = int(f)
					break
				}
			}
		}
	}
	return info
}

// probeOpenAIModel queries <endpoint>/models/<model> and extracts the context
// window size. Handles several provider-specific field layouts:
//   - top-level "context_window"                             (OpenAI standard)
//   - top-level "max_context_window_tokens"                  (some providers)
//   - "capabilities.limits.max_context_window_tokens"        (GitHub Copilot)
func (c *Client) probeOpenAIModel(ctx context.Context) ModelInfo {
	endpoint := strings.TrimRight(c.endpoint, "/")
	modelPath := url.PathEscape(c.model)
	reqURL := endpoint + "/models/" + modelPath

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return ModelInfo{}
	}
	req.Header.Set("Accept", "application/json")

	resp, err := c.probeClient.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		return ModelInfo{}
	}
	defer func() { _ = resp.Body.Close() }()

	var payload map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return ModelInfo{}
	}

	info := ModelInfo{Model: c.model}

	// Check fields in priority order; take the first non-zero value.
	if n := intField(payload, "context_window"); n > 0 {
		info.Context.MaxTokens = n
	} else if n := intField(payload, "max_context_window_tokens"); n > 0 {
		info.Context.MaxTokens = n
	} else if caps, ok := payload["capabilities"].(map[string]any); ok {
		if limits, ok := caps["limits"].(map[string]any); ok {
			if n := intField(limits, "max_context_window_tokens"); n > 0 {
				info.Context.MaxTokens = n
			}
		}
	}
	return info
}

// intField extracts a positive integer from a map field that may be float64 or int.
func intField(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	}
	return 0
}

// parseNumCtx extracts the num_ctx value from an Ollama parameters string.
// The format is one parameter per line: "num_ctx    4096".
// Returns 0 if not present.
func parseNumCtx(params string) int {
	for _, line := range strings.Split(params, "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == "num_ctx" {
			if n, err := strconv.Atoi(fields[1]); err == nil && n > 0 {
				return n
			}
		}
	}
	return 0
}

package ai

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
)

// Request body for POST /responses.
type respRequest struct {
	Model              string          `json:"model"`
	Instructions       string          `json:"instructions,omitempty"`
	Input              json.RawMessage `json:"input"`
	Tools              []respTool      `json:"tools,omitempty"`
	Stream             bool            `json:"stream,omitempty"`
	Reasoning          *respReasoning  `json:"reasoning,omitempty"`
	PreviousResponseID string          `json:"previous_response_id,omitempty"`
	// Store=true asks the Responses API to persist this response so it can be
	// referenced by future requests via previous_response_id.  We set it on
	// initial requests (no previous_response_id) so that subsequent tool-result
	// continuations can omit the full conversation history.
	Store *bool `json:"store,omitempty"`
}

// contextKeyPreviousResponseID is the context key for previous_response_id.
type contextKeyPreviousResponseID struct{}

// WithPreviousResponseID returns a derived context that carries the Responses API
// previous_response_id. When set, CompleteStream references that response and only
// sends new input items (e.g. tool outputs), not the full history.
func WithPreviousResponseID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, contextKeyPreviousResponseID{}, id)
}

func previousResponseIDFromCtx(ctx context.Context) string {
	v, _ := ctx.Value(contextKeyPreviousResponseID{}).(string)
	return v
}

type respReasoning struct {
	Effort string `json:"effort,omitempty"`
}

// respTool uses the flat format required by the Responses API.
type respTool struct {
	Type        string `json:"type"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// Input item types.
type respInputItem struct {
	Type      string          `json:"type"`
	Role      string          `json:"role,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Output    string          `json:"output,omitempty"`
}

type respContentPart struct {
	Type   string      `json:"type"`
	Text   string      `json:"text,omitempty"`
	Source *respSource `json:"source,omitempty"`
}

type respSource struct {
	Type      string `json:"type"`
	URL       string `json:"url,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}

// SSE event payloads.
type respOutputTextDeltaEvent struct {
	Delta string `json:"delta"`
}

type respOutputItemDoneEvent struct {
	Item struct {
		Type      string `json:"type"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"item"`
}

type respCompletedEvent struct {
	Response struct {
		ID     string `json:"id"`
		Status string `json:"status"`
		Output []struct {
			Type string `json:"type"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
		Error *struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	} `json:"response"`
}

// ResponsesClient is a thin HTTP+SSE client for the OpenAI Responses API.
type ResponsesClient struct {
	httpClient       *http.Client
	baseURL          string
	apiKey           string
	model            string
	disableReasoning bool
}

// ResponsesClientOption configures a ResponsesClient at construction time.
type ResponsesClientOption func(*ResponsesClient)

// WithResponsesDisableReasoning prevents reasoning_effort from being sent.
func WithResponsesDisableReasoning() ResponsesClientOption {
	return func(c *ResponsesClient) { c.disableReasoning = true }
}

// NewResponsesClient creates a ResponsesClient. If httpClient is nil, http.DefaultClient is used.
func NewResponsesClient(baseURL, apiKey, model string, httpClient *http.Client, opts ...ResponsesClientOption) *ResponsesClient {
	c := &ResponsesClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		model:      model,
		httpClient: httpClient,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Model returns the current model name (implements Modeler).
func (c *ResponsesClient) Model() string { return c.model }

// SetModel changes the active model (implements ModelManager).
func (c *ResponsesClient) SetModel(item ModelItem) { c.model = item.Model }

// SetDisableReasoning sets whether reasoning is disabled for this client.
func (c *ResponsesClient) SetDisableReasoning(v bool) { c.disableReasoning = v }

// GetModelInfo returns minimal info — the Responses API doesn't expose context window size.
func (c *ResponsesClient) GetModelInfo(_ context.Context) (ModelInfo, error) {
	return ModelInfo{Model: c.model}, nil
}

func (c *ResponsesClient) httpDo(req *http.Request) (*http.Response, error) {
	cl := c.httpClient
	if cl == nil {
		cl = http.DefaultClient
	}
	return cl.Do(req)
}

// ListModels calls GET <baseURL>/models and returns chat-capable model IDs.
func (c *ResponsesClient) ListModels(ctx context.Context) ([]ModelItem, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("responses list models: %w", err)
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.httpDo(req)
	if err != nil {
		return nil, fmt.Errorf("responses list models: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("responses list models: HTTP %d", resp.StatusCode)
	}

	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("responses list models: decode: %w", err)
	}

	var items []ModelItem
	for _, m := range body.Data {
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

// Complete collects all StreamChunks and returns the final Done message.
func (c *ResponsesClient) Complete(ctx context.Context, messages []Message, tools []ToolDefinition) (Message, error) {
	ch := c.CompleteStream(ctx, messages, tools)
	for chunk := range ch {
		if chunk.Err != nil {
			return Message{}, chunk.Err
		}
		if chunk.Done {
			return chunk.Msg, nil
		}
	}
	return Message{}, fmt.Errorf("responses stream ended without a final message")
}

// CompleteStream sends a streaming POST /responses request and streams chunks.
func (c *ResponsesClient) CompleteStream(ctx context.Context, messages []Message, tools []ToolDefinition) <-chan StreamChunk {
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

		instructions, inputItems := toResponsesItems(ctx, messages)

		inputRaw, err := json.Marshal(inputItems)
		if err != nil {
			send(StreamChunk{Err: fmt.Errorf("responses marshal input: %w", err)})
			return
		}

		reqBody := respRequest{
			Model:        c.model,
			Instructions: instructions,
			Input:        json.RawMessage(inputRaw),
			Stream:       true,
		}

		if len(tools) > 0 {
			reqBody.Tools = make([]respTool, 0, len(tools))
			for _, t := range tools {
				reqBody.Tools = append(reqBody.Tools, respTool{
					Type:        "function",
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.InputSchema,
				})
			}
		}

		if !c.disableReasoning {
			if effort := ReasoningEffortFromCtx(ctx); effort != "" {
				reqBody.Reasoning = &respReasoning{Effort: effort}
			}
		}
		if previousID := previousResponseIDFromCtx(ctx); previousID != "" {
			reqBody.PreviousResponseID = previousID
		}
		// Always request storage so every response in the chain can be referenced
		// by the next request via previous_response_id.  Without store:true on
		// follow-up requests the chain breaks after the first tool call because
		// R2 is never persisted and cannot be used as previous_response_id for R3.
		// Servers that don't support storage (most self-hosted ones) safely ignore
		// this field.
		storeTrue := true
		reqBody.Store = &storeTrue

		bodyBytes, err := json.Marshal(reqBody)
		if err != nil {
			send(StreamChunk{Err: fmt.Errorf("responses marshal request: %w", err)})
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/responses", bytes.NewReader(bodyBytes))
		if err != nil {
			send(StreamChunk{Err: fmt.Errorf("responses new request: %w", err)})
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")
		if c.apiKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
		}

		httpResp, err := c.httpDo(httpReq)
		if err != nil {
			send(StreamChunk{Err: fmt.Errorf("responses request: %w", err)})
			return
		}
		defer func() { _ = httpResp.Body.Close() }()

		if httpResp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(httpResp.Body)
			send(StreamChunk{Err: fmt.Errorf("responses API: HTTP %d: %s", httpResp.StatusCode, strings.TrimSpace(string(body)))})
			return
		}

		// toolAccum maps call_id → accumulated tool call data (index-based for ordering).
		type accumToolCall struct {
			idx    int
			callID string
			name   string
			args   string
		}
		toolAccumMap := map[string]*accumToolCall{}
		toolAccumOrder := 0
		var textBuf strings.Builder
		// responseID is captured as early as possible (response.created) so it is
		// available even when the stream terminates via [DONE] without a
		// response.completed event.
		var responseID string

		// buildDoneChunk constructs the final Done StreamChunk from accumulated state.
		buildDoneChunk := func(status string) StreamChunk {
			var finishReason string
			switch {
			case status == "incomplete":
				finishReason = "length"
			case len(toolAccumMap) > 0:
				finishReason = "tool_calls"
			default:
				finishReason = "stop"
			}
			msg := Message{Role: "assistant", Content: textBuf.String()}
			type indexedCall struct {
				idx int
				acc *accumToolCall
			}
			ordered := make([]indexedCall, 0, len(toolAccumMap))
			for _, acc := range toolAccumMap {
				ordered = append(ordered, indexedCall{acc.idx, acc})
			}
			slices.SortFunc(ordered, func(a, b indexedCall) int { return cmp.Compare(a.idx, b.idx) })
			for _, ic := range ordered {
				var args map[string]any
				if s := ic.acc.args; s != "" {
					_ = json.Unmarshal([]byte(s), &args)
				}
				msg.ToolCalls = append(msg.ToolCalls, ToolCall{
					ID:        ic.acc.callID,
					Name:      ic.acc.name,
					Arguments: args,
				})
			}
			return StreamChunk{Done: true, Msg: msg, FinishReason: finishReason, ResponseID: responseID}
		}

		scanner := bufio.NewScanner(httpResp.Body)
		// Increase scanner buffer to handle large SSE events (tool args, long deltas).
		scanner.Buffer(make([]byte, 1<<20), 8<<20) // 1 MiB initial, 8 MiB max
		var eventType string

		for scanner.Scan() {
			select {
			case <-ctx.Done():
				return
			default:
			}

			line := scanner.Text()

			if strings.HasPrefix(line, "event:") {
				eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
				continue
			}

			if !strings.HasPrefix(line, "data:") {
				if line == "" {
					// blank line resets event type
					eventType = ""
				}
				continue
			}

			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))

			if data == "[DONE]" {
				// Stream ended via [DONE]; emit Done chunk with accumulated state.
				// This handles servers that omit the response.completed event.
				send(buildDoneChunk(""))
				return
			}

			// If no explicit event: type, try extracting from JSON "type" field.
			evType := eventType
			if evType == "" {
				var probe struct {
					Type string `json:"type"`
				}
				if jerr := json.Unmarshal([]byte(data), &probe); jerr == nil {
					evType = probe.Type
				}
			}

			switch evType {
			case "response.created", "response.in_progress":
				// Capture response ID as early as possible — it is present in
				// response.created and ensures we have it even if the stream ends
				// via [DONE] without a response.completed event.
				var ev struct {
					Response struct {
						ID string `json:"id"`
					} `json:"response"`
				}
				if jerr := json.Unmarshal([]byte(data), &ev); jerr == nil && ev.Response.ID != "" {
					responseID = ev.Response.ID
				}

			case "response.output_text.delta":
				var ev respOutputTextDeltaEvent
				if jerr := json.Unmarshal([]byte(data), &ev); jerr == nil && ev.Delta != "" {
					textBuf.WriteString(ev.Delta)
					if !send(StreamChunk{Delta: ev.Delta}) {
						return
					}
				}

			case "response.output_item.done":
				var ev respOutputItemDoneEvent
				if jerr := json.Unmarshal([]byte(data), &ev); jerr == nil {
					if ev.Item.Type == "function_call" {
						callID := ev.Item.CallID
						if _, exists := toolAccumMap[callID]; !exists {
							toolAccumMap[callID] = &accumToolCall{
								idx: toolAccumOrder,
							}
							toolAccumOrder++
						}
						acc := toolAccumMap[callID]
						acc.callID = callID
						acc.name = ev.Item.Name
						acc.args = ev.Item.Arguments
					}
				}

			// response.completed is the standard end-of-stream event.
			// response.done and response.incomplete are variants used by some
			// servers / older API versions — treat them identically.
			case "response.completed", "response.done", "response.incomplete":
				var ev respCompletedEvent
				if jerr := json.Unmarshal([]byte(data), &ev); jerr != nil {
					send(StreamChunk{Err: fmt.Errorf("responses parse completed event: %w", jerr)})
					return
				}

				if ev.Response.Error != nil {
					send(StreamChunk{Err: fmt.Errorf("responses API error %s: %s", ev.Response.Error.Code, ev.Response.Error.Message)})
					return
				}

				// response.completed carries the authoritative ID; override what
				// we captured from response.created if the server provides it.
				if ev.Response.ID != "" {
					responseID = ev.Response.ID
				}

				usage := Usage{
					PromptTokens:     ev.Response.Usage.InputTokens,
					CompletionTokens: ev.Response.Usage.OutputTokens,
					TotalTokens:      ev.Response.Usage.InputTokens + ev.Response.Usage.OutputTokens,
				}
				chunk := buildDoneChunk(ev.Response.Status)
				chunk.Usage = usage
				send(chunk)
				return

			case "response.failed", "error":
				var probe struct {
					Message string `json:"message"`
					Error   struct {
						Message string `json:"message"`
						Code    string `json:"code"`
					} `json:"error"`
				}
				if jerr := json.Unmarshal([]byte(data), &probe); jerr == nil {
					msg := probe.Message
					if msg == "" {
						msg = probe.Error.Message
					}
					if msg == "" {
						msg = "unknown error from Responses API"
					}
					send(StreamChunk{Err: fmt.Errorf("responses API: %s", msg)})
					return
				}
				send(StreamChunk{Err: fmt.Errorf("responses API event %q", evType)})
				return
			}

			// Reset event type after each data line.
			eventType = ""
		}

		if serr := scanner.Err(); serr != nil {
			send(StreamChunk{Err: fmt.Errorf("responses stream read: %w", serr)})
		}
	}()
	return ch
}

// toResponsesItems converts messages into Responses API input items.
// System messages are returned as instructions; all others become input items.
// When using previous_response_id the caller (CompleteStreamIncremental) has
// already stripped everything except the new input items before calling
// CompleteStream, so no filtering is needed here.
func toResponsesItems(_ context.Context, messages []Message) (instructions string, items []respInputItem) {
	var instrParts []string

	for _, m := range messages {
		switch m.Role {
		case "system":
			instrParts = append(instrParts, m.Content)
		case "user":
			if len(m.Parts) == 0 {
				raw, _ := json.Marshal(m.Content)
				items = append(items, respInputItem{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(raw),
				})
			} else {
				var parts []respContentPart
				if m.Content != "" {
					parts = append(parts, respContentPart{Type: "input_text", Text: m.Content})
				}
				for _, p := range m.Parts {
					if p.URL != "" {
						parts = append(parts, respContentPart{
							Type:   "input_image",
							Source: &respSource{Type: "url", URL: p.URL},
						})
					} else if p.Data != nil {
						parts = append(parts, respContentPart{
							Type: "input_image",
							Source: &respSource{
								Type:      "base64",
								MediaType: p.MIMEType,
								Data:      base64.StdEncoding.EncodeToString(p.Data),
							},
						})
					}
				}
				raw, _ := json.Marshal(parts)
				items = append(items, respInputItem{
					Type:    "message",
					Role:    "user",
					Content: json.RawMessage(raw),
				})
			}
		case "assistant":
			if m.Content != "" {
				raw, _ := json.Marshal(m.Content)
				items = append(items, respInputItem{
					Type:    "message",
					Role:    "assistant",
					Content: json.RawMessage(raw),
				})
			}
			for _, tc := range m.ToolCalls {
				raw, _ := json.Marshal(tc.Arguments)
				if string(raw) == "null" {
					raw = []byte("{}")
				}
				items = append(items, respInputItem{
					Type:      "function_call",
					CallID:    tc.ID,
					Name:      tc.Name,
					Arguments: string(raw),
				})
			}
		case "tool":
			output := m.Content
			// The Output field uses omitempty, so an empty string would be dropped
			// from JSON entirely, causing servers to reject the item as unrecognisable — use a single space instead.
			if output == "" {
				output = " "
			}
			items = append(items, respInputItem{
				Type:   "function_call_output",
				CallID: m.ToolCallID,
				Output: output,
			})
		}
	}
	instructions = strings.Join(instrParts, "\n\n")
	return instructions, items
}

// StreamCompleterAdapter wraps a StreamCompleter to implement Completer
// by collecting the final Done chunk from the stream.
type StreamCompleterAdapter struct {
	sc StreamCompleter
}

// NewStreamCompleterAdapter creates a StreamCompleterAdapter.
func NewStreamCompleterAdapter(sc StreamCompleter) *StreamCompleterAdapter {
	return &StreamCompleterAdapter{sc: sc}
}

// Complete implements Completer by draining the stream and returning the final message.
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

// Compile-time interface assertions.
var _ Completer = (*ResponsesClient)(nil)
var _ StreamCompleter = (*ResponsesClient)(nil)
var _ ModelManager = (*ResponsesClient)(nil)
var _ ModelInfoGetter = (*ResponsesClient)(nil)
var _ Modeler = (*ResponsesClient)(nil)
var _ Completer = (*StreamCompleterAdapter)(nil)

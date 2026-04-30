package copilot

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"

	"github.com/icedream/werkler/internal/ai"
)

// nonChatKeywordsLocal lists model ID keywords that indicate non-chat models.
// Kept local because ai.nonChatKeywords is unexported.
var nonChatKeywordsLocal = []string{
	"embed", "embedding",
	"whisper", "tts", "transcri",
	"dall-e", "dall_e", "image",
	"moderat",
	"rerank",
}

type copilotModelsResponse struct {
	Data []copilotModelEntry `json:"data"`
}

type copilotModelEntry struct {
	ID           string `json:"id"`
	Capabilities struct {
		SupportedEndpoints []string `json:"supported_endpoints"`
	} `json:"capabilities"`
}

// CopilotMuxClient routes completions to either the /chat/completions or /responses
// backend based on which endpoints the active model supports. It fetches this
// metadata lazily from the Copilot GET /models API on the first completion call.
type CopilotMuxClient struct {
	chatClient      *ai.Client
	responsesClient *ai.ResponsesClient
	httpClient      *http.Client
	baseURL         string

	mu           sync.Mutex
	cacheLoaded  bool
	endpoints    map[string][]string // model ID -> supported_endpoints
	model        string
	useResponses bool
}

// NewCopilotMuxClient creates a CopilotMuxClient. disableReasoning is forwarded to both sub-clients.
func NewCopilotMuxClient(model string, httpClient *http.Client, disableReasoning bool) *CopilotMuxClient {
	chatOpts := []ai.ClientOption{ai.WithNoStreamUsage()}
	if disableReasoning {
		chatOpts = append(chatOpts, ai.WithDisableReasoning())
	}
	var respOpts []ai.ResponsesClientOption
	if disableReasoning {
		respOpts = append(respOpts, ai.WithResponsesDisableReasoning())
	}
	c := &CopilotMuxClient{
		chatClient:      ai.NewWithHTTPClient(CopilotAPIBaseURL, model, httpClient, chatOpts...),
		responsesClient: ai.NewResponsesClient(CopilotAPIBaseURL, "", model, httpClient, respOpts...),
		httpClient:      httpClient,
		baseURL:         CopilotAPIBaseURL,
		model:           model,
		endpoints:       make(map[string][]string),
	}
	return c
}

// Model returns the current model name (implements ai.Modeler).
func (c *CopilotMuxClient) Model() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.model
}

// SetModel updates the active model on the mux and both sub-clients.
func (c *CopilotMuxClient) SetModel(item ai.ModelItem) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.model = item.Model
	c.chatClient.SetModel(item)
	c.responsesClient.SetModel(item)
	c.resolveBackendLocked()
}

// resolveBackendLocked picks the backend based on endpoints cache. Must be called with mu held.
func (c *CopilotMuxClient) resolveBackendLocked() {
	if !c.cacheLoaded {
		return
	}
	eps := c.endpoints[c.model]
	for _, ep := range eps {
		if ep == "/chat/completions" {
			c.useResponses = false
			return
		}
	}
	for _, ep := range eps {
		if ep == "/responses" {
			c.useResponses = true
			return
		}
	}
	c.useResponses = false
}

// ensureCacheLocked fetches the /models endpoint and populates the endpoints cache.
// Must be called with mu held. Returns error on fetch failure (caller should tolerate).
func (c *CopilotMuxClient) ensureCacheLocked(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/models", nil)
	if err != nil {
		return fmt.Errorf("copilot mux models request: %w", err)
	}
	cl := c.httpClient
	if cl == nil {
		cl = http.DefaultClient
	}
	resp, err := cl.Do(req)
	if err != nil {
		return fmt.Errorf("copilot mux models fetch: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("copilot mux models: HTTP %d", resp.StatusCode)
	}

	var body copilotModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return fmt.Errorf("copilot mux models decode: %w", err)
	}

	for _, entry := range body.Data {
		c.endpoints[entry.ID] = entry.Capabilities.SupportedEndpoints
	}
	c.cacheLoaded = true
	c.resolveBackendLocked()
	return nil
}

func (c *CopilotMuxClient) getBackend(ctx context.Context) ai.StreamCompleter {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.cacheLoaded {
		_ = c.ensureCacheLocked(ctx)
	}
	if c.useResponses {
		return c.responsesClient
	}
	return c.chatClient
}

// ListModels fetches models from the Copilot /models endpoint and filters to
// those supporting /chat/completions or /responses.
func (c *CopilotMuxClient) ListModels(ctx context.Context) ([]ai.ModelItem, error) {
	cl := c.httpClient
	if cl == nil {
		cl = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("copilot list models: %w", err)
	}
	resp, err := cl.Do(req)
	if err != nil {
		return nil, fmt.Errorf("copilot list models: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("copilot list models: HTTP %d", resp.StatusCode)
	}

	var body copilotModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("copilot list models decode: %w", err)
	}

	// Update endpoints cache.
	c.mu.Lock()
	for _, entry := range body.Data {
		c.endpoints[entry.ID] = entry.Capabilities.SupportedEndpoints
	}
	if !c.cacheLoaded {
		c.cacheLoaded = true
		c.resolveBackendLocked()
	}
	c.mu.Unlock()

	var items []ai.ModelItem
	for _, entry := range body.Data {
		id := strings.ToLower(entry.ID)
		skip := false
		for _, kw := range nonChatKeywordsLocal {
			if strings.Contains(id, kw) {
				skip = true
				break
			}
		}
		if skip {
			continue
		}
		hasChat := false
		for _, ep := range entry.Capabilities.SupportedEndpoints {
			if ep == "/chat/completions" || ep == "/responses" {
				hasChat = true
				break
			}
		}
		if hasChat {
			items = append(items, ai.ModelItem{Model: entry.ID})
		}
	}

	slices.SortFunc(items, func(a, b ai.ModelItem) int { return cmp.Compare(a.Model, b.Model) })
	return items, nil
}

// Complete delegates to the active backend's Completer.
func (c *CopilotMuxClient) Complete(ctx context.Context, messages []ai.Message, tools []ai.ToolDefinition) (ai.Message, error) {
	backend := c.getBackend(ctx)
	if comp, ok := backend.(ai.Completer); ok {
		return comp.Complete(ctx, messages, tools)
	}
	return ai.Message{}, fmt.Errorf("copilot mux: backend does not implement Completer")
}

// CompleteStream delegates to the active backend's StreamCompleter.
func (c *CopilotMuxClient) CompleteStream(ctx context.Context, messages []ai.Message, tools []ai.ToolDefinition) <-chan ai.StreamChunk {
	return c.getBackend(ctx).CompleteStream(ctx, messages, tools)
}

// GetModelInfo delegates to the active backend's ModelInfoGetter if supported.
func (c *CopilotMuxClient) GetModelInfo(ctx context.Context) (ai.ModelInfo, error) {
	backend := c.getBackend(ctx)
	if g, ok := backend.(ai.ModelInfoGetter); ok {
		return g.GetModelInfo(ctx)
	}
	return ai.ModelInfo{}, nil
}

// Compile-time interface assertions.
var _ ai.StreamCompleter = (*CopilotMuxClient)(nil)
var _ ai.Completer = (*CopilotMuxClient)(nil)
var _ ai.ModelManager = (*CopilotMuxClient)(nil)
var _ ai.ModelInfoGetter = (*CopilotMuxClient)(nil)
var _ ai.Modeler = (*CopilotMuxClient)(nil)

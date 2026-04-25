package copilot

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

const (
	copilotTokenURL = "https://api.github.com/copilot_internal/v2/token"

	// CopilotAPIBaseURL is the base URL for the GitHub Copilot OpenAI-compatible API.
	CopilotAPIBaseURL = "https://api.githubcopilot.com"

	editorVersion = "vscode/1.96.2"
	pluginVersion = "copilot-chat/0.26.7"
	userAgent     = "GitHubCopilotChat/0.26.7"
)

// Transport is an http.RoundTripper that injects the required GitHub Copilot
// authentication headers and manages the short-lived Copilot API token lifecycle.
type Transport struct {
	githubToken  string
	inner        http.RoundTripper // nil = http.DefaultTransport
	mu           sync.Mutex
	cachedToken  string
	cachedExpiry time.Time
}

// NewTransport creates a Transport using the given GitHub access token.
func NewTransport(githubToken string) *Transport {
	return &Transport{githubToken: githubToken}
}

func (t *Transport) base() http.RoundTripper {
	if t.inner != nil {
		return t.inner
	}
	return http.DefaultTransport
}

// copilotToken returns a valid (possibly cached) Copilot API token,
// refreshing it from GitHub if it is absent or within 5 minutes of expiry.
func (t *Transport) copilotToken(ctx context.Context) (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.cachedToken != "" && time.Until(t.cachedExpiry) > 5*time.Minute {
		return t.cachedToken, nil
	}

	tok, exp, err := fetchCopilotToken(ctx, t.githubToken)
	if err != nil {
		return "", err
	}
	t.cachedToken = tok
	t.cachedExpiry = exp
	return tok, nil
}

func (t *Transport) clearToken() {
	t.mu.Lock()
	t.cachedToken = ""
	t.mu.Unlock()
}

// injectHeaders clones the request and sets all required Copilot headers.
func injectHeaders(req *http.Request, copilotTok string) *http.Request {
	clone := req.Clone(req.Context())
	clone.Header.Set("Authorization", "Bearer "+copilotTok)
	clone.Header.Set("Copilot-Integration-Id", "vscode-chat")
	clone.Header.Set("Editor-Version", editorVersion)
	clone.Header.Set("Editor-Plugin-Version", pluginVersion)
	clone.Header.Set("OpenAI-Organization", "github-copilot")
	clone.Header.Set("User-Agent", userAgent)
	clone.Header.Set("X-GitHub-Api-Version", "2023-07-07")
	return clone
}

// RoundTrip implements http.RoundTripper. It injects Copilot authentication
// headers and retries once on 401 with a freshly fetched token.
func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	tok, err := t.copilotToken(req.Context())
	if err != nil {
		return nil, fmt.Errorf("fetching Copilot token: %w", err)
	}

	resp, err := t.base().RoundTrip(injectHeaders(req, tok))
	if err != nil {
		return nil, err
	}

	if resp.StatusCode == http.StatusUnauthorized {
		_ = resp.Body.Close()
		t.clearToken()
		tok, err = t.copilotToken(req.Context())
		if err != nil {
			return nil, fmt.Errorf("refreshing Copilot token after 401: %w", err)
		}
		return t.base().RoundTrip(injectHeaders(req, tok))
	}
	return resp, nil
}

type copilotTokenResponse struct {
	Token     string `json:"token"`
	ExpiresAt int64  `json:"expires_at"` // Unix timestamp
}

// fetchCopilotToken exchanges a GitHub access token for a short-lived Copilot API token.
func fetchCopilotToken(ctx context.Context, githubToken string) (string, time.Time, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, copilotTokenURL, nil)
	if err != nil {
		return "", time.Time{}, err
	}
	req.Header.Set("Authorization", "token "+githubToken)
	req.Header.Set("Editor-Version", editorVersion)
	req.Header.Set("Editor-Plugin-Version", pluginVersion)
	req.Header.Set("Copilot-Integration-Id", "vscode-chat")
	req.Header.Set("OpenAI-Organization", "github-copilot")
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("X-GitHub-Api-Version", "2023-07-07")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", time.Time{}, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return "", time.Time{}, fmt.Errorf(
			"copilot token endpoint returned HTTP %d — check your GitHub token and Copilot subscription",
			resp.StatusCode,
		)
	}

	var out copilotTokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", time.Time{}, fmt.Errorf("decoding Copilot token response: %w", err)
	}
	if out.Token == "" {
		return "", time.Time{}, fmt.Errorf("empty token in Copilot token response")
	}
	return out.Token, time.Unix(out.ExpiresAt, 0), nil
}

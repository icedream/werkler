// Package copilot provides GitHub Copilot API authentication and client support.
//
// Note: The Copilot API used here is reverse-engineered and not an official
// public API. It may change without notice. Use at your own risk.
package copilot

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	deviceCodeURL = "https://github.com/login/device/code"
	tokenURL      = "https://github.com/login/oauth/access_token"

	// DefaultClientID is the GitHub OAuth App client ID used by the
	// VS Code Copilot extension. This is a public client ID, identical
	// to what tools like neovim/copilot.lua use.
	DefaultClientID = "Iv1.b507a08c87ecfe98"

	deviceFlowScope = "read:user"
)

// GitHubToken holds a persisted GitHub OAuth access token.
type GitHubToken struct {
	AccessToken string `json:"access_token"`
	TokenType   string `json:"token_type"`
	Scope       string `json:"scope"`
}

// tokenPath returns the path where the GitHub token is persisted.
func tokenPath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "werkler", "copilot", "github_token.json"), nil
}

// LoadGitHubToken loads the persisted GitHub token.
// Returns nil (no error) if no token has been saved yet.
func LoadGitHubToken() (*GitHubToken, error) {
	path, err := tokenPath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading GitHub token: %w", err)
	}
	var tok GitHubToken
	if err := json.Unmarshal(data, &tok); err != nil {
		return nil, fmt.Errorf("parsing GitHub token: %w", err)
	}
	if tok.AccessToken == "" {
		return nil, nil
	}
	return &tok, nil
}

// SaveGitHubToken atomically persists the GitHub token (mode 0600).
func SaveGitHubToken(tok *GitHubToken) error {
	path, err := tokenPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("creating token directory: %w", err)
	}
	data, err := json.Marshal(tok)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("writing token: %w", err)
	}
	return os.Rename(tmp, path)
}

// DeviceAuthPrompt is called during device flow to show the user the
// verification URL and user code they must enter.
type DeviceAuthPrompt func(verificationURI, userCode string)

// Authenticate runs the GitHub Device Flow to obtain an access token.
// If clientID is empty, DefaultClientID is used.
// The prompt callback is called once to display the verification URL and code.
func Authenticate(ctx context.Context, clientID string, prompt DeviceAuthPrompt) (*GitHubToken, error) {
	if clientID == "" {
		clientID = DefaultClientID
	}

	devResp, err := requestDeviceCode(ctx, clientID)
	if err != nil {
		return nil, fmt.Errorf("requesting device code: %w", err)
	}
	prompt(devResp.VerificationURI, devResp.UserCode)

	interval := time.Duration(devResp.Interval) * time.Second
	if interval < 5*time.Second {
		interval = 5 * time.Second
	}
	deadline := time.Now().Add(time.Duration(devResp.ExpiresIn) * time.Second)

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(interval):
		}

		if time.Now().After(deadline) {
			return nil, fmt.Errorf("device code expired; please try again")
		}

		tok, pollErr, fatal := pollForToken(ctx, clientID, devResp.DeviceCode)
		if fatal != nil {
			return nil, fatal
		}
		switch pollErr {
		case "":
			// success
		case "authorization_pending":
			continue
		case "slow_down":
			interval += 5 * time.Second
			continue
		case "expired_token":
			return nil, fmt.Errorf("device code expired; please try again")
		case "access_denied":
			return nil, fmt.Errorf("authorization denied by user")
		default:
			return nil, fmt.Errorf("authorization error: %s", pollErr)
		}

		if tok != nil {
			return tok, nil
		}
	}
}

type deviceCodeResponse struct {
	DeviceCode      string `json:"device_code"`
	UserCode        string `json:"user_code"`
	VerificationURI string `json:"verification_uri"`
	ExpiresIn       int    `json:"expires_in"`
	Interval        int    `json:"interval"`
}

func requestDeviceCode(ctx context.Context, clientID string) (*deviceCodeResponse, error) {
	body := url.Values{
		"client_id": {clientID},
		"scope":     {deviceFlowScope},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, deviceCodeURL, strings.NewReader(body.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	var out deviceCodeResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decoding device code response: %w", err)
	}
	if out.DeviceCode == "" {
		return nil, fmt.Errorf("empty device code in response (check client ID)")
	}
	return &out, nil
}

// pollForToken makes one poll request.
// Returns (token, errorCode, fatalError).
// errorCode is non-empty when the grant is still pending or has failed with a known code.
func pollForToken(ctx context.Context, clientID, deviceCode string) (*GitHubToken, string, error) {
	body := url.Values{
		"client_id":   {clientID},
		"device_code": {deviceCode},
		"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(body.Encode()))
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer func() { _ = resp.Body.Close() }()

	var result map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, "", fmt.Errorf("decoding token response: %w", err)
	}

	if errCode := result["error"]; errCode != "" {
		return nil, errCode, nil
	}
	if accessToken := result["access_token"]; accessToken != "" {
		return &GitHubToken{
			AccessToken: accessToken,
			TokenType:   result["token_type"],
			Scope:       result["scope"],
		}, "", nil
	}
	return nil, "authorization_pending", nil
}

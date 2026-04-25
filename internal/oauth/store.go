package oauth

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"golang.org/x/oauth2"
)

// StoredSession persists OAuth client credentials and token for a named MCP server.
// Client credentials are stored alongside the token so the token source can be
// reconstructed (for auto-refresh) without requiring a new browser auth flow.
type StoredSession struct {
	ClientID     string     `json:"client_id"`
	ClientSecret string     `json:"client_secret,omitempty"`
	TokenURL     string     `json:"token_url"`
	AccessToken  string     `json:"access_token"`
	RefreshToken string     `json:"refresh_token,omitempty"`
	TokenType    string     `json:"token_type,omitempty"`
	Expiry       *time.Time `json:"expiry,omitempty"`
}

// Token converts the stored fields to an oauth2.Token.
func (s *StoredSession) Token() *oauth2.Token {
	tok := &oauth2.Token{
		AccessToken:  s.AccessToken,
		RefreshToken: s.RefreshToken,
		TokenType:    s.TokenType,
	}
	if s.Expiry != nil {
		tok.Expiry = *s.Expiry
	}
	return tok
}

var unsafeForFilename = regexp.MustCompile(`[^a-zA-Z0-9._-]`)

func sessionPath(serverName string) (string, error) {
	cfgDir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("resolving config dir: %w", err)
	}
	safe := unsafeForFilename.ReplaceAllString(serverName, "_")
	if strings.TrimLeft(safe, "_") == "" {
		safe = "unnamed"
	}
	return filepath.Join(cfgDir, "werkler", "oauth", safe+".json"), nil
}

// LoadSession loads the stored OAuth session for serverName.
// Returns nil, nil if no session file exists yet.
func LoadSession(serverName string) (*StoredSession, error) {
	path, err := sessionPath(serverName)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading OAuth session for %q: %w", serverName, err)
	}
	var sess StoredSession
	if err := json.Unmarshal(data, &sess); err != nil {
		return nil, fmt.Errorf("parsing OAuth session for %q: %w", serverName, err)
	}
	return &sess, nil
}

// SaveSession persists the OAuth session for serverName with 0600 permissions.
func SaveSession(serverName string, sess *StoredSession) error {
	path, err := sessionPath(serverName)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return fmt.Errorf("creating OAuth storage dir: %w", err)
	}
	data, err := json.MarshalIndent(sess, "", "  ")
	if err != nil {
		return fmt.Errorf("marshalling OAuth session: %w", err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		return fmt.Errorf("writing OAuth session for %q: %w", serverName, err)
	}
	return nil
}

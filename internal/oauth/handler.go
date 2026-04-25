package oauth

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/auth"
	"github.com/modelcontextprotocol/go-sdk/oauthex"
	"golang.org/x/oauth2"
)

// AuthURLNotifier is called when the user must open a browser URL to authorize werkler.
// It receives the server's display name and the authorization URL.
// It must block until the OAuth callback is received and return the
// authorization code and state parameter, or an error on cancellation.
type AuthURLNotifier func(ctx context.Context, serverName, authURL string) (code, state string, err error)

// Handler implements [auth.OAuthHandler] with disk-backed token persistence.
// Tokens are loaded on the first [TokenSource] call and persisted after every
// successful authorization or automatic refresh.
type Handler struct {
	serverName  string
	redirectURI string
	notifier    AuthURLNotifier

	mu          sync.Mutex
	oauth2Cfg   *oauth2.Config     // set after successful auth; required for refresh
	tokenSource oauth2.TokenSource // cached token source, nil until first use
}

var _ auth.OAuthHandler = (*Handler)(nil)

// NewHandler creates a Handler for the given MCP server.
// redirectURI must be the redirect URI that was registered for the OAuth client;
// it should come from [CallbackServer.RedirectURI].
func NewHandler(serverName, redirectURI string, notifier AuthURLNotifier) *Handler {
	return &Handler{
		serverName:  serverName,
		redirectURI: redirectURI,
		notifier:    notifier,
	}
}

// TokenSource returns the current token source, loading a persisted session if
// available. Returns nil, nil (no error) when no stored session exists, which
// causes the transport to make an unauthenticated request and trigger [Authorize]
// on the resulting 401.
func (h *Handler) TokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.tokenSource != nil {
		return h.tokenSource, nil
	}

	sess, err := LoadSession(h.serverName)
	if err != nil {
		return nil, fmt.Errorf("loading OAuth session for %q: %w", h.serverName, err)
	}
	if sess == nil || sess.RefreshToken == "" {
		// No persisted session: return nil so the transport gets a 401 and calls Authorize.
		return nil, nil
	}

	cfg := &oauth2.Config{
		ClientID:     sess.ClientID,
		ClientSecret: sess.ClientSecret,
		Endpoint:     oauth2.Endpoint{TokenURL: sess.TokenURL},
	}
	h.oauth2Cfg = cfg
	h.tokenSource = &savingTokenSource{
		inner:      cfg.TokenSource(ctx, sess.Token()),
		serverName: h.serverName,
		cfg:        cfg,
	}
	return h.tokenSource, nil
}

// Authorize runs the full OAuth 2.1 + PKCE authorization code flow:
//  1. Discovers authorization server metadata (via protected resource metadata).
//  2. Performs dynamic client registration.
//  3. Builds an authorization URL and calls the notifier so the user can open it.
//  4. Exchanges the authorization code for tokens and persists the session.
func (h *Handler) Authorize(ctx context.Context, req *http.Request, resp *http.Response) error {
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.Copy(io.Discard, resp.Body)

	// For 403, only re-authenticate when the challenge reports insufficient_scope.
	if resp.StatusCode == http.StatusForbidden {
		challenges, parseErr := oauthex.ParseWWWAuthenticate(resp.Header[http.CanonicalHeaderKey("WWW-Authenticate")])
		if parseErr == nil && errorFromChallenges(challenges) != "insufficient_scope" {
			return nil
		}
	}

	resourceURL := req.URL.String()

	prm, err := discoverPRM(ctx, resp, resourceURL)
	if err != nil {
		return fmt.Errorf("discovering protected resource metadata: %w", err)
	}
	if len(prm.AuthorizationServers) == 0 {
		return fmt.Errorf("no authorization servers found in protected resource metadata for %q", resourceURL)
	}

	asm, err := fetchAuthServerMeta(ctx, prm.AuthorizationServers[0])
	if err != nil {
		return fmt.Errorf("discovering authorization server metadata: %w", err)
	}
	if asm == nil {
		// Fallback per MCP spec 2025-03-26: treat the resource server root as the auth server.
		u, _ := url.Parse(resourceURL)
		u.Path = ""
		root := u.String()
		asm = &oauthex.AuthServerMeta{
			Issuer:                root,
			AuthorizationEndpoint: root + "/authorize",
			TokenEndpoint:         root + "/token",
			RegistrationEndpoint:  root + "/register",
		}
	}

	if asm.RegistrationEndpoint == "" {
		return fmt.Errorf("authorization server for %q does not support dynamic client registration and no pre-registered client is configured", h.serverName)
	}

	regResp, err := oauthex.RegisterClient(ctx, asm.RegistrationEndpoint, &oauthex.ClientRegistrationMetadata{
		RedirectURIs:  []string{h.redirectURI},
		ClientName:    "werkler",
		GrantTypes:    []string{"authorization_code"},
		ResponseTypes: []string{"code"},
	}, http.DefaultClient)
	if err != nil {
		return fmt.Errorf("dynamic client registration for %q: %w", h.serverName, err)
	}

	codeVerifier := oauth2.GenerateVerifier()
	state := rand.Text()

	cfg := &oauth2.Config{
		ClientID:     regResp.ClientID,
		ClientSecret: regResp.ClientSecret,
		Endpoint: oauth2.Endpoint{
			AuthURL:  asm.AuthorizationEndpoint,
			TokenURL: asm.TokenEndpoint,
		},
		RedirectURL: h.redirectURI,
	}

	authURL := cfg.AuthCodeURL(state,
		oauth2.S256ChallengeOption(codeVerifier),
		oauth2.SetAuthURLParam("resource", prm.Resource),
	)

	code, returnedState, err := h.notifier(ctx, h.serverName, authURL)
	if err != nil {
		return fmt.Errorf("authorization for %q cancelled: %w", h.serverName, err)
	}
	if returnedState != state {
		return fmt.Errorf("OAuth state mismatch for %q: possible CSRF", h.serverName)
	}

	clientCtx := context.WithValue(ctx, oauth2.HTTPClient, http.DefaultClient)
	token, err := cfg.Exchange(clientCtx, code,
		oauth2.VerifierOption(codeVerifier),
		oauth2.SetAuthURLParam("resource", prm.Resource),
	)
	if err != nil {
		return fmt.Errorf("token exchange for %q: %w", h.serverName, err)
	}

	sess := storedSessionFromToken(regResp.ClientID, regResp.ClientSecret, asm.TokenEndpoint, token)
	if saveErr := SaveSession(h.serverName, sess); saveErr != nil {
		_, _ = fmt.Fprintf(os.Stderr, "warning: could not save OAuth session for %q: %v\n", h.serverName, saveErr)
	}

	h.mu.Lock()
	h.oauth2Cfg = cfg
	h.tokenSource = &savingTokenSource{
		inner:      cfg.TokenSource(clientCtx, token),
		serverName: h.serverName,
		cfg:        cfg,
	}
	h.mu.Unlock()

	return nil
}

// savingTokenSource wraps an oauth2.TokenSource and persists any new token to disk
// whenever a refresh yields a different access or refresh token.
type savingTokenSource struct {
	inner      oauth2.TokenSource
	serverName string
	cfg        *oauth2.Config

	mu      sync.Mutex
	lastTok *oauth2.Token
}

func (s *savingTokenSource) Token() (*oauth2.Token, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	tok, err := s.inner.Token()
	if err != nil {
		return nil, err
	}
	if s.lastTok == nil || tok.AccessToken != s.lastTok.AccessToken || tok.RefreshToken != s.lastTok.RefreshToken {
		s.lastTok = tok
		sess := storedSessionFromToken(s.cfg.ClientID, s.cfg.ClientSecret, s.cfg.Endpoint.TokenURL, tok)
		_ = SaveSession(s.serverName, sess) // best-effort; failure is non-fatal
	}
	return tok, nil
}

func storedSessionFromToken(clientID, clientSecret, tokenURL string, tok *oauth2.Token) *StoredSession {
	sess := &StoredSession{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		TokenURL:     tokenURL,
		AccessToken:  tok.AccessToken,
		RefreshToken: tok.RefreshToken,
		TokenType:    tok.TokenType,
	}
	if !tok.Expiry.IsZero() {
		t := tok.Expiry
		sess.Expiry = &t
	}
	return sess
}

// discoverPRM finds the protected resource metadata for resourceURL.
// It tries the resource_metadata hint from WWW-Authenticate challenges first,
// then standard well-known paths, and finally falls back to using the server root.
func discoverPRM(ctx context.Context, resp *http.Response, resourceURL string) (*oauthex.ProtectedResourceMetadata, error) {
	challenges, _ := oauthex.ParseWWWAuthenticate(resp.Header[http.CanonicalHeaderKey("WWW-Authenticate")])

	// Hint from WWW-Authenticate.
	for _, c := range challenges {
		if metaURL := c.Params["resource_metadata"]; metaURL != "" {
			prm, err := oauthex.GetProtectedResourceMetadata(ctx, metaURL, resourceURL, http.DefaultClient)
			if err == nil && prm != nil {
				return prm, nil
			}
		}
	}

	u, err := url.Parse(resourceURL)
	if err != nil {
		return nil, fmt.Errorf("invalid resource URL %q: %w", resourceURL, err)
	}

	// Well-known at the MCP endpoint path.
	mu := *u
	mu.Path = "/.well-known/oauth-protected-resource/" + strings.TrimLeft(u.Path, "/")
	if prm, err := oauthex.GetProtectedResourceMetadata(ctx, mu.String(), resourceURL, http.DefaultClient); err == nil && prm != nil {
		return prm, nil
	}

	// Well-known at root.
	rootU := *u
	rootU.Path = ""
	mu.Path = "/.well-known/oauth-protected-resource"
	if prm, err := oauthex.GetProtectedResourceMetadata(ctx, mu.String(), rootU.String(), http.DefaultClient); err == nil && prm != nil {
		return prm, nil
	}

	// Fallback: resource server root is the auth server.
	rootU.Path = ""
	return &oauthex.ProtectedResourceMetadata{
		Resource:             resourceURL,
		AuthorizationServers: []string{rootU.String()},
	}, nil
}

func errorFromChallenges(cs []oauthex.Challenge) string {
	for _, c := range cs {
		if c.Scheme == "bearer" && c.Params["error"] != "" {
			return c.Params["error"]
		}
	}
	return ""
}

// fetchAuthServerMeta fetches authorization server metadata for authServerURL,
// tolerating servers (e.g. Atlassian) whose metadata document reports a canonical
// issuer that differs from the URL used to retrieve it.
//
// The SDK's auth.GetAuthServerMetadata enforces RFC 8414 §3.3 (issuer must match
// the lookup URL). Some deployments violate this by serving metadata at one hostname
// while advertising a different canonical issuer. We handle this by:
//  1. Trying the standard well-known URLs with the SDK (no issuer mismatch expected).
//  2. On issuer-mismatch error: fetch the raw JSON ourselves, read the canonical
//     issuer, and retry the SDK call with that issuer so the validation passes.
func fetchAuthServerMeta(ctx context.Context, authServerURL string) (*oauthex.AuthServerMeta, error) {
	// Happy path: SDK fetch with strict issuer validation.
	asm, err := fetchAuthServerMetaStrict(ctx, authServerURL)
	if err == nil {
		return asm, nil // includes the nil-nil "not found" case
	}

	// On issuer mismatch: read the raw metadata to find the canonical issuer,
	// then retry with that URL so the SDK's issuer check passes.
	if !strings.Contains(err.Error(), "does not match issuer URL") {
		return nil, err
	}

	wellKnown := authServerURL
	u, parseErr := url.Parse(authServerURL)
	if parseErr == nil && u.Path == "" {
		u.Path = "/.well-known/oauth-authorization-server"
		wellKnown = u.String()
	}

	req, reqErr := http.NewRequestWithContext(ctx, http.MethodGet, wellKnown, nil)
	if reqErr != nil {
		return nil, err // return the original error
	}
	resp, doErr := http.DefaultClient.Do(req)
	if doErr != nil || resp.StatusCode != http.StatusOK {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	var raw struct {
		Issuer string `json:"issuer"`
	}
	if jsonErr := json.NewDecoder(resp.Body).Decode(&raw); jsonErr != nil || raw.Issuer == "" {
		return nil, err
	}

	// Retry with the canonical issuer URL the server actually reports.
	return fetchAuthServerMetaStrict(ctx, raw.Issuer)
}

// fetchAuthServerMetaStrict calls the SDK's standard metadata discovery with strict issuer validation.
func fetchAuthServerMetaStrict(ctx context.Context, issuerURL string) (*oauthex.AuthServerMeta, error) {
	u, err := url.Parse(issuerURL)
	if err != nil {
		return nil, fmt.Errorf("invalid auth server URL %q: %w", issuerURL, err)
	}
	// Build the same well-known URLs the SDK uses.
	candidates := []string{}
	base := *u
	if base.Path == "" {
		base.Path = "/.well-known/oauth-authorization-server"
		candidates = append(candidates, base.String())
		base.Path = "/.well-known/openid-configuration"
		candidates = append(candidates, base.String())
	} else {
		orig := base.Path
		base.Path = "/.well-known/oauth-authorization-server/" + strings.TrimLeft(orig, "/")
		candidates = append(candidates, base.String())
		base.Path = "/.well-known/openid-configuration/" + strings.TrimLeft(orig, "/")
		candidates = append(candidates, base.String())
	}
	for _, metaURL := range candidates {
		asm, ferr := oauthex.GetAuthServerMeta(ctx, metaURL, issuerURL, http.DefaultClient)
		if ferr != nil {
			return nil, ferr
		}
		if asm != nil {
			return asm, nil
		}
	}
	return nil, nil
}

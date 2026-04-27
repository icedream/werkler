package oauth

import (
	"context"
	"io"
	"net/http"
	"time"

	"github.com/modelcontextprotocol/go-sdk/oauthex"
)

// ProbeResult describes the OAuth requirements detected for an MCP server endpoint.
type ProbeResult struct {
	// RequiresOAuth is true when the server responded with a 401/403 that
	// included a WWW-Authenticate: Bearer challenge.
	RequiresOAuth bool

	// SupportsDCR is true when the authorization server advertises a
	// registration_endpoint (Dynamic Client Registration, RFC 7591).
	// Only meaningful when RequiresOAuth is true.
	SupportsDCR bool

	// Probed is true when the probe request completed (even if no OAuth was
	// detected).  A false Probed means the server was unreachable or the
	// request timed out; in that case RequiresOAuth and SupportsDCR are
	// indeterminate.
	Probed bool
}

// ProbeOAuth makes a single unauthenticated GET to serverURL to check whether
// the endpoint requires OAuth 2.1 authentication and, if so, whether it
// supports Dynamic Client Registration (DCR).
//
// A 5-second timeout is applied on top of any deadline already in ctx.
// On network errors, timeouts, or unexpected HTTP responses the probe returns
// a zero [ProbeResult] with Probed == false (callers should treat this as
// "unknown" rather than "no OAuth required").
func ProbeOAuth(ctx context.Context, serverURL string) ProbeResult {
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, serverURL, nil)
	if err != nil {
		return ProbeResult{}
	}

	resp, err := (&http.Client{Timeout: 5 * time.Second}).Do(req)
	if err != nil {
		return ProbeResult{}
	}
	// Only drain a small prefix — streamable endpoints may return a
	// long-lived body; we only need the headers.
	_, _ = io.CopyN(io.Discard, resp.Body, 4096)
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusUnauthorized && resp.StatusCode != http.StatusForbidden {
		return ProbeResult{Probed: true}
	}

	challenges, _ := oauthex.ParseWWWAuthenticate(resp.Header[http.CanonicalHeaderKey("WWW-Authenticate")])
	hasBearerChallenge := false
	for _, c := range challenges {
		if c.Scheme == "bearer" {
			hasBearerChallenge = true
			break
		}
	}
	if !hasBearerChallenge {
		return ProbeResult{Probed: true}
	}

	result := ProbeResult{RequiresOAuth: true, Probed: true}

	// Try to discover whether DCR is supported.
	prm, err := discoverPRM(probeCtx, resp, serverURL)
	if err != nil || prm == nil || len(prm.AuthorizationServers) == 0 {
		return result
	}

	asm, err := fetchAuthServerMeta(probeCtx, prm.AuthorizationServers[0])
	if err != nil || asm == nil {
		return result
	}

	result.SupportsDCR = asm.RegistrationEndpoint != ""
	return result
}

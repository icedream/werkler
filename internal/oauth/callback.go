package oauth

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"
)

// CallbackResult holds the values extracted from the OAuth redirect callback.
type CallbackResult struct {
	Code  string
	State string
	Err   error // non-nil when the server returned an error parameter
}

// CallbackServer is a one-shot HTTP server that waits for an OAuth redirect callback.
type CallbackServer struct {
	listener net.Listener
	server   *http.Server
	resultCh chan CallbackResult
}

// StartCallbackServer starts a callback server on a random loopback port and
// returns immediately; the server handles the callback in the background.
func StartCallbackServer() (*CallbackServer, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("starting OAuth callback server: %w", err)
	}
	resultCh := make(chan CallbackResult, 1)
	cs := &CallbackServer{
		listener: ln,
		resultCh: resultCh,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/callback", cs.handleCallback)
	cs.server = &http.Server{
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
	}
	go cs.server.Serve(ln) //nolint:errcheck
	return cs, nil
}

func (s *CallbackServer) handleCallback(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	result := CallbackResult{
		Code:  q.Get("code"),
		State: q.Get("state"),
	}
	if errParam := q.Get("error"); errParam != "" {
		if desc := q.Get("error_description"); desc != "" {
			result.Err = fmt.Errorf("%s: %s", errParam, desc)
		} else {
			result.Err = fmt.Errorf("OAuth error: %s", errParam)
		}
	}
	select {
	case s.resultCh <- result:
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = fmt.Fprintf(w, "<!doctype html><html><body>"+
			"<h2>Authorization complete</h2>"+
			"<p>You may close this window and return to werkler.</p>"+
			"</body></html>")
	default:
		// Duplicate callback request — already handled.
		w.WriteHeader(http.StatusOK)
	}
}

// Port returns the TCP port the callback server is listening on.
func (s *CallbackServer) Port() int {
	return s.listener.Addr().(*net.TCPAddr).Port
}

// RedirectURI returns the full redirect URI for this callback server.
func (s *CallbackServer) RedirectURI() string {
	return fmt.Sprintf("http://127.0.0.1:%d/callback", s.Port())
}

// Wait blocks until an OAuth callback is received or ctx is cancelled.
func (s *CallbackServer) Wait(ctx context.Context) (CallbackResult, error) {
	select {
	case result := <-s.resultCh:
		return result, result.Err
	case <-ctx.Done():
		return CallbackResult{}, ctx.Err()
	}
}

// Close shuts down the callback server.
func (s *CallbackServer) Close() {
	_ = s.server.Close()
}

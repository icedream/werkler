package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"runtime/debug"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/config"
	oauthpkg "github.com/icedream/werkler/internal/oauth"
)

// headerTransport is an http.RoundTripper that injects static headers into every request.
type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone the request so the original is not mutated.
	r := req.Clone(req.Context())
	for k, v := range t.headers {
		r.Header.Set(k, v)
	}
	return t.base.RoundTrip(r)
}

// httpClientWithHeaders returns an *http.Client whose transport injects the given headers.
// Header values are expanded with os.ExpandEnv so $VAR and ${VAR} references are resolved
// at connection time. Returns nil when headers is empty (the transport uses its default client).
func httpClientWithHeaders(headers map[string]string) *http.Client {
	if len(headers) == 0 {
		return nil
	}
	expanded := make(map[string]string, len(headers))
	for k, v := range headers {
		expanded[k] = os.ExpandEnv(v)
	}
	return &http.Client{Transport: &headerTransport{base: http.DefaultTransport, headers: expanded}}
}

// AI-facing tool name. Double underscore keeps names safe for OpenAI's
// function name regex ([a-zA-Z0-9_-]{1,64}).
const toolNameSep = "__"

var unsafeChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// sanitize replaces characters not allowed in OpenAI function names with "_".
func sanitize(s string) string {
	return unsafeChars.ReplaceAllString(s, "_")
}

// serverConn holds an active MCP client session and optional cleanup for
// in-process (builtin) servers.
type serverConn struct {
	name          string
	safeName      string                 // sanitized, used as prefix in AI tool names
	cfg           config.MCPServerConfig // stored for transparent reconnection
	session       *mcp.ClientSession
	serverSession *mcp.ServerSession // non-nil for builtin servers
}

// toolEntry maps an AI-facing tool name back to the originating connection and
// the original (unsanitized) tool name as registered on the MCP server.
type toolEntry struct {
	conn     *serverConn
	origName string
}

// Manager owns all MCP server connections for the lifetime of a chat session.
type Manager struct {
	mu           sync.Mutex
	conns        []*serverConn
	toolMap      map[string]toolEntry // AI-facing name → {conn, original name}
	mcpImpl      *mcp.Implementation
	pendingOAuth []config.MCPServerConfig         // legacy: streamable+oauth servers deferred until first prompt (ConnectOne path)
	configured   []config.MCPServerConfig         // servers registered for lazy connect (AI calls connect_server)
	connecting   map[string]bool                  // display names currently being connected (TOCTOU guard)
	oauthDisplay func(serverName, authURL string) // TUI callback: show auth URL to user; nil in non-interactive contexts
}

// werklerVersion returns the module version from build info, falling back to "dev".
func werklerVersion() string {
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return "dev"
}

// NewManager creates an uninitialised Manager.
func NewManager() *Manager {
	return &Manager{
		mcpImpl:    &mcp.Implementation{Name: "werkler", Version: werklerVersion()},
		toolMap:    make(map[string]toolEntry),
		connecting: make(map[string]bool),
	}
}

// ValidateServerNames checks that all server names are unique after sanitization.
// Call this before Connect or ConnectOne to surface duplicate names early.
func ValidateServerNames(servers []config.MCPServerConfig) error {
	seen := make(map[string]bool, len(servers))
	for _, srv := range servers {
		safe := sanitize(srv.Name)
		if seen[safe] {
			return fmt.Errorf("duplicate MCP server safe-name %q (from %q)", safe, srv.Name)
		}
		seen[safe] = true
	}
	return nil
}

// Register stores servers for lazy on-demand connection without connecting them
// immediately. All servers go into the configured pool so the AI can connect
// them explicitly via connect_server (including OAuth servers).
// Register validates server names upfront and returns an error on duplicates.
func (m *Manager) Register(servers []config.MCPServerConfig) error {
	if err := ValidateServerNames(servers); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	// All servers go to configured regardless of OAuth; the AI connects them
	// on-demand via connect_server (which handles the OAuth flow when needed).
	m.configured = append(m.configured, servers...)
	return nil
}

// ConfiguredServers returns a copy of the servers registered for lazy connection
// that have not yet been connected.
func (m *Manager) ConfiguredServers() []config.MCPServerConfig {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]config.MCPServerConfig, len(m.configured))
	copy(out, m.configured)
	return out
}

// IsConnected reports whether the server with the given display name is active.
func (m *Manager) IsConnected(name string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, c := range m.conns {
		if c.name == name {
			return true
		}
	}
	return false
}

// SetOAuthDisplay sets the callback used to show an OAuth auth URL to the user
// during ConnectByName for OAuth servers. It must be set before the first
// OAuth server connection. In non-interactive contexts (e.g. prompt mode) leave
// it nil — ConnectByName will return a clear error if OAuth interaction is needed.
func (m *Manager) SetOAuthDisplay(fn func(serverName, authURL string)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.oauthDisplay = fn
}

// ConnectByName connects a single server from the configured pool by its display
// name. Returns (true, nil) when the server is already connected.
// OAuth servers are connected inline using the registered oauthDisplay callback;
// returns an error if no callback was set and the token is not cached.
// Safe to call concurrently: a TOCTOU guard prevents two goroutines from
// connecting the same server simultaneously.
func (m *Manager) ConnectByName(ctx context.Context, name string) (alreadyConnected bool, err error) {
	m.mu.Lock()

	// Already connected?
	for _, c := range m.conns {
		if c.name == name {
			m.mu.Unlock()
			return true, nil
		}
	}

	// Already in-flight?
	if m.connecting[name] {
		m.mu.Unlock()
		return false, fmt.Errorf("server %q is already being connected", name)
	}

	// Find in configured pool.
	var found *config.MCPServerConfig
	for i := range m.configured {
		if m.configured[i].Name == name {
			found = &m.configured[i]
			break
		}
	}
	if found == nil {
		// Check legacy pendingOAuth pool (ConnectOne path, e.g. prompt mode).
		for _, p := range m.pendingOAuth {
			if p.Name == name {
				m.mu.Unlock()
				return false, fmt.Errorf("server %q is pending OAuth connection via the legacy flow", name)
			}
		}
		m.mu.Unlock()
		return false, fmt.Errorf("server %q not found in configured servers", name)
	}

	srv := *found
	m.connecting[name] = true
	m.mu.Unlock()

	// Connect outside the lock to avoid blocking other callers.
	var connErr error
	if srv.Transport == config.MCPTransportStreamable && srv.OAuth {
		connErr = m.connectOAuthServer(ctx, srv)
	} else {
		connErr = m.connectOne(ctx, srv, sanitize(srv.Name))
	}

	m.mu.Lock()
	delete(m.connecting, name)
	if connErr == nil {
		// Remove from configured pool now that it is live.
		remaining := m.configured[:0]
		for _, c := range m.configured {
			if c.Name != name {
				remaining = append(remaining, c)
			}
		}
		m.configured = remaining
	}
	m.mu.Unlock()

	if connErr != nil {
		return false, fmt.Errorf("connecting to %q: %w", name, connErr)
	}
	return false, nil
}

// Streamable servers with OAuth enabled are deferred: they are not connected here
// but instead queued for [ConnectPendingOAuth].
// Call Close when done.
func (m *Manager) Connect(ctx context.Context, servers []config.MCPServerConfig) error {
	if err := ValidateServerNames(servers); err != nil {
		return err
	}
	for _, srv := range servers {
		if _, err := m.ConnectOne(ctx, srv); err != nil {
			return fmt.Errorf("connecting to MCP server %q: %w", srv.Name, err)
		}
	}
	return nil
}

// ConnectOne establishes a connection to a single MCP server.
// Streamable+OAuth servers are queued for [ConnectPendingOAuth]; in that case
// deferred=true is returned with a nil error.
// [ValidateServerNames] should be called before ConnectOne to prevent duplicate safe-names.
func (m *Manager) ConnectOne(ctx context.Context, srv config.MCPServerConfig) (deferred bool, err error) {
	if srv.Transport == config.MCPTransportStreamable && srv.OAuth {
		m.mu.Lock()
		m.pendingOAuth = append(m.pendingOAuth, srv)
		m.mu.Unlock()
		return true, nil
	}
	safe := sanitize(srv.Name)
	return false, m.connectOne(ctx, srv, safe)
}

// HasPendingOAuth reports whether there are OAuth MCP servers not yet connected.
func (m *Manager) HasPendingOAuth() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.pendingOAuth) > 0
}

// PendingOAuthNames returns the display names of OAuth servers awaiting connection.
func (m *Manager) PendingOAuthNames() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	names := make([]string, len(m.pendingOAuth))
	for i, s := range m.pendingOAuth {
		names[i] = s.Name
	}
	return names
}

// ConnectPendingOAuth connects all deferred OAuth servers in order.
// This handles servers added via ConnectOne (e.g. prompt mode).
// For TUI lazy-connect, use ConnectByName instead.
// display is called (without blocking) when the user must open a browser URL
// for each server; the manager handles the HTTP callback internally.
// On success the server's tools become available via [Tools] and [CallTool].
func (m *Manager) ConnectPendingOAuth(ctx context.Context, display func(serverName, authURL string)) error {
	m.mu.Lock()
	pending := make([]config.MCPServerConfig, len(m.pendingOAuth))
	copy(pending, m.pendingOAuth)
	m.mu.Unlock()

	m.SetOAuthDisplay(display)
	for _, srv := range pending {
		if err := m.connectOAuthServer(ctx, srv); err != nil {
			return err
		}
		m.mu.Lock()
		remaining := m.pendingOAuth[:0]
		for _, p := range m.pendingOAuth {
			if p.Name != srv.Name {
				remaining = append(remaining, p)
			}
		}
		m.pendingOAuth = remaining
		m.mu.Unlock()
	}
	return nil
}

// connectOAuthServer establishes an OAuth-authenticated MCP connection for srv.
// It uses m.oauthDisplay to show the auth URL when user interaction is required.
// If m.oauthDisplay is nil and the cached token is expired/missing, returns an error.
func (m *Manager) connectOAuthServer(ctx context.Context, srv config.MCPServerConfig) error {
	m.mu.Lock()
	displayFn := m.oauthDisplay
	m.mu.Unlock()

	cbPort := srv.OAuthCallbackPort
	if cbPort == 0 {
		cbPort = 34217
	}
	// Pre-compute the redirect URI from the port without starting the server yet.
	// The callback server is created on-demand inside localNotifier, so it is
	// alive whenever Authorize is actually called — including after the initial
	// connect when a token expires mid-session and re-auth is triggered by a
	// subsequent tool call returning 401.
	redirectURI := fmt.Sprintf("http://localhost:%d/callback", cbPort)

	// localNotifier is called from inside handler.Authorize only when the
	// cached token is absent or expired. Starting the callback server here
	// (rather than once up-front) ensures it is always running when needed.
	localNotifier := func(ctx context.Context, serverName, authURL string) (string, string, error) {
		if displayFn == nil {
			return "", "", fmt.Errorf("server %q requires OAuth browser authentication but no interactive display is available", serverName)
		}
		cbSrv, err := oauthpkg.StartCallbackServer(cbPort)
		if err != nil {
			return "", "", fmt.Errorf("starting OAuth callback server for %q: %w", serverName, err)
		}
		defer cbSrv.Close()
		displayFn(serverName, authURL)
		result, waitErr := cbSrv.Wait(ctx)
		if waitErr != nil {
			return "", "", waitErr
		}
		return result.Code, result.State, nil
	}

	handler := oauthpkg.NewHandler(srv.Name, redirectURI, localNotifier, srv.OAuthClientID, srv.OAuthClientSecret)
	transport := &mcp.StreamableClientTransport{
		Endpoint:     srv.URL,
		HTTPClient:   httpClientWithHeaders(srv.Headers),
		OAuthHandler: handler,
	}
	client := mcp.NewClient(m.mcpImpl, nil)
	session, connErr := client.Connect(ctx, transport, nil)
	if connErr != nil {
		return fmt.Errorf("connecting to OAuth MCP server %q: %w", srv.Name, connErr)
	}

	safe := sanitize(srv.Name)
	m.mu.Lock()
	m.conns = append(m.conns, &serverConn{
		name:     srv.Name,
		safeName: safe,
		cfg:      srv,
		session:  session,
	})
	m.mu.Unlock()
	return nil
}

func (m *Manager) connectOne(ctx context.Context, srv config.MCPServerConfig, safeName string) error {
	var (
		transport     mcp.Transport
		serverSession *mcp.ServerSession
	)

	switch srv.Transport {
	case config.MCPTransportBuiltin:
		clientT, serverT := mcp.NewInMemoryTransports()

		builtinServer, err := newFilesystemServer()
		if err != nil {
			return fmt.Errorf("creating builtin filesystem server: %w", err)
		}
		ss, err := builtinServer.Connect(ctx, serverT, nil)
		if err != nil {
			return fmt.Errorf("starting builtin filesystem server: %w", err)
		}
		serverSession = ss
		transport = clientT

	case config.MCPTransportStdio:
		if srv.Command == "" {
			return fmt.Errorf("stdio transport requires a command")
		}
		cmd := exec.Command(srv.Command, srv.Args...)
		cmd.Env = os.Environ()
		for k, v := range srv.Env {
			cmd.Env = append(cmd.Env, k+"="+os.ExpandEnv(v))
		}
		transport = &mcp.CommandTransport{Command: cmd}

	case config.MCPTransportSSE:
		if srv.URL == "" {
			return fmt.Errorf("sse transport requires a url")
		}
		transport = &mcp.SSEClientTransport{Endpoint: srv.URL}

	case config.MCPTransportStreamable:
		if srv.URL == "" {
			return fmt.Errorf("streamable transport requires a url")
		}
		transport = &mcp.StreamableClientTransport{
			Endpoint:   srv.URL,
			HTTPClient: httpClientWithHeaders(srv.Headers),
		}

	default:
		return fmt.Errorf("unknown transport %q", srv.Transport)
	}

	client := mcp.NewClient(m.mcpImpl, nil)
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		if serverSession != nil {
			if err := serverSession.Close(); err != nil {
				// Non-fatal: log and continue; the client session closure is more important.
				_, _ = fmt.Fprintf(os.Stderr, "warning: closing builtin server session: %v\n", err)
			}
		}
		return err
	}

	m.mu.Lock()
	m.conns = append(m.conns, &serverConn{
		name:          srv.Name,
		safeName:      safeName,
		cfg:           srv,
		session:       session,
		serverSession: serverSession,
	})
	m.mu.Unlock()
	return nil
}

// Tools returns all tools available across connected MCP servers as AI tool definitions.
// Tool names are formatted as "<safe-server-name>__<tool-name>" to satisfy
// OpenAI's function name constraints. The original tool names are stored internally
// so CallTool can dispatch correctly even when names contain unsafe characters.
func (m *Manager) Tools(ctx context.Context) ([]ai.ToolDefinition, error) {
	m.mu.Lock()
	conns := m.conns
	m.mu.Unlock()

	var tools []ai.ToolDefinition
	newMap := make(map[string]toolEntry)
	for _, conn := range conns {
		for t, err := range conn.session.Tools(ctx, nil) {
			if err != nil {
				return nil, fmt.Errorf("listing tools from %q: %w", conn.name, err)
			}
			aiName := conn.safeName + toolNameSep + sanitize(t.Name)
			if len(aiName) > 64 {
				aiName = aiName[:64]
			}
			newMap[aiName] = toolEntry{conn: conn, origName: t.Name}
			tools = append(tools, ai.ToolDefinition{
				Name:        aiName,
				Description: t.Description,
				InputSchema: t.InputSchema,
			})
		}
	}

	m.mu.Lock()
	m.toolMap = newMap
	m.mu.Unlock()

	return tools, nil
}

// ToolNamesForServer returns the AI-prefixed tool names for a specific server
// (identified by its display name). It queries the server directly, bypassing
// the cached toolMap, so callers can use it immediately after connecting.
// Returns nil if the server is not found.
func (m *Manager) ToolNamesForServer(ctx context.Context, name string) ([]string, error) {
	m.mu.Lock()
	var conn *serverConn
	for _, c := range m.conns {
		if c.name == name {
			conn = c
			break
		}
	}
	m.mu.Unlock()
	if conn == nil {
		return nil, nil
	}
	var names []string
	for t, err := range conn.session.Tools(ctx, nil) {
		if err != nil {
			return nil, err
		}
		names = append(names, conn.safeName+toolNameSep+sanitize(t.Name))
	}
	return names, nil
}

// CallTool dispatches a tool call to the appropriate MCP server.
// Tool names must be in the format returned by Tools ("<safe-server-name>__<tool-name>").
// The original (unsanitized) tool name is looked up from the internal map built
// by the most recent Tools() call, so original names with unsafe characters are
// preserved correctly.
// If the call fails with a transport error (and the context is still alive),
// CallTool transparently reconnects the server and retries once.
func (m *Manager) CallTool(ctx context.Context, name string, args map[string]any) (string, error) {
	m.mu.Lock()
	entry, ok := m.toolMap[name]
	m.mu.Unlock()
	if !ok {
		return "", fmt.Errorf("unknown tool %q (call Tools first to populate the tool map)", name)
	}

	result, callErr := entry.conn.session.CallTool(ctx, &mcp.CallToolParams{
		Name:      entry.origName,
		Arguments: args,
	})

	// If the call failed and the context is still active, assume the connection
	// dropped (transport error) and attempt a transparent reconnect + single retry.
	if callErr != nil && ctx.Err() == nil {
		if reconnErr := m.reconnectConn(ctx, entry.conn); reconnErr == nil {
			// Rebuild toolMap so the retry uses the fresh session.
			if _, refreshErr := m.Tools(ctx); refreshErr == nil {
				m.mu.Lock()
				newEntry, newOK := m.toolMap[name]
				m.mu.Unlock()
				if newOK {
					result, callErr = newEntry.conn.session.CallTool(ctx, &mcp.CallToolParams{
						Name:      newEntry.origName,
						Arguments: args,
					})
				}
			}
		}
		// If reconnect or retry failed, callErr still holds the original error.
	}

	if callErr != nil {
		return "", fmt.Errorf("calling tool %q on %q: %w", entry.origName, entry.conn.name, callErr)
	}
	content, isErr := renderResult(result)
	if isErr {
		return "", fmt.Errorf("%s", content)
	}
	return content, nil
}

// reconnectConn closes a dropped connection and establishes a fresh one using
// the same config. OAuth servers are skipped since they need user interaction.
// After a successful reconnect the caller should refresh the tool map via Tools.
// On failure the server config is re-added to configured so connect_server can retry.
func (m *Manager) reconnectConn(ctx context.Context, old *serverConn) error {
	if old.cfg.Transport == config.MCPTransportStreamable && old.cfg.OAuth {
		return fmt.Errorf("server %q uses OAuth — reconnect requires user interaction", old.cfg.Name)
	}

	// Close the stale session (errors are non-fatal; the session is broken anyway).
	_ = old.session.Close()
	if old.serverSession != nil {
		_ = old.serverSession.Close()
	}

	// Remove the stale conn so connectOne does not leave a duplicate.
	m.mu.Lock()
	remaining := make([]*serverConn, 0, len(m.conns))
	for _, c := range m.conns {
		if c != old {
			remaining = append(remaining, c)
		}
	}
	m.conns = remaining
	m.mu.Unlock()

	if err := m.connectOne(ctx, old.cfg, old.safeName); err != nil {
		// Reconnect failed — re-add to configured so connect_server remains available.
		m.mu.Lock()
		m.configured = append(m.configured, old.cfg)
		m.mu.Unlock()
		return err
	}
	return nil
}

// Close shuts down all MCP server connections.
func (m *Manager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, conn := range m.conns {
		if err := conn.session.Close(); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "warning: closing MCP session %q: %v\n", conn.name, err)
		}
		if conn.serverSession != nil {
			if err := conn.serverSession.Close(); err != nil {
				_, _ = fmt.Fprintf(os.Stderr, "warning: closing builtin server session %q: %v\n", conn.name, err)
			}
		}
	}
	m.conns = nil
}

func splitToolName(name string) (server, tool string, ok bool) {
	idx := strings.Index(name, toolNameSep)
	if idx < 0 {
		return "", "", false
	}
	return name[:idx], name[idx+len(toolNameSep):], true
}

// renderResult converts an MCP CallToolResult to a string for feeding back to
// the AI. Text content is concatenated; non-text content is JSON-encoded so the
// model can still reason about it. Returns (content, true) when the MCP server
// set IsError=true so callers can surface the failure appropriately.
func renderResult(result *mcp.CallToolResult) (string, bool) {
	var parts []string
	for _, c := range result.Content {
		switch tc := c.(type) {
		case *mcp.TextContent:
			parts = append(parts, tc.Text)
		default:
			raw, err := json.Marshal(c)
			if err == nil {
				parts = append(parts, string(raw))
			}
		}
	}
	out := strings.Join(parts, "\n")
	if result.IsError {
		if out == "" {
			out = "(tool returned an error with no message)"
		}
		return out, true
	}
	return out, false
}

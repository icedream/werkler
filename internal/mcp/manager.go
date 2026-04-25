package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"regexp"
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
	safeName      string // sanitized, used as prefix in AI tool names
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
	pendingOAuth []config.MCPServerConfig // streamable+oauth servers deferred until first prompt
}

// NewManager creates an uninitialised Manager.
func NewManager() *Manager {
	return &Manager{
		mcpImpl: &mcp.Implementation{Name: "werkler", Version: "v0.1.0"},
		toolMap: make(map[string]toolEntry),
	}
}

// Connect establishes connections to all configured MCP servers.
// Streamable servers with OAuth enabled are deferred: they are not connected here
// but instead queued for [ConnectPendingOAuth].
// Call Close when done.
func (m *Manager) Connect(ctx context.Context, servers []config.MCPServerConfig) error {
	seen := make(map[string]bool)
	for _, srv := range servers {
		safe := sanitize(srv.Name)
		if seen[safe] {
			return fmt.Errorf("duplicate MCP server safe-name %q (from %q)", safe, srv.Name)
		}
		seen[safe] = true

		if srv.Transport == config.MCPTransportStreamable && srv.OAuth {
			m.mu.Lock()
			m.pendingOAuth = append(m.pendingOAuth, srv)
			m.mu.Unlock()
			continue
		}

		if err := m.connectOne(ctx, srv, safe); err != nil {
			return fmt.Errorf("connecting to MCP server %q: %w", srv.Name, err)
		}
	}
	return nil
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
// display is called (without blocking) when the user must open a browser URL
// for each server; the manager handles the HTTP callback internally.
// On success the server's tools become available via [Tools] and [CallTool].
func (m *Manager) ConnectPendingOAuth(ctx context.Context, display func(serverName, authURL string)) error {
	m.mu.Lock()
	pending := make([]config.MCPServerConfig, len(m.pendingOAuth))
	copy(pending, m.pendingOAuth)
	m.mu.Unlock()

	for _, srv := range pending {
		cbSrv, err := oauthpkg.StartCallbackServer()
		if err != nil {
			return fmt.Errorf("starting OAuth callback server for %q: %w", srv.Name, err)
		}

		// localNotifier is called from inside handler.Authorize.
		// It shows the URL to the user and blocks until the browser callback arrives.
		localNotifier := func(ctx context.Context, serverName, authURL string) (string, string, error) {
			display(serverName, authURL)
			result, waitErr := cbSrv.Wait(ctx)
			if waitErr != nil {
				return "", "", waitErr
			}
			return result.Code, result.State, nil
		}

		handler := oauthpkg.NewHandler(srv.Name, cbSrv.RedirectURI(), localNotifier)
		transport := &mcp.StreamableClientTransport{
			Endpoint:     srv.URL,
			HTTPClient:   httpClientWithHeaders(srv.Headers),
			OAuthHandler: handler,
		}
		client := mcp.NewClient(m.mcpImpl, nil)
		session, connErr := client.Connect(ctx, transport, nil)
		cbSrv.Close()
		if connErr != nil {
			return fmt.Errorf("connecting to OAuth MCP server %q: %w", srv.Name, connErr)
		}

		safe := sanitize(srv.Name)
		m.mu.Lock()
		m.conns = append(m.conns, &serverConn{
			name:     srv.Name,
			safeName: safe,
			session:  session,
		})
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

// CallTool dispatches a tool call to the appropriate MCP server.
// Tool names must be in the format returned by Tools ("<safe-server-name>__<tool-name>").
// The original (unsanitized) tool name is looked up from the internal map built
// by the most recent Tools() call, so original names with unsafe characters are
// preserved correctly.
func (m *Manager) CallTool(ctx context.Context, name string, args map[string]any) (string, error) {
	m.mu.Lock()
	entry, ok := m.toolMap[name]
	m.mu.Unlock()
	if !ok {
		return "", fmt.Errorf("unknown tool %q (call Tools first to populate the tool map)", name)
	}

	result, err := entry.conn.session.CallTool(ctx, &mcp.CallToolParams{
		Name:      entry.origName,
		Arguments: args,
	})
	if err != nil {
		return "", fmt.Errorf("calling tool %q on %q: %w", entry.origName, entry.conn.name, err)
	}

	return renderResult(result), nil
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
// model can still reason about it.
func renderResult(result *mcp.CallToolResult) string {
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
		} else {
			out = "Error: " + out
		}
	}
	return out
}

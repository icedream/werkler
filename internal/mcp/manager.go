package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/config"
)

// toolNameSep is the separator used between server name and tool name in the
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

// Manager owns all MCP server connections for the lifetime of a chat session.
type Manager struct {
	mu      sync.Mutex
	conns   []*serverConn
	mcpImpl *mcp.Implementation
}

// NewManager creates an uninitialised Manager.
func NewManager() *Manager {
	return &Manager{
		mcpImpl: &mcp.Implementation{Name: "werkler", Version: "v0.1.0"},
	}
}

// Connect establishes connections to all configured MCP servers.
// Call Close when done.
func (m *Manager) Connect(ctx context.Context, servers []config.MCPServerConfig) error {
	seen := make(map[string]bool)
	for _, srv := range servers {
		safe := sanitize(srv.Name)
		if seen[safe] {
			return fmt.Errorf("duplicate MCP server safe-name %q (from %q)", safe, srv.Name)
		}
		seen[safe] = true
		if err := m.connectOne(ctx, srv, safe); err != nil {
			return fmt.Errorf("connecting to MCP server %q: %w", srv.Name, err)
		}
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
		transport = &mcp.StreamableClientTransport{Endpoint: srv.URL}

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
// OpenAI's function name constraints.
func (m *Manager) Tools(ctx context.Context) ([]ai.ToolDefinition, error) {
	m.mu.Lock()
	conns := m.conns
	m.mu.Unlock()

	var tools []ai.ToolDefinition
	for _, conn := range conns {
		for t, err := range conn.session.Tools(ctx, nil) {
			if err != nil {
				return nil, fmt.Errorf("listing tools from %q: %w", conn.name, err)
			}
			aiName := conn.safeName + toolNameSep + sanitize(t.Name)
			if len(aiName) > 64 {
				aiName = aiName[:64]
			}
			tools = append(tools, ai.ToolDefinition{
				Name:        aiName,
				Description: t.Description,
				InputSchema: t.InputSchema,
			})
		}
	}
	return tools, nil
}

// CallTool dispatches a tool call to the appropriate MCP server.
// Tool names must be in the format returned by Tools ("<safe-server-name>__<tool-name>").
func (m *Manager) CallTool(ctx context.Context, name string, args map[string]any) (string, error) {
	safeName, toolName, ok := splitToolName(name)
	if !ok {
		return "", fmt.Errorf("tool name %q missing server prefix", name)
	}

	m.mu.Lock()
	var conn *serverConn
	for _, c := range m.conns {
		if c.safeName == safeName {
			conn = c
			break
		}
	}
	m.mu.Unlock()

	if conn == nil {
		return "", fmt.Errorf("no MCP server with safe-name %q", safeName)
	}

	rawArgs, err := json.Marshal(args)
	if err != nil {
		return "", fmt.Errorf("marshalling tool arguments: %w", err)
	}

	result, err := conn.session.CallTool(ctx, &mcp.CallToolParams{
		Name:      toolName,
		Arguments: rawArgs,
	})
	if err != nil {
		return "", fmt.Errorf("calling tool %q on %q: %w", toolName, conn.name, err)
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

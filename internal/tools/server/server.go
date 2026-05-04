package server

import (
	"context"
	"fmt"
	"strings"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/config"
	"github.com/icedream/werkler/internal/tools/toolutil"
)

// MCPConnector is the minimal interface the connect_server handler needs from
// the MCP manager.
type MCPConnector interface {
	ConfiguredServers() []config.MCPServerConfig
	ConnectByName(ctx context.Context, name string) (alreadyConnected bool, err error)
	ToolNamesForServer(ctx context.Context, name string) ([]string, error)
}

// Handler holds the server-connection tool handlers.
type Handler struct{ mgr MCPConnector }

// NewHandler creates a Handler. mgr may be nil (disables connect_server).
func NewHandler(mgr MCPConnector) *Handler { return &Handler{mgr: mgr} }

// Tools returns Builtin definitions. Returns an empty slice when no MCP
// servers are configured (connect_server is not registered in that case).
func (h *Handler) Tools() []toolutil.Builtin {
	if h.mgr == nil {
		return nil
	}
	configured := h.mgr.ConfiguredServers()
	if len(configured) == 0 {
		return nil
	}
	nameList := make([]string, len(configured))
	for i, srv := range configured {
		nameList[i] = srv.Name
	}
	def := ai.ToolDefinition{
		Name: "connect_server",
		Description: "Connect to a configured MCP server to make its tools available. " +
			"Call this immediately when the user's request requires tools from that server — " +
			"do not ask for permission first and do not connect servers unrelated to the current task. " +
			"After connecting, the server's tools will be listed in the result.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{
					"type":        "string",
					"description": "Name of the server to connect",
					"enum":        nameList,
				},
			},
			"required": []string{"name"},
		},
	}
	return []toolutil.Builtin{{Def: def, Handle: h.handleConnectServer}}
}

func (h *Handler) handleConnectServer(ctx context.Context, args map[string]any) (string, error) {
	name := toolutil.StringArg(args, "name")
	if name == "" {
		return "", fmt.Errorf("connect_server: name is required")
	}
	already, err := h.mgr.ConnectByName(ctx, name)
	if err != nil {
		return "", err
	}
	if already {
		return fmt.Sprintf("Server %q is already connected.", name), nil
	}
	toolNames, err := h.mgr.ToolNamesForServer(ctx, name)
	if err != nil || len(toolNames) == 0 {
		return fmt.Sprintf("Connected to server %q. New tools from this server are now available.", name), nil
	}
	return fmt.Sprintf(
		"Connected to server %q. You can now call these tools directly:\n- %s",
		name, strings.Join(toolNames, "\n- "),
	), nil
}

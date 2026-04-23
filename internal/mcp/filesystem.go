package mcp

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type readFileParams struct {
	Path string `json:"path"`
}

type listDirectoryParams struct {
	Path string `json:"path"`
}

// newFilesystemServer creates the bundled in-process filesystem MCP server.
// It exposes two tools: read_file and list_directory.
func newFilesystemServer() (*mcp.Server, error) {
	server := mcp.NewServer(&mcp.Implementation{Name: "werkler-filesystem", Version: "v0.1.0"}, nil)

	mcp.AddTool(server,
		&mcp.Tool{
			Name:        "read_file",
			Description: "Read the full contents of a file at the given path.",
		},
		func(ctx context.Context, _ *mcp.CallToolRequest, p readFileParams) (*mcp.CallToolResult, any, error) {
			if p.Path == "" {
				return errorResult("path is required"), nil, nil
			}
			clean := filepath.Clean(p.Path)
			data, err := os.ReadFile(clean)
			if err != nil {
				return errorResult(fmt.Sprintf("read_file: %v", err)), nil, nil
			}
			return &mcp.CallToolResult{
				Content: []mcp.Content{&mcp.TextContent{Text: string(data)}},
			}, nil, nil
		},
	)

	mcp.AddTool(server,
		&mcp.Tool{
			Name:        "list_directory",
			Description: "List the files and directories inside the given directory path.",
		},
		func(ctx context.Context, _ *mcp.CallToolRequest, p listDirectoryParams) (*mcp.CallToolResult, any, error) {
			if p.Path == "" {
				return errorResult("path is required"), nil, nil
			}
			clean := filepath.Clean(p.Path)
			entries, err := os.ReadDir(clean)
			if err != nil {
				return errorResult(fmt.Sprintf("list_directory: %v", err)), nil, nil
			}
			var lines string
			for _, e := range entries {
				suffix := ""
				if e.IsDir() {
					suffix = "/"
				}
				if lines != "" {
					lines += "\n"
				}
				lines += e.Name() + suffix
			}
			return &mcp.CallToolResult{
				Content: []mcp.Content{&mcp.TextContent{Text: lines}},
			}, nil, nil
		},
	)

	return server, nil
}

func errorResult(msg string) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		IsError: true,
		Content: []mcp.Content{&mcp.TextContent{Text: msg}},
	}
}

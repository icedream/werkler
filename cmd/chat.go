package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	mcppkg "github.com/icedream/werkler/internal/mcp"
)

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Start an interactive AI chat session",
	Long: `Start an interactive chat session with the configured AI model.
The AI can use tools provided by connected MCP servers to complete tasks.

Type /exit or press Ctrl-D to quit.`,
	RunE: runChat,
}

func init() {
	rootCmd.AddCommand(chatCmd)
}

func runChat(_ *cobra.Command, _ []string) error {
	if cfg.AI.APIKey == "" {
		return fmt.Errorf("no API key configured — set ai.api_key in config or use --api-key")
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	aiClient := ai.New(cfg.AI.Endpoint, cfg.AI.APIKey, cfg.AI.Model)

	manager := mcppkg.NewManager()
	defer manager.Close()

	if len(cfg.MCP.Servers) == 0 {
		fmt.Fprintln(os.Stderr, "Note: no MCP servers configured — AI will have no tools available.")
	} else {
		fmt.Fprintf(os.Stderr, "Connecting to %d MCP server(s)...\n", len(cfg.MCP.Servers))
		if err := manager.Connect(ctx, cfg.MCP.Servers); err != nil {
			return fmt.Errorf("MCP setup: %w", err)
		}
	}

	loop := chat.NewLoop(aiClient, manager, cfg.MCP.AutoApproveTools, os.Stdin, os.Stdout)
	return loop.Run(ctx)
}

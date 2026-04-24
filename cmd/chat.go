package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/charmbracelet/glamour"
	"github.com/spf13/cobra"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	mcppkg "github.com/icedream/werkler/internal/mcp"
	"github.com/icedream/werkler/internal/ui"
)

var (
	chatPrompt string
)

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Start an interactive AI chat session",
	Long: `Start an interactive chat session with the configured AI model.
The AI can use tools provided by connected MCP servers to complete tasks.

When --prompt is given, run a single non-interactive turn and print the result.
Otherwise the full TUI is launched. Press Ctrl-C to quit.`,
	RunE: runChat,
}

func init() {
	chatCmd.Flags().StringVarP(&chatPrompt, "prompt", "p", "", "Run a single non-interactive prompt and exit")
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

	session := chat.NewSession(manager, cfg.MCP.AutoApproveTools)

	if chatPrompt != "" {
		return runPromptMode(ctx, aiClient, session)
	}
	return runInteractiveMode(ctx, aiClient, session)
}

func runPromptMode(ctx context.Context, aiClient *ai.Client, session *chat.Session) error {
	result, err := chat.RunPrompt(ctx, aiClient, session, chatPrompt, os.Stderr)
	if err != nil {
		return err
	}
	rendered, err := glamour.Render(result, "auto")
	if err != nil {
		fmt.Println(result)
		return nil
	}
	fmt.Print(rendered)
	return nil
}

func runInteractiveMode(ctx context.Context, aiClient *ai.Client, session *chat.Session) error {
	serverNames := make([]string, 0, len(cfg.MCP.Servers))
	for _, s := range cfg.MCP.Servers {
		serverNames = append(serverNames, s.Name)
	}
	return ui.RunTUI(ctx, aiClient, session, cfg.AI.Model, serverNames)
}

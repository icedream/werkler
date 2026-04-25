package cmd

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/charmbracelet/glamour"
	"github.com/spf13/cobra"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	mcppkg "github.com/icedream/werkler/internal/mcp"
	"github.com/icedream/werkler/internal/sessionstore"
	"github.com/icedream/werkler/internal/tools"
	"github.com/icedream/werkler/internal/ui"
)

var (
	chatPrompt    string
	chatVerbose   bool
	chatResume    bool
	chatSessionID string
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
	chatCmd.Flags().BoolVarP(&chatVerbose, "verbose", "v", false, "Print tool calls to stderr (--prompt mode only)")
	chatCmd.Flags().BoolVar(&chatResume, "resume", false, "Open the session picker to resume a previous session")
	chatCmd.Flags().StringVar(&chatSessionID, "session", "", "Resume the session with this ID (or unique prefix)")
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

	toolMgr := tools.New(manager, nil, nil)
	session := chat.NewSession(toolMgr, cfg.MCP.AutoApproveTools, cfg.MCP.AutoApprovePaths)
	toolMgr.SetPathApprover(session)

	store := sessionstore.New(sessionstore.DefaultDir())

	if chatPrompt != "" {
		if session.HasPendingOAuth() {
			names := strings.Join(session.PendingOAuthNames(), ", ")
			return fmt.Errorf("OAuth authentication required for: %s — run `werkler chat` to authenticate interactively", names)
		}
		return runPromptMode(ctx, aiClient, session)
	}

	opts := ui.SessionOptions{Store: store}
	if chatSessionID != "" {
		sess, err := store.LoadByPrefix(chatSessionID)
		if err != nil {
			return fmt.Errorf("loading session: %w", err)
		}
		opts.Initial = sess
		// Re-apply session-approved tools so they persist across resumes.
		for _, t := range sess.ApprovedTools {
			session.ApproveForSession(t)
		}
	} else if chatResume {
		opts.OpenPicker = true
	}

	return runInteractiveMode(ctx, aiClient, session, toolMgr, opts)
}

func runPromptMode(ctx context.Context, aiClient *ai.Client, session *chat.Session) error {
	var progress io.Writer
	if chatVerbose {
		progress = os.Stderr
	}
	result, err := chat.RunPrompt(ctx, aiClient, session, chatPrompt, progress)
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

func runInteractiveMode(ctx context.Context, aiClient *ai.Client, session *chat.Session, toolMgr *tools.Manager, opts ui.SessionOptions) error {
	serverNames := make([]string, 0, len(cfg.MCP.Servers))
	for _, s := range cfg.MCP.Servers {
		serverNames = append(serverNames, s.Name)
	}
	return ui.RunTUI(ctx, aiClient, session, toolMgr, cfg.AI.Model, serverNames, opts)
}

package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/charmbracelet/glamour"
	"github.com/spf13/cobra"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/config"
	"github.com/icedream/werkler/internal/copilot"
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
	chatProvider  string
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
	chatCmd.Flags().StringVar(&chatProvider, "provider", "", "Name of the AI provider to use (overrides ai.active in config)")
	rootCmd.AddCommand(chatCmd)
}

func runChat(_ *cobra.Command, _ []string) error {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	multiClient, displayName, err := buildAIClient()
	if err != nil {
		return err
	}

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
		return runPromptMode(ctx, multiClient, session)
	}

	opts := ui.SessionOptions{Store: store}
	if chatSessionID != "" {
		sess, err := store.LoadByPrefix(chatSessionID)
		if err != nil {
			return fmt.Errorf("loading session: %w", err)
		}
		opts.Initial = sess
		for _, t := range sess.ApprovedTools {
			session.ApproveForSession(t)
		}
	} else if chatResume {
		opts.OpenPicker = true
	}

	return runInteractiveMode(ctx, multiClient, session, toolMgr, displayName, opts)
}

// buildAIClient constructs a MultiClient from the current config, returning
// the client and its initial model display name.
func buildAIClient() (*ai.MultiClient, string, error) {
	providers, err := config.NormalizeProviders(&cfg.AI)
	if err != nil {
		return nil, "", err
	}

	// Determine which provider should be active.
	activeName := config.ActiveProviderName(&cfg.AI, providers)
	if chatProvider != "" {
		activeName = chatProvider
	}

	mc := &ai.MultiClient{}
	for _, p := range providers {
		switch p.Type {
		case config.ProviderTypeOpenAI, "": // empty type defaults to openai
			client := ai.New(p.Endpoint, p.APIKey, p.Model)
			mc.AddProvider(p.Name, client)
		case config.ProviderTypeCopilot:
			tok, loadErr := copilot.LoadGitHubToken()
			if loadErr != nil {
				return nil, "", fmt.Errorf("loading Copilot token for provider %q: %w", p.Name, loadErr)
			}
			if tok == nil {
				return nil, "", fmt.Errorf(
					"GitHub Copilot provider %q is not authenticated — run `werkler auth copilot` first",
					p.Name,
				)
			}
			transport := copilot.NewTransport(tok.AccessToken)
			client := ai.NewWithHTTPClient(copilot.CopilotAPIBaseURL, p.Model, &http.Client{Transport: transport})
			mc.AddProvider(p.Name, client)
		default:
			return nil, "", fmt.Errorf("unknown provider type %q for provider %q", p.Type, p.Name)
		}
	}

	if activeName != "" && !mc.SwitchToProvider(activeName) {
		return nil, "", fmt.Errorf("active provider %q is not configured", activeName)
	}

	return mc, mc.CurrentModelDisplay(), nil
}

func runPromptMode(ctx context.Context, aiClient ai.Completer, session *chat.Session) error {
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

func runInteractiveMode(ctx context.Context, aiClient ai.StreamCompleter, session *chat.Session, toolMgr *tools.Manager, displayName string, opts ui.SessionOptions) error {
	serverNames := make([]string, 0, len(cfg.MCP.Servers))
	for _, s := range cfg.MCP.Servers {
		serverNames = append(serverNames, s.Name)
	}
	return ui.RunTUI(ctx, aiClient, session, toolMgr, displayName, serverNames, opts)
}

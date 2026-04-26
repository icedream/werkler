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
	"github.com/icedream/werkler/internal/skills"
	"github.com/icedream/werkler/internal/todostore"
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

	providers, err := config.NormalizeProviders(&cfg.AI)
	if err != nil {
		return err
	}

	multiClient, displayName, err := buildAIClient(providers)
	if err != nil {
		return err
	}

	manager := mcppkg.NewManager()
	defer manager.Close()

	switch {
	case chatPrompt != "":
		// Prompt mode: connect synchronously before running the prompt.
		if len(cfg.MCP.Servers) > 0 {
			fmt.Fprintf(os.Stderr, "Connecting to %d MCP server(s)...\n", len(cfg.MCP.Servers))
			if err := manager.Connect(ctx, cfg.MCP.Servers); err != nil {
				return fmt.Errorf("MCP setup: %w", err)
			}
		}
	case len(cfg.MCP.Servers) == 0:
		fmt.Fprintln(os.Stderr, "Note: no MCP servers configured — AI will have no tools available.")
	default:
		// TUI mode with servers: validate names upfront, then connect in background.
		if err := mcppkg.ValidateServerNames(cfg.MCP.Servers); err != nil {
			return fmt.Errorf("MCP setup: %w", err)
		}
	}

	toolMgr := tools.New(manager, nil, nil)
	session := chat.NewSession(toolMgr, cfg.MCP.AutoApproveTools, cfg.MCP.AutoApprovePaths)
	toolMgr.SetPathApprover(session)

	if cfg.MCP.AutoApproveCWDRead {
		if cwd, err := os.Getwd(); err == nil {
			session.SetCWDReadPrefix(cwd)
		}
	}

	reviewer, reviewerLabel, err := buildReviewerClient(providers)
	if err != nil {
		return fmt.Errorf("rubber duck reviewer: %w", err)
	}
	if reviewer != nil {
		toolMgr.SetReviewer(reviewer, reviewerLabel)
	}

	// Load skills from the configured directory (default: ~/.agents/skills).
	skillsDir := skills.DefaultDir()
	if cfg.Skills.Dir != "" {
		skillsDir = skills.ExpandTilde(cfg.Skills.Dir)
	}
	loadedSkills, err := skills.LoadDir(skillsDir, os.Stderr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: skills load error: %v\n", err)
	}
	if len(loadedSkills) > 0 {
		fmt.Fprintf(os.Stderr, "Loaded %d skill(s).\n", len(loadedSkills))
		toolMgr.SetSkills(loadedSkills)
	}

	todoStore := todostore.New()
	toolMgr.SetTodoStore(todoStore)

	store := sessionstore.New(sessionstore.DefaultDir())

	if chatPrompt != "" {
		if session.HasPendingOAuth() {
			names := strings.Join(session.PendingOAuthNames(), ", ")
			return fmt.Errorf("OAuth authentication required for: %s — run `werkler chat` to authenticate interactively", names)
		}
		return runPromptMode(ctx, multiClient, session)
	}

	opts := ui.SessionOptions{
		Store:     store,
		Skills:    loadedSkills,
		TodoStore: todoStore,
		PersistToolApproval: func(toolName string) error {
			return config.AppendAutoApproveTool(flagConfigPath, toolName)
		},
		PersistPathApproval: func(path string, _ bool) error {
			return config.AppendAutoApprovePath(flagConfigPath, path)
		},
		PersistMCPServer: func(srv config.MCPServerConfig) error {
			return config.AppendMCPServer(flagConfigPath, srv)
		},
	}
	if len(cfg.MCP.Servers) > 0 {
		opts.MCPManager = manager
		opts.MCPServers = cfg.MCP.Servers
	}
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

// buildProviderClient constructs a single AI client from a ProviderConfig.
func buildProviderClient(p config.ProviderConfig) (*ai.Client, error) {
	switch p.Type {
	case config.ProviderTypeOpenAI, "": // empty type defaults to openai
		return ai.New(p.Endpoint, p.APIKey, p.Model), nil
	case config.ProviderTypeCopilot:
		tok, loadErr := copilot.LoadGitHubToken()
		if loadErr != nil {
			return nil, fmt.Errorf("loading Copilot token for provider %q: %w", p.Name, loadErr)
		}
		if tok == nil {
			return nil, fmt.Errorf(
				"GitHub Copilot provider %q is not authenticated — run `werkler auth copilot` first",
				p.Name,
			)
		}
		transport := copilot.NewTransport(tok.AccessToken)
		return ai.NewWithHTTPClient(copilot.CopilotAPIBaseURL, p.Model, &http.Client{Transport: transport}), nil
	default:
		return nil, fmt.Errorf("unknown provider type %q for provider %q", p.Type, p.Name)
	}
}

// buildAIClient constructs a MultiClient from the given normalized providers,
// returning the client and its initial model display name.
func buildAIClient(providers []config.ProviderConfig) (*ai.MultiClient, string, error) {
	// Determine which provider should be active.
	activeName := config.ActiveProviderName(&cfg.AI, providers)
	if chatProvider != "" {
		activeName = chatProvider
	}

	mc := &ai.MultiClient{}
	for _, p := range providers {
		// --model flag overrides the active provider's model.
		if flagModel != "" && p.Name == activeName {
			p.Model = flagModel
		}
		client, err := buildProviderClient(p)
		if err != nil {
			return nil, "", err
		}
		mc.AddProvider(p.Name, client)
	}

	if activeName != "" && !mc.SwitchToProvider(activeName) {
		return nil, "", fmt.Errorf("active provider %q is not configured", activeName)
	}

	return mc, mc.CurrentModelDisplay(), nil
}

// buildReviewerClient constructs an AI client for rubber duck reviews based on
// the [ai.rubber_duck] config section. Returns nil, "", nil when unconfigured.
func buildReviewerClient(providers []config.ProviderConfig) (ai.Completer, string, error) {
	rd := cfg.AI.RubberDuck
	if !rd.IsConfigured() {
		return nil, "", nil
	}

	var providerCfg config.ProviderConfig

	if rd.Provider != "" {
		// Reference an existing provider by name.
		found := false
		for _, p := range providers {
			if p.Name == rd.Provider {
				providerCfg = p
				found = true
				break
			}
		}
		if !found {
			return nil, "", fmt.Errorf("rubber duck provider %q not found in [[ai.providers]]", rd.Provider)
		}
		if rd.Model != "" {
			providerCfg.Model = rd.Model
		}
	} else {
		// Standalone config: validate required fields.
		if rd.Type == "" {
			rd.Type = config.ProviderTypeOpenAI
		}
		if rd.Type == config.ProviderTypeOpenAI && rd.APIKey == "" {
			return nil, "", fmt.Errorf(
				"[ai.rubber_duck] standalone config requires api_key (or set provider = \"<name>\" to reference an existing provider)",
			)
		}
		endpoint := rd.Endpoint
		if endpoint == "" && rd.Type == config.ProviderTypeOpenAI {
			endpoint = "https://api.openai.com/v1"
		}
		providerCfg = config.ProviderConfig{
			Name:     "rubber_duck",
			Type:     rd.Type,
			Endpoint: endpoint,
			APIKey:   rd.APIKey,
			Model:    rd.Model,
		}
	}

	client, err := buildProviderClient(providerCfg)
	if err != nil {
		return nil, "", err
	}

	label := string(providerCfg.Type) + "/" + providerCfg.Model
	if rd.Provider != "" {
		label = rd.Provider + "/" + providerCfg.Model
	}

	return client, label, nil
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

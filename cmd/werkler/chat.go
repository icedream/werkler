package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/charmbracelet/glamour"
	"github.com/spf13/cobra"

	"github.com/icedream/werkler/internal/agents"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/config"
	"github.com/icedream/werkler/internal/copilot"
	mcppkg "github.com/icedream/werkler/internal/mcp"
	"github.com/icedream/werkler/internal/memorystore"
	"github.com/icedream/werkler/internal/sessionstore"
	"github.com/icedream/werkler/internal/skills"
	"github.com/icedream/werkler/internal/todostore"
	"github.com/icedream/werkler/internal/tools"
	"github.com/icedream/werkler/internal/ui"
)

var (
	chatPrompt          string
	chatVerbose         bool
	chatResume          bool
	chatSessionID       string
	chatProvider        string
	chatAutopilot       bool
	chatAutopilotMaxCyc int
	chatMode            string
	chatAgent           string
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
	chatCmd.Flags().BoolVar(&chatAutopilot, "autopilot", false, "Enable autopilot mode: AI works autonomously until task_complete is called")
	chatCmd.Flags().IntVar(&chatAutopilotMaxCyc, "autopilot-max-cycles", 0, "Maximum autopilot cycles before pausing (0 = use config default)")
	chatCmd.Flags().StringVar(&chatMode, "mode", "", "Activate a named mode preset (e.g. default, plan, document)")
	chatCmd.Flags().StringVar(&chatAgent, "agent", "", "Activate a named custom agent on startup")
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
		// TUI mode with servers: servers are registered and connected lazily.
		// ValidateServerNames is called inside Register at TUI startup.
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

	// Load custom agents from the configured directory (~/.config/werkler/agents).
	agentsDir := agents.DefaultDir()
	loadedAgents, err := agents.LoadDir(agentsDir, os.Stderr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: agents load error: %v\n", err)
	}
	if len(loadedAgents) > 0 {
		fmt.Fprintf(os.Stderr, "Loaded %d agent(s).\n", len(loadedAgents))
	}

	todoStore := todostore.New()
	toolMgr.SetTodoStore(todoStore)

	// Set up cross-session project memory (stored in ~/.config/werkler/memory/).
	var memStore *memorystore.MemoryStore
	if cwd, err := os.Getwd(); err == nil {
		if ms, err := memorystore.New(cwd); err == nil {
			memStore = ms
			toolMgr.SetMemoryStore(ms)
		} else {
			fmt.Fprintf(os.Stderr, "Warning: project memory unavailable: %v\n", err)
		}
	}

	store := sessionstore.New(sessionstore.DefaultDir())

	if chatPrompt != "" {
		if session.HasPendingOAuth() {
			names := strings.Join(session.PendingOAuthNames(), ", ")
			return fmt.Errorf("OAuth authentication required for: %s — run `werkler chat` to authenticate interactively", names)
		}
		return runPromptMode(ctx, multiClient, session, memStore)
	}

	allModes, err := chat.AllModes(cfg.Modes)
	if err != nil {
		return fmt.Errorf("loading modes: %w", err)
	}
	activeMode, modeErr := chat.ResolveMode(chatMode, cfg.Modes)
	if modeErr != nil {
		return fmt.Errorf("unknown mode %q: %w", chatMode, modeErr)
	}
	if cfg.ImplementationMode != "" {
		if _, err := chat.ResolveMode(cfg.ImplementationMode, cfg.Modes); err != nil {
			return fmt.Errorf("implementation_mode: %w", err)
		}
	}

	opts := ui.SessionOptions{
		Store:     store,
		Skills:    loadedSkills,
		Agents:    loadedAgents,
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
		RemoveMCPServer: func(name string) error {
			return config.RemoveMCPServer(flagConfigPath, name)
		},
		Autopilot:          chatAutopilot,
		AutopilotMaxCycles: resolveAutopilotMax(chatAutopilotMaxCyc, cfg.Autopilot.MaxCycles),
		MemoryStore:        memStore,
		ActiveMode:         activeMode,
		AllModes:           allModes,
		ConfiguredModes:    cfg.Modes,
		ImplementationMode: cfg.ImplementationMode,
	}
	// Resolve --agent flag: fail fast if the name is unknown.
	if chatAgent != "" {
		var found *agents.Agent
		for i := range loadedAgents {
			if loadedAgents[i].Name == chatAgent {
				found = &loadedAgents[i]
				break
			}
		}
		if found == nil {
			return fmt.Errorf("agent %q not found in %s", chatAgent, agentsDir)
		}
		opts.InitialAgent = found
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

// resolveAutopilotMax picks the effective autopilot cycle cap:
// flag > config > 0 (TUI will use the default).
func resolveAutopilotMax(flagVal, cfgVal int) int {
	if flagVal > 0 {
		return flagVal
	}
	return cfgVal
}

// buildProviderClient constructs a single AI client from a ProviderConfig.
func buildProviderClient(p config.ProviderConfig) (*ai.Client, error) {
	switch p.Type {
	case config.ProviderTypeOpenAI, "": // empty type defaults to openai
		// Apply the reasoning alias transport so that Ollama-hosted thinking
		// models (which emit delta.reasoning rather than delta.reasoning_content)
		// are displayed correctly. The transform is a no-op for providers that
		// already use reasoning_content or don't emit thinking tokens at all.
		return ai.NewWithTransport(p.Endpoint, p.APIKey, p.Model,
			ai.NewReasoningAliasTransport(nil)), nil
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
		transport := ai.NewReasoningAliasTransport(copilot.NewTransport(tok.AccessToken))
		return ai.NewWithHTTPClient(copilot.CopilotAPIBaseURL, p.Model, &http.Client{Transport: transport},
			ai.WithNoStreamUsage(),
		), nil
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

func runPromptMode(ctx context.Context, aiClient ai.Completer, session *chat.Session, memStore *memorystore.MemoryStore) error {
	activeMode, modeErr := chat.ResolveMode(chatMode, cfg.Modes)
	if modeErr != nil {
		return fmt.Errorf("unknown mode %q: %w", chatMode, modeErr)
	}

	opts := chat.PromptOptions{
		Autopilot:          chatAutopilot,
		AutopilotMaxCycles: resolveAutopilotMax(chatAutopilotMaxCyc, cfg.Autopilot.MaxCycles),
		SystemPromptExtra:  activeMode.SystemPromptExtra,
	}
	if memStore != nil {
		opts.InitialMemory = memStore.BuildInjectionSection()
	}
	if chatVerbose {
		opts.Progress = os.Stderr
	}
	result, err := chat.RunPrompt(ctx, aiClient, session, chatPrompt, opts)
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

package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	toml "github.com/pelletier/go-toml/v2"
	"github.com/spf13/viper"
)

// MCPTransport defines how to connect to an MCP server.
type MCPTransport string

const (
	MCPTransportBuiltin    MCPTransport = "builtin"
	MCPTransportStdio      MCPTransport = "stdio"
	MCPTransportSSE        MCPTransport = "sse"        // legacy SSE (pre-2025 MCP spec)
	MCPTransportStreamable MCPTransport = "streamable" // streamable HTTP (2025 MCP spec)
)

// MCPServerConfig describes a single MCP server to connect to.
type MCPServerConfig struct {
	Name      string       `mapstructure:"name"`
	Transport MCPTransport `mapstructure:"transport"`
	// Hint is an optional human-readable description of what tools this server
	// provides. The AI uses it to decide when to connect to the server lazily.
	Hint string `mapstructure:"hint"`
	// For stdio transport:
	Command string            `mapstructure:"command"`
	Args    []string          `mapstructure:"args"`
	Env     map[string]string `mapstructure:"env"`
	// For sse/streamable transport:
	URL     string            `mapstructure:"url"`
	Headers map[string]string `mapstructure:"headers"` // extra HTTP request headers (e.g. Authorization)
	// OAuth enables OAuth 2.1 + PKCE authentication for streamable transport.
	// Authentication is deferred: the browser flow runs the first time a prompt is submitted.
	OAuth             bool   `mapstructure:"oauth"`
	OAuthClientID     string `mapstructure:"oauth_client_id"`     // pre-registered client ID (required when server has no DCR)
	OAuthClientSecret string `mapstructure:"oauth_client_secret"` // optional; omit for public clients using PKCE
	OAuthCallbackPort int    `mapstructure:"oauth_callback_port"` // local callback port; 0 = random (default: 34217)
}

// MCPConfig holds all MCP-related settings.
type MCPConfig struct {
	Servers          []MCPServerConfig `mapstructure:"servers"`
	AutoApproveTools []string          `mapstructure:"auto_approve_tools"`
	AutoApprovePaths []string          `mapstructure:"auto_approve_paths"`
	// AutoApproveCWDRead controls whether the AI may read any file under the
	// current working directory without per-path approval. Defaults to true.
	AutoApproveCWDRead bool `mapstructure:"auto_approve_cwd_read"`
}

// ProviderType identifies which kind of AI provider a ProviderConfig describes.
type ProviderType string

const (
	ProviderTypeOpenAI  ProviderType = "openai"
	ProviderTypeCopilot ProviderType = "copilot"
)

// ProviderConfig describes a single named AI provider.
type ProviderConfig struct {
	Name     string       `mapstructure:"name"`
	Type     ProviderType `mapstructure:"type"`
	Endpoint string       `mapstructure:"endpoint"` // OpenAI-compatible: base URL
	APIKey   string       `mapstructure:"api_key"`  // OpenAI-compatible: API key
	Model    string       `mapstructure:"model"`    // default model for this provider
	// ContextWindow overrides the auto-detected context window size (tokens).
	// Set this when the provider does not expose context window size via its API
	// (e.g. GitHub Copilot) to enable automatic context compaction.
	// 0 means auto-detect only.
	ContextWindow int `mapstructure:"context_window"`
}

// RubberDuckConfig configures an optional secondary AI model used for critical
// review of plans and implementations. When configured, the AI is offered a
// rubber_duck_review tool it can call to get independent feedback.
type RubberDuckConfig struct {
	// Provider names an existing [[ai.providers]] entry to reuse.
	// When set, Type/Endpoint/APIKey are ignored.
	Provider string `mapstructure:"provider"`
	// Model overrides the provider's default model (optional).
	Model string `mapstructure:"model"`
	// Type, Endpoint, APIKey configure a standalone provider not listed in
	// [[ai.providers]]. Ignored when Provider is set.
	Type     ProviderType `mapstructure:"type"`
	Endpoint string       `mapstructure:"endpoint"`
	APIKey   string       `mapstructure:"api_key"`
}

// IsConfigured reports whether the rubber duck section contains enough
// information to attempt building a reviewer client.
func (r RubberDuckConfig) IsConfigured() bool {
	return r.Provider != "" || r.APIKey != "" || r.Type == ProviderTypeCopilot
}

// AIConfig holds AI provider settings.
// AIConfig holds AI provider settings.
type AIConfig struct {
	// Multi-provider configuration.
	Providers []ProviderConfig `mapstructure:"providers"`
	// Active names which provider to use. Defaults to the first provider.
	Active string `mapstructure:"active"`

	// RubberDuck configures an optional secondary reviewer AI.
	RubberDuck RubberDuckConfig `mapstructure:"rubber_duck"`
}

// SkillsConfig holds settings for loading agent skills.
type SkillsConfig struct {
	// Dir is the directory to scan for skills. Defaults to ~/.agents/skills.
	// A leading ~ is expanded to the user's home directory.
	Dir string `mapstructure:"dir"`
}

// AutopilotConfig holds settings for autonomous operation mode.
type AutopilotConfig struct {
	// MaxCycles is the maximum number of autonomous continuation cycles before
	// pausing and notifying the user. Defaults to 50.
	MaxCycles int `mapstructure:"max_cycles"`
}

// ModeConfig defines a named mode preset that bundles session options together.
// Built-in modes are "default", "plan", and "document".
// User-defined modes can extend a built-in via the Base field.
type ModeConfig struct {
	Name string `mapstructure:"name"`
	// Base optionally names a built-in mode to inherit from.
	Base string `mapstructure:"base"`
	// Color is passed to lipgloss.Color for the TUI border when this mode is active.
	// Accepts a #rrggbb hex string (e.g. "#5f87ff") or a 256-color index (e.g. "69").
	// Empty inherits from the base or uses the default.
	Color string `mapstructure:"color"`
	// SystemPromptExtra is appended to the system prompt when this mode is active.
	SystemPromptExtra string `mapstructure:"system_prompt_extra"`
	// Autopilot, when non-nil, overrides the default autopilot on/off setting.
	Autopilot *bool `mapstructure:"autopilot"`
	// AutopilotMaxCycles overrides the cycle cap (0 = inherit from base or config default).
	AutopilotMaxCycles int `mapstructure:"autopilot_max_cycles"`
	// AutoApproveTools lists additional tool-name globs to auto-approve in this mode.
	AutoApproveTools []string `mapstructure:"auto_approve_tools"`
}

// Config is the root configuration structure.
type Config struct {
	AI        AIConfig        `mapstructure:"ai"`
	MCP       MCPConfig       `mapstructure:"mcp"`
	Skills    SkillsConfig    `mapstructure:"skills"`
	Autopilot AutopilotConfig `mapstructure:"autopilot"`
	Modes     []ModeConfig    `mapstructure:"modes"`
	// ImplementationMode is the name of the mode preset to switch to when the
	// AI calls start_implementation. Empty means use the default mode.
	ImplementationMode string `mapstructure:"implementation_mode"`
}

// DefaultConfigPath returns the default config file path.
func DefaultConfigPath() string {
	cfgDir, err := os.UserConfigDir()
	if err != nil {
		cfgDir = filepath.Join(os.Getenv("HOME"), ".config")
	}
	return filepath.Join(cfgDir, "werkler", "config.toml")
}

// Load reads and merges config from the given file path.
// An empty path uses the default location.
func Load(path string) (*Config, error) {
	v := viper.New()

	// Defaults
	v.SetDefault("mcp.auto_approve_cwd_read", true)

	if path == "" {
		path = DefaultConfigPath()
	}

	v.SetConfigFile(path)
	v.SetConfigType("toml")

	if err := v.ReadInConfig(); err != nil {
		var notFound viper.ConfigFileNotFoundError
		if !errors.As(err, &notFound) && !os.IsNotExist(err) {
			var decodeErr *toml.DecodeError
			if errors.As(err, &decodeErr) {
				row, col := decodeErr.Position()
				return nil, fmt.Errorf("config file %q:%d:%d: %w", path, row, col, err)
			}
			return nil, fmt.Errorf("reading config file %q: %w", path, err)
		}
		// Missing config file is fine — defaults + env vars + flags cover it.
	}

	// Support WERKLER_AI_API_KEY etc. from environment.
	v.SetEnvPrefix("WERKLER")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	cfg := &Config{}
	if err := v.Unmarshal(cfg); err != nil {
		return nil, fmt.Errorf("unmarshalling config: %w", err)
	}

	return cfg, nil
}

// NormalizeProviders validates and returns the list of configured providers.
// Returns an error when no providers are configured or names are duplicated.
func NormalizeProviders(ai *AIConfig) ([]ProviderConfig, error) {
	if len(ai.Providers) == 0 {
		return nil, fmt.Errorf("no AI providers configured — add at least one [[ai.providers]] entry to config")
	}
	seen := make(map[string]bool, len(ai.Providers))
	for _, p := range ai.Providers {
		if p.Name == "" {
			return nil, fmt.Errorf("provider has an empty name")
		}
		if seen[p.Name] {
			return nil, fmt.Errorf("duplicate provider name %q", p.Name)
		}
		seen[p.Name] = true
	}
	return ai.Providers, nil
}

// ActiveProviderName returns the name of the active provider.
// Falls back to the first provider when Active is unset or invalid.
func ActiveProviderName(ai *AIConfig, providers []ProviderConfig) string {
	if ai.Active != "" {
		for _, p := range providers {
			if p.Name == ai.Active {
				return p.Name
			}
		}
	}
	if len(providers) > 0 {
		return providers[0].Name
	}
	return ""
}

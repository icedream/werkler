package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

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
}

// AIConfig holds AI provider settings.
// The legacy flat fields (Endpoint, APIKey, Model) are honoured when
// Providers is empty, so existing single-provider configs keep working.
type AIConfig struct {
	// Legacy flat config (backward compatible with single-provider setups).
	Endpoint string `mapstructure:"endpoint"`
	APIKey   string `mapstructure:"api_key"`
	Model    string `mapstructure:"model"`

	// Multi-provider configuration.
	Providers []ProviderConfig `mapstructure:"providers"`
	// Active names which provider to use. Defaults to the first provider.
	Active string `mapstructure:"active"`
}

// Config is the root configuration structure.
type Config struct {
	AI  AIConfig  `mapstructure:"ai"`
	MCP MCPConfig `mapstructure:"mcp"`
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
	v.SetDefault("ai.endpoint", "https://api.openai.com/v1")
	v.SetDefault("ai.model", "gpt-4o")

	if path == "" {
		path = DefaultConfigPath()
	}

	v.SetConfigFile(path)
	v.SetConfigType("toml")

	if err := v.ReadInConfig(); err != nil {
		var notFound viper.ConfigFileNotFoundError
		if !errors.As(err, &notFound) && !os.IsNotExist(err) {
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

// ApplyOverrides applies non-zero CLI flag values on top of a loaded Config.
// These flags only affect the legacy flat fields and therefore only apply
// in single-provider (non-providers[]) setups.
func ApplyOverrides(cfg *Config, endpoint, apiKey, model string) {
	if endpoint != "" {
		cfg.AI.Endpoint = endpoint
	}
	if apiKey != "" {
		cfg.AI.APIKey = apiKey
	}
	if model != "" {
		cfg.AI.Model = model
	}
}

// NormalizeProviders resolves the effective list of ProviderConfigs from AIConfig.
//
// If Providers is non-empty it is returned as-is (after validation).
// Otherwise a single "openai" provider is synthesised from the legacy flat
// Endpoint / APIKey / Model fields for backward compatibility.
//
// Returns an error when configuration is missing or invalid.
func NormalizeProviders(ai *AIConfig) ([]ProviderConfig, error) {
	if len(ai.Providers) > 0 {
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

	// Legacy single-provider synthesis.
	if ai.APIKey == "" && ai.Endpoint == "" {
		return nil, fmt.Errorf("no AI provider configured — set ai.api_key in config or use --api-key")
	}
	endpoint := ai.Endpoint
	if endpoint == "" {
		endpoint = "https://api.openai.com/v1"
	}
	model := ai.Model
	if model == "" {
		model = "gpt-4o"
	}
	return []ProviderConfig{{
		Name:     "default",
		Type:     ProviderTypeOpenAI,
		Endpoint: endpoint,
		APIKey:   ai.APIKey,
		Model:    model,
	}}, nil
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

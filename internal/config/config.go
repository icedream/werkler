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

// AIConfig holds AI provider settings.
type AIConfig struct {
	Endpoint string `mapstructure:"endpoint"`
	APIKey   string `mapstructure:"api_key"`
	Model    string `mapstructure:"model"`
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

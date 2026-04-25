package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/icedream/werkler/internal/config"
)

var (
	flagConfigPath string
	flagEndpoint   string
	flagAPIKey     string
	flagModel      string

	cfg *config.Config
)

var rootCmd = &cobra.Command{
	Use:   "werkler",
	Short: "AI-assisted software development toolkit",
	Long: `Werkler is an AI-powered CLI tool that helps software developers with
routine tasks: writing and reviewing code, designing software, drafting tickets,
and more.`,
	SilenceUsage:  true,
	SilenceErrors: true,
	PersistentPreRunE: func(_ *cobra.Command, _ []string) error {
		return loadConfig()
	},
}

// Execute runs the root command.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().StringVar(&flagConfigPath, "config", "", "config file (default: $XDG_CONFIG_HOME/werkler/config.toml)")
	rootCmd.PersistentFlags().StringVar(&flagEndpoint, "endpoint", "", "AI API base URL (overrides config)")
	rootCmd.PersistentFlags().StringVar(&flagAPIKey, "api-key", "", "AI API key (overrides config)")
	rootCmd.PersistentFlags().StringVar(&flagModel, "model", "", "model name (overrides config)")
}

func loadConfig() error {
	var err error
	cfg, err = config.Load(flagConfigPath)
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}
	config.ApplyOverrides(cfg, flagEndpoint, flagAPIKey, flagModel)
	return nil
}

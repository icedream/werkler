package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/icedream/werkler/internal/config"
)

var (
	flagConfigPath string
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
	RunE: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			// Check if we are in an interactive terminal.
			// If not, show help instead of launching the TUI.
			if stat, err := os.Stdin.Stat(); err == nil && (stat.Mode()&os.ModeCharDevice) == 0 {
				cmd.Help()
				return nil
			}
		}

		return chatCmd.RunE(cmd, args)
	},
	PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
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
	rootCmd.PersistentFlags().StringVar(&flagModel, "model", "", "model name (overrides config active provider's model)")
}

func loadConfig() error {
	var err error
	cfg, err = config.Load(flagConfigPath)
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}
	return nil
}

package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
)

// providerAuth is the interface that every AI-provider authenticator must satisfy.
// Implementations live in their respective internal/*/authenticator.go files and
// are registered in cmd/auth_providers.go — no changes to this file are needed
// when adding a new provider.
type providerAuth interface {
	// ProviderType returns the provider type name used as the subcommand name
	// and in config (e.g. "copilot").
	ProviderType() string
	// Description returns a short one-line description for `werkler auth <type> --help`.
	Description() string
	// LongDescription returns the full help text (may be empty).
	LongDescription() string
	// IsAuthenticated reports whether valid credentials are already saved.
	IsAuthenticated() (bool, error)
	// Authenticate runs the interactive auth flow, writing prompts/progress to
	// stdout. It must respect context cancellation.
	Authenticate(ctx context.Context) error
	// StatusLine returns a single-line human-readable status string shown by
	// `werkler auth status`.
	StatusLine() string
}

// authProviders is the registry of all known provider authenticators.
// Populated by cmd/auth_providers.go.
var authProviders []providerAuth

var authCmd = &cobra.Command{
	Use:   "auth",
	Short: "Manage authentication for AI providers",
}

var authStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show authentication status for all registered providers",
	RunE:  runAuthStatus,
}

func init() {
	authCmd.AddCommand(authStatusCmd)
	rootCmd.AddCommand(authCmd)
}

// registerAuthProvider adds p to the registry and wires up its cobra subcommand.
// Call this from cmd/auth_providers.go (or any init() that runs before Execute).
func registerAuthProvider(p providerAuth) {
	authProviders = append(authProviders, p)

	var force bool
	sub := &cobra.Command{
		Use:   p.ProviderType(),
		Short: p.Description(),
		Long:  p.LongDescription(),
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runAuthProvider(cmd.Context(), p, force)
		},
	}
	sub.Flags().BoolVar(&force, "force", false, "Re-authenticate even if already authenticated")
	authCmd.AddCommand(sub)
}

func runAuthProvider(ctx context.Context, p providerAuth, force bool) error {
	if !force {
		ok, err := p.IsAuthenticated()
		if err != nil {
			return fmt.Errorf("checking auth status: %w", err)
		}
		if ok {
			fmt.Printf("Already authenticated for %q. Use --force to re-authenticate.\n", p.ProviderType())
			return nil
		}
	}

	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	return p.Authenticate(ctx)
}

func runAuthStatus(_ *cobra.Command, _ []string) error {
	if len(authProviders) == 0 {
		fmt.Println("No auth providers registered.")
		return nil
	}
	for _, p := range authProviders {
		fmt.Println(p.StatusLine())
	}
	return nil
}

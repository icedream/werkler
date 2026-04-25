package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/icedream/werkler/internal/copilot"
)

var authCmd = &cobra.Command{
	Use:   "auth",
	Short: "Manage authentication for AI providers",
}

var authCopilotForce bool

var authCopilotCmd = &cobra.Command{
	Use:   "copilot",
	Short: "Authenticate with GitHub Copilot using the GitHub Device Flow",
	Long: `Authenticate with GitHub Copilot by running the GitHub Device Flow.

You will be shown a URL and a short code. Open the URL in your browser,
enter the code, and approve the authorization. werkler will then save a
GitHub access token to ~/.config/werkler/copilot/github_token.json (mode 0600).

The saved token is used automatically whenever the "copilot" provider is
active. No re-authentication is needed unless the token is revoked.`,
	RunE: runAuthCopilot,
}

var authStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show authentication status for AI providers",
	RunE:  runAuthStatus,
}

func init() {
	authCopilotCmd.Flags().BoolVar(&authCopilotForce, "force", false, "Re-authenticate even if already authenticated")
	authCmd.AddCommand(authCopilotCmd)
	authCmd.AddCommand(authStatusCmd)
	rootCmd.AddCommand(authCmd)
}

func runAuthCopilot(_ *cobra.Command, _ []string) error {
	if !authCopilotForce {
		existing, err := copilot.LoadGitHubToken()
		if err != nil {
			return fmt.Errorf("checking existing token: %w", err)
		}
		if existing != nil {
			fmt.Println("Already authenticated with GitHub Copilot.")
			fmt.Println("Use --force to re-authenticate.")
			return nil
		}
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	fmt.Println("Authenticating with GitHub Copilot via GitHub Device Flow…")
	fmt.Println()

	tok, err := copilot.Authenticate(ctx, "", func(verificationURI, userCode string) {
		fmt.Printf("Open this URL in your browser:\n  %s\n\n", verificationURI)
		fmt.Printf("Then enter the code:  %s\n\n", userCode)
		fmt.Println("Waiting for authorization…")
	})
	if err != nil {
		return fmt.Errorf("authentication failed: %w", err)
	}

	if err := copilot.SaveGitHubToken(tok); err != nil {
		return fmt.Errorf("saving token: %w", err)
	}

	fmt.Println()
	fmt.Println("✓ Successfully authenticated with GitHub Copilot!")
	fmt.Println("  You can now use  type = \"copilot\"  in [[ai.providers]].")
	return nil
}

func runAuthStatus(_ *cobra.Command, _ []string) error {
	tok, err := copilot.LoadGitHubToken()
	if err != nil {
		return fmt.Errorf("checking Copilot token: %w", err)
	}
	if tok != nil {
		fmt.Println("GitHub Copilot: ✓ authenticated")
	} else {
		fmt.Println("GitHub Copilot: ✗ not authenticated  (run `werkler auth copilot` to authenticate)")
	}
	return nil
}

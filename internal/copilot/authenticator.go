package copilot

import (
	"context"
	"fmt"
	"os"
)

// Authenticator implements cmd.providerAuth for the GitHub Copilot provider.
// It drives the Device Flow and persists the GitHub access token.
type Authenticator struct{}

func (Authenticator) ProviderType() string { return "copilot" }

func (Authenticator) Description() string {
	return "Authenticate with GitHub Copilot using the GitHub Device Flow"
}

func (Authenticator) LongDescription() string {
	return `Authenticate with GitHub Copilot by running the GitHub Device Flow.

You will be shown a URL and a short code. Open the URL in your browser,
enter the code, and approve the authorization. werkler will then save a
GitHub access token to ~/.config/werkler/copilot/github_token.json (mode 0600).

The saved token is used automatically whenever the "copilot" provider is
active. No re-authentication is needed unless the token is revoked.`
}

func (Authenticator) IsAuthenticated() (bool, error) {
	tok, err := LoadGitHubToken()
	if err != nil {
		return false, err
	}
	return tok != nil, nil
}

func (Authenticator) Authenticate(ctx context.Context) error {
	fmt.Println("Authenticating with GitHub Copilot via GitHub Device Flow…")
	fmt.Println()

	tok, err := Authenticate(ctx, "", func(verificationURI, userCode string) {
		fmt.Printf("Open this URL in your browser:\n  %s\n\n", verificationURI)
		fmt.Printf("Then enter the code:  %s\n\n", userCode)
		fmt.Println("Waiting for authorization…")
	})
	if err != nil {
		return fmt.Errorf("authentication failed: %w", err)
	}

	if err := SaveGitHubToken(tok); err != nil {
		return fmt.Errorf("saving token: %w", err)
	}

	fmt.Println()
	fmt.Println(`✓ Successfully authenticated with GitHub Copilot!`)
	fmt.Println(`  You can now use  type = "copilot"  in [[ai.providers]].`)
	return nil
}

func (Authenticator) StatusLine() string {
	tok, err := LoadGitHubToken()
	if err != nil {
		return fmt.Sprintf("GitHub Copilot: error reading token (%v)", err)
	}
	if tok == nil {
		return "GitHub Copilot: ✗ not authenticated  (run `werkler auth copilot`)"
	}
	hostname, _ := os.Hostname()
	_ = hostname
	return "GitHub Copilot: ✓ authenticated"
}

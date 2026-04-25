package main

// auth_providers.go registers all provider authenticators with the auth
// subcommand machinery defined in auth.go.
//
// To add a new provider that requires authentication:
//  1. Implement the providerAuth interface (see auth.go) in its own package
//     under internal/<provider>/authenticator.go.
//  2. Add one registerAuthProvider(...) call below.

import "github.com/icedream/werkler/internal/copilot"

func init() {
	registerAuthProvider(copilot.Authenticator{})
}

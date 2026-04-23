## Conventions

### Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

Format: `<type>(<scope>): <description>`

Common types:
- `feat` — new feature
- `fix` — bug fix
- `refactor` — code change that is neither a fix nor a feature
- `docs` — documentation only
- `chore` — maintenance tasks (deps, config, tooling)
- `ci` — changes to CI/CD configuration
- `test` — adding or updating tests

Scopes use the closest package name, especially when it overlaps with a relevant feature
(e.g. `config`, `ai`, `mcp`, `chat`, `cmd`, `ci`). Omit the scope when it doesn't add clarity.

Example: `feat(mcp): add stdio transport support`

### CI/CD

GitHub Actions is used for CI. Workflows live in `.github/workflows/`.

The CI pipeline runs on every push and pull request to `main` and should:
- Build the project (`go build ./...`)
- Run the linter (`golangci-lint run`)
- Run tests (`go test ./...`)

Go version used in CI should match the version declared in `go.mod`.

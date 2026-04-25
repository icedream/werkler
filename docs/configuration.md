# Configuration

Werkler is configured with a TOML file and can be partially overridden via
environment variables or command-line flags.

## Config file location

| Platform | Default path |
|----------|--------------|
| Linux / BSD | `$XDG_CONFIG_HOME/werkler/config.toml` (usually `~/.config/werkler/config.toml`) |
| macOS | `~/Library/Application Support/werkler/config.toml` |
| Windows | `%AppData%\werkler\config.toml` |

Pass a different path with `--config /path/to/config.toml`.

The file does not need to exist — missing keys fall back to defaults or flags.

---

## AI provider

Werkler talks to any OpenAI-compatible chat API.

```toml
[ai]
# Base URL of the API. Defaults to https://api.openai.com/v1.
endpoint = "https://api.openai.com/v1"

# API key sent as Bearer token.
api_key = "sk-..."

# Model name as understood by the provider.
model = "gpt-4o"
```

### Using OpenAI

```toml
[ai]
endpoint = "https://api.openai.com/v1"
api_key  = "sk-..."
model    = "gpt-4o"
```

### Using a local model (Ollama)

Ollama exposes an OpenAI-compatible endpoint on port 11434. No key is required.

```toml
[ai]
endpoint = "http://localhost:11434/v1"
api_key  = "ollama"   # any non-empty string; Ollama ignores it
model    = "llama3.2"
```

### Using a self-hosted cluster (KubeAI / vLLM)

```toml
[ai]
endpoint = "https://kubeai.example.com/openai/v1"
api_key  = "your-token"
model    = "devstral-small-2"
```

### Environment variables

Every `ai.*` key can be set via an environment variable instead of the file,
which is useful for CI or secret managers:

| Key | Environment variable |
|-----|----------------------|
| `ai.endpoint` | `WERKLER_AI_ENDPOINT` |
| `ai.api_key`  | `WERKLER_AI_API_KEY`  |
| `ai.model`    | `WERKLER_AI_MODEL`    |

Environment variables override the config file but are overridden by
command-line flags.

### Command-line flags

```
werkler --api-key sk-... --model gpt-4o chat
werkler --endpoint http://localhost:11434/v1 --model llama3.2 chat
```

---

## Tools (MCP servers)

Werkler gives the AI access to external capabilities through
[Model Context Protocol (MCP)](https://modelcontextprotocol.io) servers.
Each server exposes a set of named tools the AI can call — things like
reading files, running shell commands, querying databases, or calling APIs.

```toml
[mcp]
# Tools the AI may call without asking you first (supports glob patterns).
# Leave empty to always ask for approval.
auto_approve_tools = []

[[mcp.servers]]
name      = "my-server"
transport = "..."
# transport-specific fields follow
```

### Built-in filesystem server

A built-in read/write filesystem server is bundled with Werkler. It does not
require any external process.

```toml
[[mcp.servers]]
name      = "fs"
transport = "builtin"
```

The tools it exposes are prefixed `fs__` (e.g. `fs__read_file`,
`fs__write_file`).

### stdio server

Runs a local MCP server as a child process, communicating over its
standard input/output. This is the most common transport for community MCP
servers.

```toml
[[mcp.servers]]
name      = "git"
transport = "stdio"
command   = "uvx"
args      = ["mcp-server-git", "--repository", "/path/to/repo"]

# Optional extra environment variables for the child process.
# Values support $VAR expansion from the current environment.
[mcp.servers.env]
HOME = "$HOME"
```

> **Tip:** Many community servers are distributed as npm or Python packages.
> `npx -y @modelcontextprotocol/server-filesystem /workspace` and
> `uvx mcp-server-git` are typical invocations.

### SSE server (legacy, pre-2025 MCP spec)

Connects to a remote MCP server using the original Server-Sent Events
transport. Use this for servers that have not yet migrated to the 2025
streamable HTTP spec.

```toml
[[mcp.servers]]
name      = "legacy-api"
transport = "sse"
url       = "https://mcp.example.com/sse"
```

### Streamable HTTP server (2025 MCP spec)

Connects to a remote MCP server using the current streamable HTTP transport.

```toml
[[mcp.servers]]
name      = "cloud-tools"
transport = "streamable"
url       = "https://mcp.example.com/mcp"
```

Extra HTTP request headers can be injected with a `[mcp.servers.headers]` block.
This is useful for servers that authenticate via a static token in the
`Authorization` header. Values support `$VAR` and `${VAR}` environment variable
expansion:

```toml
[[mcp.servers]]
name      = "cloud-tools"
transport = "streamable"
url       = "https://mcp.example.com/mcp"

[mcp.servers.headers]
Authorization = "Bearer $CLOUD_TOOLS_TOKEN"
```

### Streamable HTTP server with OAuth authentication

Some remote MCP servers — such as
[Atlassian Rovo](https://support.atlassian.com/atlassian-rovo-mcp-server/docs/getting-started-with-the-atlassian-remote-mcp-server/)
— require OAuth 2.1 authentication. Add `oauth = true` to the server block:

```toml
[[mcp.servers]]
name      = "atlassian"
transport = "streamable"
url       = "https://mcp.atlassian.com/v1/mcp"
oauth     = true
```

`oauth` is only valid with `transport = "streamable"`.

#### How authentication works

OAuth-enabled servers are not connected at startup. The first time you submit
a prompt that requires those tools, Werkler pauses and shows you a browser
link directly in the chat:

```
To connect to atlassian, open this URL in your browser:
https://auth.atlassian.com/authorize?...

Waiting for authorization…
```

1. Open the URL in your browser and complete the login flow.
2. Your browser is redirected to a local callback URL that Werkler is
   listening on (`http://127.0.0.1:<random-port>/callback`).
3. Werkler exchanges the authorization code for tokens, then connects the
   server and proceeds with your original prompt automatically.

Tokens (including refresh tokens) are stored encrypted-at-rest in
`~/.config/werkler/oauth/<server-name>.json` (mode `0600`). Subsequent
werkler sessions reuse the stored token and refresh it automatically —
you will not be prompted again unless the refresh token expires or is
revoked.

> **Note:** Because the OAuth flow requires a browser, it cannot run in
> non-interactive (`--prompt`) mode. If you try, Werkler will print an error
> and exit. Run `werkler chat` once first to authenticate, then use
> `--prompt` freely afterwards.

---

## GitHub MCP server

[GitHub's MCP server](https://github.com/github/github-mcp-server) gives the
AI access to repositories, issues, pull requests, GitHub Actions, and more.

There are two ways to run it: a **remote** hosted server and a **local** server
that you run yourself.

### Remote server — OAuth (recommended)

The remote server at `https://api.githubcopilot.com/mcp/` supports standard
OAuth 2.1, but GitHub does not support Dynamic Client Registration — you must
create a **GitHub OAuth App** first (free, takes about a minute):

1. Go to **GitHub → Settings → Developer settings → OAuth Apps → New OAuth App**
2. Set **Application name** to anything (e.g. `werkler`)
3. Set **Homepage URL** to `http://localhost`
4. Set **Authorization callback URL** to `http://localhost`
   (werkler uses a random local port; GitHub accepts any `localhost` redirect)
5. Click **Register application**, then note the **Client ID**
6. Click **Generate a new client secret** and note the secret

Then configure werkler:

```toml
[[mcp.servers]]
name              = "github"
transport         = "streamable"
url               = "https://api.githubcopilot.com/mcp/"
oauth             = true
oauth_client_id   = "$GITHUB_MCP_CLIENT_ID"
oauth_client_secret = "$GITHUB_MCP_CLIENT_SECRET"
```

Set the environment variables before starting werkler:

```sh
export GITHUB_MCP_CLIENT_ID=Ov23li...
export GITHUB_MCP_CLIENT_SECRET=...
werkler chat
```

The first time you submit a prompt, werkler shows an authorization URL in the
chat. After you approve in the browser, the token is saved and future sessions
re-use it automatically.

### Remote server — Personal Access Token

If you prefer not to use the browser OAuth flow, create a
[GitHub PAT](https://github.com/settings/tokens) with the scopes your use case
requires (e.g. `repo`, `read:org`) and pass it as a header. Use an environment
variable so the token is not stored in the config file:

```toml
[[mcp.servers]]
name      = "github"
transport = "streamable"
url       = "https://api.githubcopilot.com/mcp/"

[mcp.servers.headers]
Authorization = "Bearer $GITHUB_TOKEN"
```

Then set the token in your shell before starting werkler:

```sh
export GITHUB_TOKEN=ghp_your_token_here
werkler chat
```

### Local server — stdio (Docker)

If you do not have a Copilot subscription, or want full control, run the
server locally. The official Docker image is the easiest option:

```toml
[[mcp.servers]]
name      = "github"
transport = "stdio"
command   = "docker"
args      = ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
             "ghcr.io/github/github-mcp-server"]

[mcp.servers.env]
GITHUB_PERSONAL_ACCESS_TOKEN = "$GITHUB_PERSONAL_ACCESS_TOKEN"
```

Set the token in your shell before starting werkler:

```sh
export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_token_here
werkler chat
```

### Local server — stdio (binary)

Download the pre-built binary from the
[releases page](https://github.com/github/github-mcp-server/releases) and
point werkler at it:

```toml
[[mcp.servers]]
name      = "github"
transport = "stdio"
command   = "/usr/local/bin/github-mcp-server"
args      = ["stdio"]

[mcp.servers.env]
GITHUB_PERSONAL_ACCESS_TOKEN = "$GITHUB_PERSONAL_ACCESS_TOKEN"
```

### Auto-approving GitHub tools

Most GitHub read operations are safe to auto-approve. Write operations (creating
issues, pushing code, etc.) should stay under manual review.

```toml
[mcp]
auto_approve_tools = [
  "github__get_*",
  "github__list_*",
  "github__search_*",
]
```

---

## Tool approval

Every time the AI wants to call a tool, Werkler asks for your approval —
unless the tool matches an entry in `auto_approve_tools`. Approval choices
in the interactive TUI:

| Key | Effect |
|-----|--------|
| `y` | Allow this call once |
| `a` | Allow this tool for the rest of the session |
| `n` | Deny this call (the AI is told it was denied) |

In non-interactive mode (`--prompt`), only tools that match
`auto_approve_tools` are called. All others receive a denial message so the
AI can respond gracefully.

### Glob patterns

`auto_approve_tools` entries are matched with standard shell glob syntax
against the full tool name (`<server>__<tool>`):

```toml
[mcp]
auto_approve_tools = [
  "fs__read_file",       # exactly this one tool
  "fs__list_*",          # all listing tools on the fs server
  "git__*",              # every tool on the git server
  "*",                   # approve everything (use with care)
]
```

---

## Full example config

```toml
[ai]
endpoint = "https://api.openai.com/v1"
api_key  = "sk-..."
model    = "gpt-4o"

[mcp]
auto_approve_tools = [
  "fs__read_file",
  "fs__list_directory",
  "git__log",
  "git__diff",
]

# Built-in filesystem server
[[mcp.servers]]
name      = "fs"
transport = "builtin"

# Git tools via a local stdio server
[[mcp.servers]]
name      = "git"
transport = "stdio"
command   = "uvx"
args      = ["mcp-server-git", "--repository", "."]

# A remote tool server
[[mcp.servers]]
name      = "search"
transport = "streamable"
url       = "https://mcp.example.com/mcp"

# Atlassian Rovo (requires OAuth — browser login on first use)
[[mcp.servers]]
name      = "atlassian"
transport = "streamable"
url       = "https://mcp.atlassian.com/v1/mcp"
oauth     = true

# GitHub MCP server (OAuth — requires a pre-registered GitHub OAuth App)
[[mcp.servers]]
name                = "github"
transport           = "streamable"
url                 = "https://api.githubcopilot.com/mcp/"
oauth               = true
oauth_client_id     = "$GITHUB_MCP_CLIENT_ID"
oauth_client_secret = "$GITHUB_MCP_CLIENT_SECRET"
```

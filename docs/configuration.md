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
If the file contains a TOML syntax error, werkler will report the exact line and
column number, e.g. `config.toml:12:5: ...`, to help you locate the problem.

## AI provider

Werkler supports multiple AI providers simultaneously. You can switch between
them in the model picker (`ctrl+p`) or by running `werkler chat --provider=name`.

### Multi-provider config

Define providers as an array of tables and name the active one:

```toml
[ai]
active = "copilot"   # which provider to use by default

[[ai.providers]]
name     = "openai"
type     = "openai"
endpoint = "https://api.openai.com/v1"
api_key  = "sk-..."
model    = "gpt-4o"

[[ai.providers]]
name  = "copilot"
type  = "copilot"
model = "gpt-4o"   # default model; overridable in the TUI
```

Provider types:

| `type`    | Description |
|-----------|-------------|
| `openai`  | Any OpenAI-compatible API (OpenAI, Ollama, vLLM, KubeAI, etc.) |
| `copilot` | GitHub Copilot (requires `werkler auth copilot` first) |

When multiple providers are configured you can switch between them live in the
TUI model picker (`ctrl+p`): models from all providers appear in a flat list
as `ProviderName: model-id`.

### Using OpenAI

```toml
[[ai.providers]]
name     = "openai"
type     = "openai"
endpoint = "https://api.openai.com/v1"
api_key  = "sk-..."
model    = "gpt-4o"
```

### Using a local model (Ollama)

Ollama exposes an OpenAI-compatible endpoint on port 11434. No key is required.

```toml
[[ai.providers]]
name     = "ollama"
type     = "openai"
endpoint = "http://localhost:11434/v1"
api_key  = "ollama"   # any non-empty string; Ollama ignores it
model    = "llama3.2"
```

### Using a self-hosted cluster (KubeAI / vLLM)

```toml
[[ai.providers]]
name     = "kubeai"
type     = "openai"
endpoint = "https://kubeai.example.com/openai/v1"
api_key  = "your-token"
model    = "devstral-small-2"
```

### Using GitHub Copilot

GitHub Copilot is available as a native provider (type `copilot`). It gives
access to all models enabled on your Copilot plan — GPT-4o, Claude, Gemini,
and others — through the same OpenAI-compatible API.

> **Note:** The Copilot API used here is reverse-engineered and not an
> official public API. It may change without notice. Use responsibly and in
> accordance with [GitHub's Acceptable Use Policies](https://docs.github.com/site-policy/acceptable-use-policies/github-acceptable-use-policies).

#### Step 1 — Authenticate

```sh
werkler auth copilot
```

This runs the GitHub Device Flow. You will be shown a short code and a URL.
Open the URL in your browser, enter the code, and approve the authorization.
The resulting GitHub token is saved to
`~/.config/werkler/copilot/github_token.json` (mode `0600`). Future sessions
reuse it automatically — re-authentication is only needed if you revoke the token.

#### Step 2 — Configure

```toml
[[ai.providers]]
name  = "copilot"
type  = "copilot"
model = "gpt-4o"   # default model (can be changed in the TUI)
```

#### Step 3 — (Optional) use alongside another provider

```toml
[ai]
active = "copilot"

[[ai.providers]]
name     = "openai"
type     = "openai"
endpoint = "https://api.openai.com/v1"
api_key  = "sk-..."
model    = "gpt-4o"

[[ai.providers]]
name  = "copilot"
type  = "copilot"
model = "claude-sonnet-4-5"
```

#### Authentication management

| Command | Description |
|---------|-------------|
| `werkler auth copilot` | Authenticate (device flow) |
| `werkler auth copilot --force` | Re-authenticate even if already authenticated |
| `werkler auth status` | Show authentication status for all providers |

### Command-line flags

```
werkler --model gpt-4o chat            # override active provider's model
werkler chat --provider=copilot        # switch active provider
```

The `--model` flag overrides the model for the active provider. Use `--provider`
to select a different named provider from `[[ai.providers]]`.

## Mode presets

Named modes bundle a system-prompt extension, an optional TUI border color,
and autopilot/tool-approval overrides. See [modes.md](modes.md) for the full
reference.

Quick example — an "autopilot-plan" mode that combines the built-in `plan`
prompt with autopilot pre-enabled:

```toml
[[modes]]
name                 = "plan-autopilot"
base                 = "plan"
autopilot            = true
autopilot_max_cycles = 30
```

## Rubber duck reviewer

Werkler can optionally route planning and code-review requests to a second AI
model — a "rubber duck" — before the main model implements anything. When this
is configured, the main AI receives a `rubber_duck_review` tool it can invoke to
get independent critical feedback on a plan, piece of code, or reasoning.

The reviewer is entirely separate from the main chat model and can be a
different model from the same or a different provider. The AI decides when to
use it; the tool is auto-approved (no confirmation prompt) since you explicitly
configured the destination.

### Reference an existing provider

Point the rubber duck at a provider already listed in `[[ai.providers]]`, with
an optional model override:

```toml
[ai.rubber_duck]
provider = "copilot"          # must match a [[ai.providers]] name
model    = "claude-opus-4-5"  # optional: override the provider's default model
```

### Standalone provider

Configure a separate AI provider just for reviews without adding it to the main
`[[ai.providers]]` list:

```toml
[ai.rubber_duck]
type     = "openai"
endpoint = "https://api.openai.com/v1"
api_key  = "sk-..."
model    = "o3"
```

For Copilot (shares the token from `werkler auth copilot`):

```toml
[ai.rubber_duck]
type  = "copilot"
model = "claude-opus-4-5"
```

### Fields

| Field | Description |
|-------|-------------|
| `provider` | Name of an existing `[[ai.providers]]` entry to reuse. When set, `type`/`endpoint`/`api_key` are ignored. |
| `model` | Model to use. Overrides the referenced provider's default when `provider` is set; required for standalone configs. |
| `type` | Provider type for standalone config: `openai` or `copilot`. Defaults to `openai` when omitted. |
| `endpoint` | Base URL for standalone OpenAI-compatible providers. Defaults to `https://api.openai.com/v1`. |
| `api_key` | API key for standalone OpenAI-compatible providers. Required when `type = "openai"`. |

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

### Lazy connection

MCP servers are **not connected at startup**. Instead, they are registered
with the AI as a list of "available but not yet connected" servers. The AI
calls the built-in `connect_server` tool whenever it decides a server's tools
would help with the current task, and the connection happens at that point.

This keeps startup instant and avoids connecting servers the AI never ends up
needing.

#### Helping the AI decide: the `hint` field

Without a hint, the AI only knows a server by name and has to guess what tools
it might provide. A short `hint` tells it up front:

```toml
[[mcp.servers]]
name      = "github"
transport = "streamable"
url       = "https://api.githubcopilot.com/mcp/"
hint      = "GitHub — search repos, read issues and PRs, create comments"
```

The hint is injected into the system prompt before every AI turn. Keep it to
one sentence. If no hint is set, the server name alone is shown.

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
# GitHub MCP via PAT
[[mcp.servers]]
name      = "github"
transport = "streamable"
url       = "https://api.githubcopilot.com/mcp/"

[mcp.servers.headers]
Authorization = "Bearer $GITHUB_TOKEN"
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

Or for servers like GitHub's MCP in OAuth mode which require an explicit client ID & secret:

```toml
[[mcp.servers]]
name                = "github"
transport           = "streamable"
url                 = "https://api.githubcopilot.com/mcp/"
oauth               = true
oauth_client_id     = "$GITHUB_MCP_CLIENT_ID"
oauth_client_secret = "$GITHUB_MCP_CLIENT_SECRET"
# oauth_callback_port defaults to 34217 — must match your OAuth App's callback URL
```

Make sure to use `http://localhost:34217/callback` as callback URL. Adjust the port as configured.

`oauth` is only valid with `transport = "streamable"`.

#### How authentication works

OAuth-enabled servers follow the same lazy connection model as other servers
— they are not connected at startup — but the connection itself triggers a
browser-based flow instead of a silent TCP connection.

The first time you submit a prompt that causes the AI to call `connect_server`
for an OAuth server, Werkler pauses and shows you a browser link directly in
the chat:

```
To connect to atlassian, open this URL in your browser:
https://auth.atlassian.com/authorize?...

Waiting for authorization…
```

1. Open the URL in your browser and complete the login flow.
2. Your browser is redirected to `http://localhost:34217/callback` (the default
   callback port; see `oauth_callback_port` below).
3. Werkler exchanges the authorization code for tokens, then connects the
   server and proceeds with your original prompt automatically.

Tokens (including refresh tokens) are stored encrypted-at-rest in
`~/.config/werkler/oauth/<server-name>.json` (mode `0600`). Subsequent
werkler sessions reuse the stored token and refresh it automatically —
you will not be prompted again unless the refresh token expires or is
revoked.

#### OAuth configuration fields

| Field | Default | Description |
|---|---|---|
| `oauth` | `false` | Enable OAuth for this streamable server |
| `oauth_client_id` | — | Pre-registered client ID (required when the auth server has no Dynamic Client Registration) |
| `oauth_client_secret` | — | Client secret (optional for public clients) |
| `oauth_callback_port` | `34217` | Local TCP port werkler listens on for the browser redirect. Must match the callback URL registered in your OAuth App. Set to `0` to let the OS pick a random port. |

> **Note:** Because the OAuth flow requires a browser, it cannot run in
> non-interactive (`--prompt`) mode. If you try, Werkler will print an error
> and exit. Run `werkler chat` once first to authenticate, then use
> `--prompt` freely afterwards.
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

## Tool approval

Every time the AI wants to call a tool, Werkler asks for your approval —
unless the tool matches an entry in `auto_approve_tools`. Approval choices
in the interactive TUI:

| Key | Effect |
|-----|--------|
| `y` | Allow this call once |
| `a` | Allow this tool for the rest of the session |
| `p` | Allow permanently — appends the tool name to `auto_approve_tools` in your config file |
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

### Path access

File operations (read, write, edit, delete) additionally require per-path
approval the first time the AI accesses a path. These prompts appear in the
TUI as a separate dialog before the tool is executed.

| Key | Effect |
|-----|--------|
| `y` | Allow this path access once |
| `d` | Allow the whole containing directory for this session (covers all files beneath it) |
| `a` | Allow all remaining paths in this batch |
| `p` | Allow permanently — appends the path to `auto_approve_paths` in your config file |
| `n` | Deny and cancel the current tool call |

By default, the AI may read any file under the **current working directory**
without prompting — the assumption is you started werkler in a project you
want the AI to work with. You can disable this:

```toml
[mcp]
auto_approve_cwd_read = false   # default: true
```

Additional paths (or glob patterns) can be permanently pre-approved:

```toml
[mcp]
auto_approve_paths = [
  "/home/user/shared-docs",
]
```

## Full example config

```toml
[ai]
active = "copilot"

[[ai.providers]]
name     = "openai"
type     = "openai"
endpoint = "https://api.openai.com/v1"
api_key  = "sk-..."
model    = "gpt-4o"

[[ai.providers]]
name  = "copilot"
type  = "copilot"
model = "claude-sonnet-4-5"

# Optional: rubber duck reviewer (separate model for critical feedback)
# [ai.rubber_duck]
# provider = "copilot"         # reuse an existing provider
# model    = "claude-opus-4-5" # with a more capable model for review

# Optional: custom mode presets
# [[modes]]
# name                 = "plan-autopilot"
# base                 = "plan"
# autopilot            = true
# autopilot_max_cycles = 30

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

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
```

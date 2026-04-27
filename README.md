# Werkler

**AI-assisted software development toolkit**

Werkler is an AI-powered CLI tool that helps software developers with routine tasks:
writing and reviewing code, designing software, drafting tickets, and technical documentation.

## 🚀 Features

- **Multi-provider AI support**: Use OpenAI, GitHub Copilot, and other OpenAI-compatible APIs simultaneously
- **Model Context Protocol (MCP) integration**: Connect to external tools and APIs with proper OAuth + DCR support
- **Interactive TUI**: Full-featured terminal UI with model switching, session management, and more
- **Session persistence**: Save and resume chat sessions
- **Tool approval system**: Control which AI actions require confirmation
- **Skills system**: Load reusable instruction sets
- **Custom agents**: Define named AI personas with optional tool restrictions
- **Rubber duck reviewer**: Optional secondary AI model for critical feedback
- **Autopilot mode**: Autonomous task execution with cycle limits

## 📦 Installation

### From source

```sh
git clone https://github.com/icedream/werkler.git
cd werkler
go build -o werkler ./cmd/werkler
chmod +x werkler
sudo mv werkler /usr/local/bin/
```

## 🛠️ Configuration

Werkler uses a TOML configuration file located at:

- **Linux / BSD**: `$XDG_CONFIG_HOME/werkler/config.toml` (usually `~/.config/werkler/config.toml`)
- **macOS**: `~/Library/Application Support/werkler/config.toml`
- **Windows**: `%AppData%\werkler\config.toml`

```sh
# Create a basic config
mkdir -p ~/.config/werkler
werkler chat  # Generates default config if missing
```

See [Configuration Documentation](docs/configuration.md) for detailed configuration options including:
- Multiple AI providers (OpenAI, GitHub Copilot)
- MCP server setup (stdio, streamable HTTP, SSE)
- OAuth authentication
- Rubber duck reviewer setup
- Tool auto-approval patterns

## 🎯 Usage

### Quick start

```sh
# Start interactive chat
werkler chat

# Run a prompt and exit
werkler chat --prompt "Help me design an API"
```

### Common tasks

```sh
# List saved sessions
werkler sessions list

# Delete a session
werkler sessions delete <id>

# Delete a session without a confirmation prompt
werkler sessions delete <id> --force

# Authenticate with GitHub Copilot
werkler auth copilot

# Check authentication status
werkler auth status
```

### Options

```sh
# Use a specific config file
werkler --config /path/to/config.toml chat

# Override the active AI model
werkler --model gpt-4o chat

# Switch AI provider
werkler chat --provider=openai

# Resume the session picker on startup
werkler chat --resume

# Resume a specific saved session by ID (or unique prefix)
werkler chat --session abc123

# Activate a named mode preset on startup
werkler chat --mode plan

# Activate a named agent on startup
werkler chat --agent code-review-agent

# Enable autopilot for autonomous work
werkler chat --autopilot --prompt "Write tests for pkg/api"

# Limit autopilot cycles
werkler chat --autopilot --autopilot-max-cycles 20 --prompt "..."

# Enable verbose output in prompt mode
werkler chat --prompt "..." --verbose
```

## 🛠️ AI Providers

Werkler supports multiple AI providers. See [Configuration](docs/configuration.md#ai-provider) for details:

- **OpenAI-compatible**: Any OpenAI-compatible API (OpenAI, Ollama, vLLM, KubeAI, etc.)
- **GitHub Copilot**: Native provider with device flow authentication
- **Multiple providers**: Switch between providers live in the TUI

## 💬 Skills

Skills are reusable instruction sets that can be loaded on demand. See [Skills Documentation](docs/skills.md):

```sh
# Create a skill
mkdir -p ~/.agents/skills/my-skill
cat > ~/.agents/skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: Instructions for the AI
---

Your instructions here...
EOF
```

Then ask the AI: "Use the my-skill skill and review my code."

## 🔄 Tool Approval

Werkler asks for approval before calling tools (unless auto-approved):

- `y`: Allow this call once
- `a`: Allow this tool for the session
- `p`: Allow permanently (adds to config)
- `n`: Deny

Configure auto-approved tools in your config:

```toml
[mcp]
auto_approve_tools = [
  "fs__read_file",
  "git__log",
]
```

## 📚 Documentation

- [Configuration](docs/configuration.md) - Complete configuration reference
- [TUI Reference](docs/tui.md) - Keyboard shortcuts, slash commands, and interactive features
- [Skills](docs/skills.md) - Reusable instruction sets
- [Agents](docs/agents.md) - Custom AI personas with tool restrictions
- [Autopilot Mode](docs/autopilot.md) - Autonomous task execution
- [Mode Presets](docs/modes.md) - Built-in and custom mode configuration
- [Project Memory](docs/memory.md) - Persistent per-project AI memory

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request.

## 📜 License

See [LICENSE](LICENSE) file.

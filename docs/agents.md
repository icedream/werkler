# Custom Agents

Custom agents are named AI personas you define. When an agent is active,
its instructions are injected into the system prompt so the AI adopts a
specific role, follows project-specific rules, or restricts itself to a
particular set of tools.

Agents are stored as TOML files and can be created interactively with the
built-in wizard or by writing a file by hand.

## How agents work

1. Werkler loads all agent files from the agents directory at startup.
2. Each agent's **name**, **description**, and **when** hint are listed in the
   system prompt so the AI knows what is available.
3. When the AI decides an agent is appropriate (or you activate one
   explicitly), it calls the built-in `use_agent` tool. The agent's
   `instructions` are injected into the conversation and take effect
   immediately for the rest of the session.
4. Only one agent can be active at a time. Activating another agent replaces
   the previous one.

You can also activate or deactivate an agent directly:

```
> Use the code-review-agent agent and review my PR.
```

## Managing agents in the TUI

### `/agent` — create, activate, or deactivate

| Command             | Effect                                              |
| ------------------- | --------------------------------------------------- |
| `/agent new`        | Open the creation wizard                            |
| `/agent <name>`     | Activate the named agent immediately                |
| `/agent off`        | Deactivate the current agent and return to default  |

You can also activate an agent on startup with the `--agent` flag:

```sh
werkler chat --agent code-review-agent
```

## Agent file format

Each agent is a single TOML file in the agents directory:

```
~/.config/werkler/agents/
├── code-review-agent.toml
└── ticket-writer.toml
```

### Fields

| Field          | Required | Description                                                                                     |
| -------------- | -------- | ----------------------------------------------------------------------------------------------- |
| `name`         | ✓        | Machine-readable identifier. Letters, digits, `-`, `_` only. Used with `use_agent`.            |
| `description`  | ✓        | One-line summary shown to the AI when it decides which agent to activate.                       |
| `when`         | ✓        | Describes the situations in which the AI should activate this agent (shown in the system prompt). |
| `instructions` | ✓        | System-prompt text injected when the agent is active.                                           |
| `tools`        | —        | Optional list of allowed tool names. Omit for unrestricted access; set to `[]` to disallow all tools (except infrastructure tools). |

### Example

```toml
name        = "code-review-agent"
description = "Expert code reviewer specialising in idiomatic style, performance, and correctness."
when        = "Activate when the user is reviewing a pull request or performing a final code audit."

instructions = """
## Mode: Code Review

You are performing a code review. Focus on:
- Correctness and logic errors
- Security vulnerabilities
- Performance issues
- Missing error handling

Be concise. Point out only meaningful issues. Do not rewrite correct code.
"""
```

### Restricting tools

The `tools` field is an optional allowlist. When omitted the agent has access
to every available tool. When set, only the listed tools are available
(infrastructure tools such as `use_agent`, `ask_user`, `task_start`, and
`task_complete` are always available regardless).

```toml
# Allow only read-only filesystem and git tools
tools = [
  "fs__read_file",
  "fs__list_directory",
  "git__log",
  "git__diff",
]
```

Use `<server>.*` to grant all tools from an MCP server:

```toml
tools = ["github.*", "fs.*"]
```

## Agents directory

### Default location

```
~/.config/werkler/agents/
```

Override with the `WERKLER_AGENTS_DIR` environment variable:

```sh
WERKLER_AGENTS_DIR=/path/to/my/agents werkler chat
```

## Creating an agent with the wizard

Type `/agent new` (or `/agent` with no argument) in the TUI to open the
creation wizard. The wizard has two modes:

- **AI-assisted** (default) — describe the agent in plain English and the AI
  generates the name, description, when-hint, instructions, and tool list for
  you. You review the generated TOML before saving.
- **Manual** (`m` to switch) — fill in each field yourself using a form.

After filling in the agent details the wizard shows a tool picker where you
can deselect individual tools to restrict the agent's access. Confirm the
final TOML and the wizard saves the file to the agents directory automatically.

## Adding an agent to the README

Once created, you can reference your agents in documentation by listing them
and their `when` hints so teammates know when each is appropriate.

## Troubleshooting

| Symptom                          | Cause                                  | Fix                                                                 |
| -------------------------------- | -------------------------------------- | ------------------------------------------------------------------- |
| Agent not listed by the AI       | File not in agents directory           | Check `WERKLER_AGENTS_DIR` or `~/.config/werkler/agents/`          |
| Agent not loaded on startup      | TOML parse error                       | Check stderr output; validate TOML syntax                           |
| `--agent` flag gives "not found" | Name mismatch                          | The `name` field in the file must match the flag value exactly      |
| AI ignores the `tools` allowlist | Infrastructure tools are always on     | `use_agent`, `ask_user`, etc. cannot be removed from any agent      |

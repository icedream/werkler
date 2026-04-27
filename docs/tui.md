# TUI Reference

This document covers all keyboard shortcuts, slash commands, and interactive
features of the Werkler TUI (`werkler chat` without `--prompt`).

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` / `Ctrl+D` | Quit |
| `Ctrl+P` | Open model picker |
| `Ctrl+R` | Open session picker (resume a saved session) |
| `Alt+M` | Toggle mouse reporting — mouse on: scroll wheel works; mouse off: terminal text selection works |
| `↑` / `↓` | Scroll conversation **or** navigate prompt history when the input is empty |
| `PgUp` / `PgDn` | Scroll conversation |
| `Esc` (once, while AI is running) | Arm cancellation — a second `Esc` cancels the current operation |
| `Esc` (while input has text) | Clear the input |
| `Esc` (with queued prompts, input empty) | Remove the last queued prompt |

## Slash commands

Type `/` to open autocomplete. Press `↑`/`↓` to navigate, `Enter` to select,
`Esc` to close.

| Command | Description |
|---------|-------------|
| `/model` | Switch the active AI model |
| `/tools` | Enable or disable individual tools for this session |
| `/skills` | Enable or disable individual skills for this session (only shown when skills are loaded) |
| `/clear` | Clear the conversation history |
| `/new` | Start a new session (clears history and detaches from current saved session) |
| `/compact` | Summarize the conversation to free up context window space |
| `/registry` | Browse and add MCP servers from the MCP registry |
| `/todos` | Toggle the todo sidebar |
| `/autopilot` | Toggle autopilot mode |
| `/help` | Show keyboard shortcuts and slash commands inline |
| `/quit` | Quit werkler |
| `/allow-all` | Toggle allow-all mode — approves all tool calls and path access without prompting. A `⚠ allow-all ON` indicator appears in the status bar when active. |
| `/reasoning` | Toggle display of model reasoning/thinking content (on by default). |

## Prompt queuing

You can type and submit a prompt at any time, even while the AI is busy:

- **Enter** — queue the prompt; it will be sent after the current AI turn fully completes.
- **Ctrl+J** (or Ctrl+Enter on supported terminals) — queue the prompt as **urgent**; it will be injected before the AI's next tool call, interrupting the current task cycle.
- **`/urgent` prefix** — prefix your message with `/urgent ` and submit with Enter to mark it urgent (same as Ctrl+J, useful on terminals where Ctrl+J is not distinct from Enter).

The status bar shows `+N queued` (or `+N queued (M urgent)`) while prompts are pending. Press `Esc` with an empty input to remove the last queued prompt.

## Prompt history

Press `↑` / `↓` while the input box is empty to navigate your prompt history for
the current session. This does not affect conversation scroll.

## Todo sidebar

When the AI uses the `todo_add` / `todo_update` tools to manage tasks, a sidebar
appears automatically. Use `/todos` to toggle it manually.

Todos have four statuses: `pending`, `in_progress`, `done`, `blocked`.

## Token usage

After each AI response the status bar briefly shows the token counts for that
turn (`↑N/↓N`). This helps you monitor context window consumption.

## Session management

Werkler auto-saves sessions after every AI turn. Use `Ctrl+R` to open the
session picker and resume any saved session. Sessions are stored in
`~/.local/share/werkler/sessions/` (Linux/macOS) or `%AppData%\werkler\sessions\` (Windows).

To manage sessions from the command line:

```sh
werkler sessions list              # list saved sessions
werkler sessions delete <id>       # delete a session by (prefix of) ID
```

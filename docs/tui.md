# TUI Reference

This document covers all keyboard shortcuts, slash commands, and interactive
features of the Werkler TUI (`werkler chat` without `--prompt`).

## Keyboard shortcuts

| Key                                      | Action                                                                                          |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `Shift+Tab`                              | Cycle to the next mode preset (default → plan → document → custom modes)                        |
| `Ctrl+C` / `Ctrl+D`                      | Quit                                                                                            |
| `Ctrl+P`                                 | Open model picker                                                                               |
| `Ctrl+R`                                 | Open session picker (resume a saved session)                                                    |
| `Alt+M`                                  | Toggle mouse reporting — mouse on: scroll wheel works; mouse off: terminal text selection works |
| `↑` / `↓`                                | Scroll conversation **or** navigate prompt history when the input is empty                      |
| `PgUp` / `PgDn`                          | Scroll conversation                                                                             |
| `Esc` (once, while AI is running)        | Arm cancellation — a second `Esc` cancels the current operation                                 |
| `Esc` (while input has text)             | Clear the input                                                                                 |
| `Esc` (with queued prompts, input empty) | Remove the last queued prompt                                                                   |

## Slash commands

Type `/` to open autocomplete. Press `↑`/`↓` to navigate, `Enter` to select,
`Esc` to close.

| Command                | Description                                                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/mode`                | Switch the active mode preset (default, plan, document, or custom)                                                                                     |
| `/model`               | Switch the active AI model                                                                                                                             |
| `/tools`               | Enable or disable individual tools for this session                                                                                                    |
| `/skills`              | Enable or disable individual skills for this session (only shown when skills are loaded)                                                               |
| `/clear`               | Clear the conversation history                                                                                                                         |
| `/new`                 | Start a new session (clears history and detaches from current saved session)                                                                           |
| `/compact`             | Summarize the conversation to free up context window space                                                                                             |
| `/registry`            | Browse and add MCP servers from the MCP registry                                                                                                       |
| `/todos`               | Toggle the todo sidebar                                                                                                                                |
| `/autopilot`           | Toggle autopilot mode                                                                                                                                  |
| `/help`                | Show keyboard shortcuts and slash commands inline                                                                                                      |
| `/quit`                | Quit werkler                                                                                                                                           |
| `/image <path-or-url>` | Load a local image file and attach it to your next message so the AI can see it                                                                        |
| `/allow-all`           | Toggle allow-all mode — approves all tool calls and path access without prompting. A `⚠ allow-all ON` indicator appears in the status bar when active. |
| `/reasoning`           | Toggle display of model reasoning/thinking content (on by default).                                                                                    |
| `/agent [new\|<name>\|off]` | Create a new agent with the wizard (`/agent new`), activate a named agent (`/agent <name>`), or deactivate the current agent (`/agent off`). See [Agents](agents.md). |
| `/expand [<handle>\|all]`   | Expand collapsed process output — show full output for a process handle, or `all` to expand every collapsed handle.                                    |
| `/collapse [<handle>\|all]` | Collapse expanded process output to save screen space.                                                                                                 |
| `/sidebar [wider\|narrower\|reset]` | Resize the todo sidebar.                                                                                                                    |

## Prompt queuing

You can type and submit a prompt at any time, even while the AI is busy:

- **Enter** — queue the prompt; it will be sent after the current AI turn fully completes.
- **Ctrl+J** (or Ctrl+Enter on supported terminals) — queue the prompt as **urgent**; it will be injected before the AI's next tool call, interrupting the current task cycle.
- **`/urgent` prefix** — prefix your message with `/urgent ` and submit with Enter to mark it urgent (same as Ctrl+J, useful on terminals where Ctrl+J is not distinct from Enter).

The status bar shows `+N queued` (or `+N queued (M urgent)`) while prompts are pending. Press `Esc` with an empty input to remove the last queued prompt.

## Prompt history

Press `↑` / `↓` while the input box is empty to navigate your prompt history for
the current session. This does not affect conversation scroll.

## Mode presets

The active mode shapes the AI's behaviour for the current session (planning,
documentation, etc.). The mode name and a colored border accent appear in the
header when a non-default mode is active.

Press **Shift+Tab** to cycle through modes, or use `/mode` to pick one directly.

See [modes.md](modes.md) for a full reference including custom mode configuration.

## Attaching images

If your AI model supports vision, you can show it image files:

```
/image /path/to/screenshot.png
/image ~/designs/mockup.jpg
```

After running `/image`, the image is staged as a pending attachment. Send your
next message as normal and the image will be included. The AI can also load
images itself using the built-in `read_image` tool when you ask it to look at a
file.

Supported formats: PNG, JPEG, GIF, WebP.

## Todo sidebar

When the AI uses the `todo_add` / `todo_update` tools to manage tasks, a sidebar
appears automatically. Use `/todos` to toggle it manually.

Todos have four statuses: `pending`, `in_progress`, `done`, `blocked`.

## Task title

While the AI is working it may call the `task_start` tool to report what it is
currently doing (e.g. _"Implementing OAuth callback"_ or _"Writing tests for
parser"_). The title is displayed in the status bar next to the activity
indicator and updates as the AI moves between phases. It is cleared
automatically when the turn ends.

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

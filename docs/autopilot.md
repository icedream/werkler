# Autopilot Mode

Autopilot mode lets werkler work autonomously on a goal without requiring you
to type "Continue" after every response. The AI keeps working — calling tools,
tracking progress, making decisions — until it signals that the task is fully
done or a configurable cycle limit is reached.

## How it works

When autopilot is active, werkler does the following after every AI turn that
ends without tool calls:

1. Checks whether `task_complete` was called. If so, the loop ends and a
   summary is shown.
2. If the cycle cap has been reached, the loop pauses and notifies you.
3. Otherwise, a hidden "Continue working." message is injected and the AI
   is called again immediately.

The continuation message is ephemeral — it is never saved to the session file
or shown in the conversation history. From the session's perspective, it looks
like one long uninterrupted work session.

## Enabling autopilot

**CLI flag (recommended for automation):**

```sh
werkler chat --autopilot --prompt "Refactor all usages of the legacy API in src/"
```

**TUI slash command:**

Type `/autopilot` in the input box and press Enter to toggle it on or off.
A banner confirms the change, and the status bar shows the cycle counter while
active.

**Status bar indicators:**

| Indicator | Meaning |
|-----------|---------|
| `⚡ 3/50` | Autopilot active, 3 cycles completed out of a cap of 50 |
| `⚡ paused (50 cycles)` | Cap reached, waiting for you to resume or disable |

## Stopping autopilot

**Signalling done (preferred):** The AI calls `task_complete(summary)`. This
ends the loop cleanly, appends a summary item to the conversation, and returns
to idle.

**Manual disable:** Type `/autopilot` in the TUI to toggle it off at any point.

**Cycle cap:** When the cap is hit, the loop pauses automatically. Press Enter
(with an empty input) to resume for another full cap's worth of cycles, or
type `/autopilot` to disable.

**Cancel:** Press Escape during a turn to cancel the current AI call, then
`/autopilot` to disable.

## The `task_complete` tool

The AI should call `task_complete(summary)` when it has finished all the work
it can do. Outside of autopilot mode the tool still works — it returns to idle
and shows the summary.

The `summary` argument should be a concise description of what was accomplished,
for example:

```
Refactored 14 files: replaced all calls to legacyClient.Get() with the new
httpClient.Fetch() pattern. Updated tests. No regressions found.
```

The AI is instructed to call `ask_user` instead of `task_complete` if it is
completely blocked and needs human input.

## Cycle cap

The cap prevents runaway loops. When hit, autopilot pauses rather than errors,
so you can inspect the current state and decide whether to continue.

**Default:** 50 cycles.

**Override in config:**

```toml
[autopilot]
max_cycles = 100
```

**Override per invocation:**

```sh
werkler chat --autopilot --autopilot-max-cycles 20 --prompt "..."
```

A "cycle" is one complete end-of-turn response from the AI (no tool calls).
Tool call chains within a single turn do not count toward the cycle cap.

## Working with the todo list

The AI's built-in todo tools (`todo_add`, `todo_update`, `todo_list`) are
natural companions to autopilot mode. The system prompt instructs the AI to
use them when given a long-running goal. You can watch progress in the todo
sidebar (`/todos` to open it).

A typical autopilot session for a multi-file refactor might look like:

1. AI calls `todo_add` for each part of the task
2. AI marks each todo `in_progress` before starting it, `done` when finished
3. AI calls `task_complete` once all todos are done

## Non-interactive mode (`--prompt`)

Autopilot works in `--prompt` mode too:

```sh
werkler chat --autopilot --prompt "Write unit tests for all untested functions in pkg/api"
```

If the cycle cap is hit in `--prompt` mode, the command exits with an error
describing the situation. The AI's last response is still printed.

OAuth-protected MCP tools cannot be used in `--prompt` mode — authenticate
interactively first with `werkler chat`, then use `--prompt`.

## Recommended use cases

Autopilot works best when the goal is well-defined, the AI has the tools it
needs, and the result is easy to verify:

- Refactoring a codebase to a new API or pattern
- Writing tests for existing functions
- Fixing a batch of lint warnings or type errors
- Generating documentation from source code
- Triage and labelling of issues from a tracker
- Creating structured tickets from a set of requirements

## What to watch out for

**Review changes before committing.** Autopilot can modify many files. Use
`git diff` or your usual review flow before committing AI-generated changes.

**Be careful with write-capable tools auto-approved.** If `auto_approve_tools`
includes write tools (file edits, issue creation, code push), the AI will use
them without prompting. Consider using a narrower approval list for long-running
tasks, or keeping write tools under manual approval.

**Context window limits.** Very long autopilot runs accumulate a large
conversation history. Werkler will compact the context automatically when it
gets close to the limit, but very complex tasks may benefit from being split
into shorter sessions.

**The AI may go in circles.** If a task is ambiguous or the AI lacks the tools
to make progress, it may loop without making real headway. The cycle cap is
your safety net. Check the todo sidebar and the conversation to understand
where things stand.

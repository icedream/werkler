# Mode Presets

Modes let you change how the AI behaves for different types of work — planning,
writing documentation, or general development — without editing your system
prompt by hand. Each mode bundles a system-prompt extension, an optional
border color for visual orientation, and optional autopilot and tool-approval
overrides.

## Built-in modes

| Mode | Color | Behaviour |
|------|-------|-----------|
| `default` | (none) | General-purpose assistant |
| `plan` | Blue | Prioritises task decomposition, structured tickets, dependency analysis, and risk review. Will not write implementation code unless asked. |
| `document` | Green | Prioritises clear, accurate prose, consistent terminology, and self-contained documents. |

## Switching modes

**Shift+Tab** — cycles through all available modes in order. The active mode
name is shown in the TUI header, and the border changes color when a non-default
mode is active.

**/mode** — opens a picker to jump directly to any mode:

```
[✓] default    General-purpose assistant
[ ] plan       Structured planning and ticket drafting
[ ] document   Documentation writing
```

Modes take effect immediately for the next AI turn.

## User-defined modes

Define custom modes in your config file under `[[modes]]`:

```toml
[[modes]]
name               = "review"
color              = "202"       # orange (terminal 256-color index)
system_prompt_extra = """
## Mode: Code Review

You are performing a code review. Focus on:
- Correctness and logic errors
- Security vulnerabilities
- Performance issues
- Missing error handling

Be concise. Point out only meaningful issues.
"""
```

### Extending a built-in

Use `base` to inherit from an existing mode and extend it:

```toml
[[modes]]
name               = "plan-autopilot"
base               = "plan"
autopilot          = true
autopilot_max_cycles = 30
```

This example creates a mode that combines the `plan` system-prompt extension
with autopilot enabled and a 30-cycle cap.

### All `[[modes]]` fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Mode name. Shadows a built-in when matching `default`, `plan`, or `document`. |
| `base` | string | Inherit all settings from this mode before applying overrides. |
| `color` | string | Terminal 256-color index for the TUI border accent (e.g. `"33"`, `"202"`). |
| `system_prompt_extra` | string | Text appended to the system prompt while the mode is active. |
| `autopilot` | bool | Override the autopilot on/off setting for this mode. |
| `autopilot_max_cycles` | int | Override the autopilot cycle cap (0 = inherit). |
| `auto_approve_tools` | list | Additional tool globs to auto-approve while this mode is active. |

### Overriding a built-in

When a user-defined mode has the same name as a built-in, it fully overrides
it. This lets you customise the `plan` or `document` prompts:

```toml
[[modes]]
name               = "plan"
base               = "plan"            # inherit the built-in's prompt
system_prompt_extra = """
(All the original planning instructions…)

Additionally: always format tickets as GitHub issues with title, body, and
label suggestions.
"""
color              = "69"              # keep the original blue
```

## Combining with autopilot

Modes and autopilot are independent; you can mix them freely. A mode may
pre-enable autopilot (`autopilot = true`) or you can toggle autopilot
separately with `/autopilot`. Whichever setting wins is shown in the status
bar.

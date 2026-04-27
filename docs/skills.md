# Skills

Skills are reusable instruction sets that the AI loads on demand. Each skill's
full content is injected into the conversation the moment it is invoked, so the
AI follows the instructions it contains for the rest of the session.

Skills can contain anything: code style guides, project conventions,
step-by-step process descriptions, or domain knowledge the AI should apply.

## How skills work

1. Werkler scans the skills directory at startup and registers every valid skill
   it finds.
2. Each enabled skill's **name and description** are listed in the system prompt
   so the AI knows what is available — but the full content is **not** injected
   upfront.
3. When the AI decides a skill is relevant (or you ask it to use one), it calls
   the `use_skill` tool. The skill's full content is returned as the tool
   result and from that point the AI follows its instructions.
4. All loaded skills start **enabled** by default. You can toggle individual
   skills on or off for the current session using `/skills` (see below).

You can trigger a skill explicitly by just asking:

```
> Use the go-guidelines skill and review my code.
```

## Managing skills in the TUI

### `/skills` — toggle skills for this session

Type `/skills` (or press `/` and select it from the autocomplete list) to open
the skill picker. Every loaded skill is shown with a checkbox:

```
[✓] go-guidelines   Go best practices for performance, modern syntax, and testing.
[ ] my-conventions  Project-specific naming and style rules.
```

- **`space`** — toggle the highlighted skill on or off
- **`enter`** / **`esc`** — close the picker

Disabling a skill removes it from the AI's awareness for the rest of the
session — its name and description are no longer listed in the system prompt so
the AI will not call `use_skill` for it. Re-enabling restores it.

> **Note:** Toggling a skill does not retroactively add or remove skill content
> that was already injected into an earlier message in the conversation.

## Skill file format

Each skill lives in its own subdirectory. The subdirectory can have any name;
the skill is identified by the `name` field inside the file.

```
~/.agents/skills/
└── go-guidelines/
    └── SKILL.md
```

`SKILL.md` must start with a YAML front-matter block delimited by `---`:

```markdown
---
name: go-guidelines
description: Go best practices for performance, modern syntax, and testing.
---

# Go Guidelines

Always prefer early returns over deeply nested if-blocks.
Use `errors.As` for error type assertions, not type switches.

...more instructions...
```

### Front-matter fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | ✓ | Machine-readable identifier. Used with `use_skill`. |
| `description` | ✓ | Short summary shown to the AI when it decides which skill to load. |

### Shell command expansion

Lines matching `` !`command` `` in the skill body are treated as shell commands.
They are executed once when the skill is loaded (at werkler startup), and their
stdout replaces the line in the skill's content.

This is useful for injecting dynamic context such as the project's language
version:

```markdown
---
name: go-guidelines
description: Go best practices.
---

Detected Go version:
!`go version | awk '{print $3}'`

Use the features of this Go version and below.
```

Commands run with a 10-second timeout in the current working directory. If a
command fails, a short error placeholder is inserted instead.

## Skills directory

### Default location

```
~/.agents/skills/
```

This is a shared location — the same skills directory is used by other AI tools
that follow the same convention (e.g. GitHub Copilot CLI).

### Changing the location

Override the directory in your config file:

```toml
[skills]
dir = "~/.my-skills"
```

A leading `~` is expanded to your home directory. Relative paths are not
supported — use an absolute path or the `~` prefix.

### Per-project skills

There is no automatic per-project skill loading. To use project-specific skills:

1. Create a skills directory inside (or alongside) your project.
2. Point werkler at it when you start:

```sh
werkler chat --config ./werkler.toml
```

Where `werkler.toml` contains:

```toml
[skills]
dir = "./.werkler/skills"
```

Or set it globally for this project by placing a `config.toml` in the project
and using `--config`:

```toml
[skills]
dir = "~/.agents/skills"   # still include global skills
```

> **Note:** Werkler currently loads from a single skills directory. If you need
> both global and project skills active simultaneously, symlink them into one
> directory or concatenate them.

## Adding a skill

### Global skill (available in all projects)

```sh
mkdir -p ~/.agents/skills/my-skill
cat > ~/.agents/skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: A short description of what this skill teaches the AI.
---

Your instructions here.
EOF
```

### Project-local skill

Create a `SKILL.md` in a directory you point werkler at via `skills.dir`:

```sh
mkdir -p .werkler/skills/code-style
cat > .werkler/skills/code-style/SKILL.md << 'EOF'
---
name: code-style
description: Coding conventions for this project.
---

- Use 2-space indentation in all TypeScript files.
- All exported functions must have JSDoc comments.
- Prefer `const` over `let` everywhere.
EOF
```

## Example skill

Here is a more complete example demonstrating shell expansion:

```markdown
---
name: node-guidelines
description: Node.js and TypeScript best practices for this project.
---

# Node.js Guidelines

Project runtime:
!`node --version 2>/dev/null || echo "not installed"`

TypeScript version:
!`npx tsc --version 2>/dev/null || echo "not installed"`

## Rules

- Always use strict TypeScript (`"strict": true` in tsconfig.json).
- Prefer `async`/`await` over raw Promise chains.
- Use `zod` for runtime input validation.
- Do not use `any` — use `unknown` with type guards instead.
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Skill not listed in `use_skill` | Missing or invalid front matter | Check that `SKILL.md` starts with `---` and has both `name` and `description` |
| Skill not found at startup | Wrong directory | Check `skills.dir` in your config, or confirm `~/.agents/skills/<name>/SKILL.md` exists |
| Warning printed on startup | A skill failed to load | Check stderr output; usually a malformed YAML front matter |
| Shell expansion line not replaced | Command timed out or failed | The placeholder `[error: ...]` is injected; fix or remove the `!`...`` ` line |

Skills that fail to load are skipped with a warning printed to stderr. Werkler
continues with the skills that loaded successfully.

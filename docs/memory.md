# Project Memory

Werkler gives the AI a persistent, per-project note store so it can remember
things across sessions — architectural decisions, naming conventions, known
issues, important file locations, and anything else you want it to carry
forward automatically.

## How it works

When you start `werkler chat` from a project directory, a memory store for
that directory is opened automatically. Any named memory files found in the
store are injected into the system prompt at the start of every session. No
configuration is required.

Memory files are stored as Markdown under:

```
~/.config/werkler/memory/<sha256-of-abs-cwd>/
```

Each named memory is a file `<slug>.md` in that directory.
Slugs are lowercase alphanumeric strings with hyphens — e.g. `general`,
`api-notes`, `architecture`.

## Ancestor memories

When werkler starts, it also loads read-only memories from every parent
directory. For example, if you open werkler in `/home/user/work/myproject`,
it loads memories from (in order):

1. `/home/user/work/myproject` — read/write (current project)
2. `/home/user/work` — read-only
3. `/home/user` — read-only
4. ... up to the root

This means you can promote a memory to a parent directory to share it across
multiple sub-projects (e.g. across services in a monorepo). Ancestor memories
appear in the system prompt but cannot be written to directly from the current
project — they must be edited from the appropriate directory, or promoted there
using `memory_promote`.

## Memory tools

The AI has the following built-in tools for managing memories. These are always
available (no MCP server required).

| Tool | Description |
|------|-------------|
| `memory_list` | List all named memory files for the current project |
| `memory_read` | Read a specific named memory file |
| `memory_write` | Write (replace) a named memory file |
| `memory_delete` | Delete a named memory file |
| `memory_promote` | Move a memory file to a parent directory's store |

### Limits

| Limit | Value |
|-------|-------|
| Maximum file size | 8 KB per named memory |
| Maximum files per project | 50 |

If a memory file is too large to fit in the system prompt budget, its content
is omitted and a `(omitted, budget exhausted)` notice is shown instead. The AI
can still access it by calling `memory_read`.

## Using memory effectively

Ask the AI to remember something once and it will persist automatically:

```
> Remember that we use snake_case for database column names in this project.
```

The AI will call `memory_write` to store this in e.g. `conventions.md`. On the
next session, that note will be injected into the system prompt before the
first message.

To check what the AI currently remembers:

```
> What do you have in your memory for this project?
```

The AI will call `memory_list` and summarise what it finds.

## Promoting a memory across projects

If a note turns out to be relevant to the whole workspace (e.g. a monorepo
root), ask the AI to promote it:

```
> The "conventions" memory applies to the whole repo, not just this service.
> Please promote it.
```

The AI will call `memory_promote("conventions", "..")`, which moves the file
to the parent directory's memory store. From that point it is visible to all
sub-projects.

## Direct inspection

You can inspect and edit memory files directly — they are plain Markdown:

```sh
# List memories for the current directory
ls ~/.config/werkler/memory/$(echo -n "$PWD" | sha256sum | cut -d' ' -f1)/

# View a memory
cat ~/.config/werkler/memory/.../general.md

# Delete a memory
rm ~/.config/werkler/memory/.../general.md
```

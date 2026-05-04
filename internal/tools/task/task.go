package task

import (
	"context"
	"fmt"
	"strings"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/todostore"
	"github.com/icedream/werkler/internal/tools/toolutil"
)

// Context is the narrow interface the task handler needs from the Manager.
type Context interface {
	TodoStore() *todostore.Store
	NotifyTaskTitle(title string)
}

// Handler holds the task and todo tool handlers.
type Handler struct{ ctx Context }

// NewHandler creates a Handler. ctx.TodoStore() may return nil, in which case
// todo tools are not registered.
func NewHandler(ctx Context) *Handler { return &Handler{ctx: ctx} }

// Tools returns Builtin definitions for task and todo tools.
// Todo tools are only included when a TodoStore is available.
func (h *Handler) Tools() []toolutil.Builtin {
	out := []toolutil.Builtin{
		{Def: taskStartDef, Handle: h.handleTaskStart},
		{Def: taskCompleteDef, Handle: h.handleTaskComplete},
	}
	if h.ctx.TodoStore() != nil {
		out = append(out,
			toolutil.Builtin{Def: todoAddDef, Handle: h.handleTodoAdd},
			toolutil.Builtin{Def: todoAddManyDef, Handle: h.handleTodoAddMany},
			toolutil.Builtin{Def: todoUpdateDef, Handle: h.handleTodoUpdate},
			toolutil.Builtin{Def: todoListDef, Handle: h.handleTodoList},
		)
	}
	return out
}

var taskStartDef = ai.ToolDefinition{
	Name: "task_start",
	Description: `Set the title of the task you are currently working on.
Call this whenever you begin a new sub-task or phase of work so the user can see
what you are doing in the status bar. You can call it multiple times to update
the title as work progresses. The title should be a short, human-readable phrase
such as "Implementing OAuth callback" or "Writing tests for parser".
Do NOT call this for every small step — only when starting a meaningful new phase.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"title": map[string]any{
				"type":        "string",
				"description": "Short description of the current task or phase",
			},
		},
		"required": []string{"title"},
	},
}

var taskCompleteDef = ai.ToolDefinition{
	Name: "task_complete",
	Description: `Signal that the assigned task is fully complete. Call this when all work is done and no further action is needed.
In autopilot mode this stops the autonomous loop. Outside autopilot mode it marks the task done and returns to idle.
Provide a concise summary of what was accomplished.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"summary": map[string]any{
				"type":        "string",
				"description": "Concise summary of what was accomplished",
			},
		},
		"required": []string{"summary"},
	},
}

var todoAddDef = ai.ToolDefinition{
	Name: "todo_add",
	Description: `Add a single todo item to the session task list.
Always supply a short kebab-case id (e.g. "write-readme", "fix-login-bug") so you can
reference the todo later. Use proactively at the start of multi-step tasks.
To add several todos at once, use todo_add_many instead.
IMPORTANT: Duplicate titles are not allowed. If a todo with the same title already exists
you will receive the existing item back — do NOT call todo_add again with the same title.
IMPORTANT: Duplicate IDs are not allowed. Choose a unique id; if the id already exists
you will receive the existing item back.
Only rephrase/re-id if this is genuinely a distinct task.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id":          map[string]any{"type": "string", "description": "Short kebab-case identifier (e.g. write-readme). Must be unique in this session."},
			"title":       map[string]any{"type": "string", "description": "Short one-line title"},
			"description": map[string]any{"type": "string", "description": "Optional detail or acceptance criteria"},
		},
		"required": []string{"title"},
	},
}

var todoAddManyDef = ai.ToolDefinition{
	Name: "todo_add_many",
	Description: `Add multiple todo items to the session task list in a single call.
Prefer this over repeated todo_add calls whenever you know the full list of tasks upfront.
Duplicate titles or IDs are skipped with a note in the result — do not retry them.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"items": map[string]any{
				"type":        "array",
				"description": "List of todo items to add",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":          map[string]any{"type": "string", "description": "Short kebab-case identifier. Must be unique in this session."},
						"title":       map[string]any{"type": "string", "description": "Short one-line title"},
						"description": map[string]any{"type": "string", "description": "Optional detail or acceptance criteria"},
					},
					"required": []string{"title"},
				},
			},
		},
		"required": []string{"items"},
	},
}

var todoUpdateDef = ai.ToolDefinition{
	Name:        "todo_update",
	Description: `Update the status (or title/description) of an existing todo item.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id":          map[string]any{"type": "string", "description": "Todo ID returned by todo_add"},
			"status":      map[string]any{"type": "string", "enum": []string{"pending", "in_progress", "done", "blocked"}, "description": "New status"},
			"title":       map[string]any{"type": "string", "description": "Replace title"},
			"description": map[string]any{"type": "string", "description": "Replace description"},
		},
		"required": []string{"id"},
	},
}

var todoListDef = ai.ToolDefinition{
	Name:        "todo_list",
	Description: `Return the current todo list as text so you can review progress.`,
	InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
}

func (h *Handler) handleTaskStart(_ context.Context, args map[string]any) (string, error) {
	h.ctx.NotifyTaskTitle(toolutil.StringArg(args, "title"))
	return "ok", nil
}

func (h *Handler) handleTaskComplete(_ context.Context, args map[string]any) (string, error) {
	summary := toolutil.StringArg(args, "summary")
	if summary == "" {
		summary = "Task complete."
	}
	return summary, nil
}

func (h *Handler) handleTodoAdd(_ context.Context, args map[string]any) (string, error) {
	store := h.ctx.TodoStore()
	title := toolutil.StringArg(args, "title")
	if title == "" {
		return "error: title is required", nil
	}
	requestedID := toolutil.StringArg(args, "id")
	for _, t := range store.List() {
		if t.Title == title {
			return fmt.Sprintf("duplicate: todo %q already exists with id=%s status=%s — use todo_update to change it", title, t.ID, t.Status), nil
		}
		if requestedID != "" && t.ID == requestedID {
			return fmt.Sprintf("duplicate: id %q already exists (title=%q status=%s) — choose a different id or use todo_update", requestedID, t.Title, t.Status), nil
		}
	}
	id := store.Add(requestedID, title, toolutil.StringArg(args, "description"))
	return fmt.Sprintf("added: id=%s status=pending title=%q", id, title), nil
}

func (h *Handler) handleTodoAddMany(_ context.Context, args map[string]any) (string, error) {
	store := h.ctx.TodoStore()
	rawItems, _ := args["items"].([]any)
	if len(rawItems) == 0 {
		return "error: items array is required and must not be empty", nil
	}
	var sb strings.Builder
	existing := store.List()
	for i, raw := range rawItems {
		item, ok := raw.(map[string]any)
		if !ok {
			fmt.Fprintf(&sb, "item %d: skipped (invalid format)\n", i+1)
			continue
		}
		title := toolutil.StringArg(item, "title")
		if title == "" {
			fmt.Fprintf(&sb, "item %d: skipped (title is required)\n", i+1)
			continue
		}
		requestedID := toolutil.StringArg(item, "id")
		skipped := false
		for _, t := range existing {
			if t.Title == title {
				fmt.Fprintf(&sb, "duplicate: %q already exists id=%s status=%s\n", title, t.ID, t.Status)
				skipped = true
				break
			}
			if requestedID != "" && t.ID == requestedID {
				fmt.Fprintf(&sb, "duplicate id %q (title=%q status=%s) — choose a different id\n", requestedID, t.Title, t.Status)
				skipped = true
				break
			}
		}
		if skipped {
			continue
		}
		id := store.Add(requestedID, title, toolutil.StringArg(item, "description"))
		existing = store.List()
		fmt.Fprintf(&sb, "added: id=%s title=%q\n", id, title)
	}
	return strings.TrimSpace(sb.String()), nil
}

func (h *Handler) handleTodoUpdate(_ context.Context, args map[string]any) (string, error) {
	store := h.ctx.TodoStore()
	id := toolutil.StringArg(args, "id")
	if id == "" {
		return "error: id is required", nil
	}
	var f todostore.UpdateFields
	if v := toolutil.StringArg(args, "status"); v != "" {
		f.Status = &v
	}
	if v := toolutil.StringArg(args, "title"); v != "" {
		f.Title = &v
	}
	if v := toolutil.StringArg(args, "description"); v != "" {
		f.Description = &v
	}
	if err := store.Update(id, f); err != nil {
		return "error: " + err.Error(), nil
	}
	for _, t := range store.List() {
		if t.ID == id {
			return fmt.Sprintf("updated: id=%s status=%s title=%q", t.ID, t.Status, t.Title), nil
		}
	}
	return "updated: " + id, nil
}

func (h *Handler) handleTodoList(_ context.Context, _ map[string]any) (string, error) {
	store := h.ctx.TodoStore()
	todos := store.List()
	if len(todos) == 0 {
		return "no todos", nil
	}
	var sb strings.Builder
	icons := map[string]string{
		todostore.StatusPending:    "○",
		todostore.StatusInProgress: "▶",
		todostore.StatusDone:       "✓",
		todostore.StatusBlocked:    "✗",
	}
	for _, t := range todos {
		icon := icons[t.Status]
		if icon == "" {
			icon = "?"
		}
		fmt.Fprintf(&sb, "%s [%s] %s: %s\n", icon, t.ID, t.Status, t.Title)
		if t.Description != "" {
			fmt.Fprintf(&sb, "    %s\n", t.Description)
		}
	}
	return strings.TrimRight(sb.String(), "\n"), nil
}

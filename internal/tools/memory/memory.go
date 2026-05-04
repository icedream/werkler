package memory

import (
	"context"
	"fmt"
	"strings"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/memorystore"
	"github.com/icedream/werkler/internal/tools/toolutil"
)

// Context is the narrow interface the memory handler needs from the Manager.
type Context interface {
	MemoryStore() *memorystore.MemoryStore
}

// Handler holds the memory tool handlers.
type Handler struct{ ctx Context }

// NewHandler creates a Handler.
func NewHandler(ctx Context) *Handler { return &Handler{ctx: ctx} }

// Tools returns Builtin definitions for all memory tools.
func (h *Handler) Tools() []toolutil.Builtin {
	return []toolutil.Builtin{
		{Def: listDef, Handle: h.handleMemoryList},
		{Def: readDef, Handle: h.handleMemoryRead},
		{Def: writeDef, Handle: h.handleMemoryWrite},
		{Def: deleteDef, Handle: h.handleMemoryDelete},
		{Def: promoteDef, Handle: h.handleMemoryPromote},
	}
}

var listDef = ai.ToolDefinition{
	Name:        "memory_list",
	Description: `List all named project memory files for the current directory, with their sizes.`,
	InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
}

var readDef = ai.ToolDefinition{
	Name: "memory_read",
	Description: `Read a named project memory file.
All memories are injected into your system prompt at session start; call this
only to re-read a specific memory mid-session (e.g. one that was too large to inject).`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type":        "string",
				"description": `Memory name (slug: lowercase letters, digits, hyphens; e.g. "general", "api-notes")`,
			},
		},
		"required": []string{"name"},
	},
}

var writeDef = ai.ToolDefinition{
	Name: "memory_write",
	Description: `Write (replace) a named project memory file.
Use named files to keep different concerns separate (e.g. "general", "conventions", "architecture").
Maximum ` + fmt.Sprintf("%d", memorystore.MaxBytesPerFile) + ` bytes per file; up to ` + fmt.Sprintf("%d", memorystore.MaxFiles) + ` files per project.
Use this to persist project knowledge across sessions: conventions, architecture decisions,
known issues, preferred patterns, important file locations.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type":        "string",
				"description": `Memory name (slug: lowercase letters, digits, hyphens; e.g. "general", "api-notes")`,
			},
			"content": map[string]any{
				"type":        "string",
				"description": "Markdown content to store (replaces the named file's previous content)",
			},
		},
		"required": []string{"name", "content"},
	},
}

var deleteDef = ai.ToolDefinition{
	Name: "memory_delete",
	Description: `Delete a named project memory file.
Use only when the memory is fully obsolete. This cannot be undone without rewriting from scratch.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type":        "string",
				"description": "Name of the memory file to delete",
			},
		},
		"required": []string{"name"},
	},
}

var promoteDef = ai.ToolDefinition{
	Name: "memory_promote",
	Description: `Move a named memory file to a parent directory's store, making it available to all sub-projects.
Use this when a note turns out to be relevant across the whole project or workspace (e.g. a monorepo root),
not just the current directory. The memory is deleted from the current directory after being moved.
Call memory_list first if you are unsure which memories exist.
target_directory accepts a relative path (e.g. ".." or "../..") or an absolute path; it must be a parent of the current project directory.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type":        "string",
				"description": "Name of the memory to promote",
			},
			"target_directory": map[string]any{
				"type":        "string",
				"description": `Relative path to the target parent directory, e.g. ".." or "../.."`,
			},
		},
		"required": []string{"name", "target_directory"},
	},
}

func (h *Handler) handleMemoryList(_ context.Context, _ map[string]any) (string, error) {
	entries := h.ctx.MemoryStore().List()
	if len(entries) == 0 {
		return "(no project memories stored yet)", nil
	}
	var sb strings.Builder
	for _, e := range entries {
		fmt.Fprintf(&sb, "- %s (%d bytes)\n", e.Name, e.Size)
	}
	return strings.TrimSpace(sb.String()), nil
}

func (h *Handler) handleMemoryRead(_ context.Context, args map[string]any) (string, error) {
	name := toolutil.StringArg(args, "name")
	if name == "" {
		return "error: name is required", nil
	}
	content, err := h.ctx.MemoryStore().ReadNamed(name)
	if err != nil {
		return "error reading project memory: " + err.Error(), nil
	}
	if content == "" {
		return fmt.Sprintf("(no memory named %q exists)", name), nil
	}
	return content, nil
}

func (h *Handler) handleMemoryWrite(_ context.Context, args map[string]any) (string, error) {
	name := toolutil.StringArg(args, "name")
	content := toolutil.StringArg(args, "content")
	if name == "" {
		return "error: name is required", nil
	}
	if err := h.ctx.MemoryStore().WriteNamed(name, content); err != nil {
		return "error writing project memory: " + err.Error(), nil
	}
	return fmt.Sprintf("memory %q saved (%d bytes)", name, len(strings.TrimSpace(content))), nil
}

func (h *Handler) handleMemoryDelete(_ context.Context, args map[string]any) (string, error) {
	name := toolutil.StringArg(args, "name")
	if name == "" {
		return "error: name is required", nil
	}
	if err := h.ctx.MemoryStore().DeleteNamed(name); err != nil {
		return "error deleting project memory: " + err.Error(), nil
	}
	return fmt.Sprintf("memory %q deleted", name), nil
}

func (h *Handler) handleMemoryPromote(_ context.Context, args map[string]any) (string, error) {
	name := toolutil.StringArg(args, "name")
	targetDir := toolutil.StringArg(args, "target_directory")
	if name == "" {
		return "error: name is required", nil
	}
	if targetDir == "" {
		return "error: target_directory is required", nil
	}
	if err := h.ctx.MemoryStore().Promote(name, targetDir); err != nil {
		return "error promoting memory: " + err.Error(), nil
	}
	return fmt.Sprintf("memory %q promoted to %s", name, targetDir), nil
}

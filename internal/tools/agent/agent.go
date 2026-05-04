package agent

import (
	"context"
	"fmt"

	"github.com/icedream/werkler/internal/agents"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/skills"
	"github.com/icedream/werkler/internal/tools/toolutil"
)

// Context is the narrow interface the agent handler needs from the Manager.
type Context interface {
	Skills() []skills.Skill
	Agents() []agents.Agent
	NotifyAgentActivate(name string)
}

// Handler holds the skill and agent tool handlers.
type Handler struct{ ctx Context }

// NewHandler creates a Handler.
func NewHandler(ctx Context) *Handler { return &Handler{ctx: ctx} }

// Tools returns Builtin definitions for use_skill and use_agent.
// Each tool is only included when the corresponding list is non-empty.
func (h *Handler) Tools() []toolutil.Builtin {
	var out []toolutil.Builtin
	if sk := h.ctx.Skills(); len(sk) > 0 {
		names := make([]any, len(sk))
		for i, s := range sk {
			names[i] = s.Name
		}
		out = append(out, toolutil.Builtin{
			Def: ai.ToolDefinition{
				Name:        "use_skill",
				Description: "Load the instructions for a named skill into the conversation.",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{
							"type":        "string",
							"enum":        names,
							"description": "Name of the skill to load",
						},
					},
					"required": []string{"name"},
				},
			},
			Handle: h.handleUseSkill,
		})
	}
	if ag := h.ctx.Agents(); len(ag) > 0 {
		names := make([]any, len(ag))
		for i, a := range ag {
			names[i] = a.Name
		}
		out = append(out, toolutil.Builtin{
			Def: ai.ToolDefinition{
				Name:        "use_agent",
				Description: "Activate a named agent persona for the current session.",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{
							"type":        "string",
							"enum":        names,
							"description": "Agent name to activate",
						},
					},
					"required": []string{"name"},
				},
			},
			Handle: h.handleUseAgent,
		})
	}
	return out
}

func (h *Handler) handleUseSkill(_ context.Context, args map[string]any) (string, error) {
	name := toolutil.StringArg(args, "name")
	for _, s := range h.ctx.Skills() {
		if s.Name == name {
			return s.Content, nil
		}
	}
	return "skill not found: " + name, nil
}

func (h *Handler) handleUseAgent(_ context.Context, args map[string]any) (string, error) {
	name := toolutil.StringArg(args, "name")
	for _, a := range h.ctx.Agents() {
		if a.Name == name {
			h.ctx.NotifyAgentActivate(name)
			return fmt.Sprintf("Agent %q activated.", name), nil
		}
	}
	return fmt.Sprintf("agent not found: %q", name), nil
}

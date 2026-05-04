package user

import (
	"context"
	"fmt"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/tools/toolutil"
)

// Asker is implemented by interactive callers to present a question to the
// user and wait for their answer.
type Asker func(ctx context.Context, question string, choices []string, recommendedChoice string, allowFreeform bool) (string, error)

// PlanConfirmer is implemented by interactive callers to show a plan summary
// and ask whether to proceed.
type PlanConfirmer func(ctx context.Context, summary string) (string, error)

// Context is the narrow interface the user-interaction handler needs.
type Context interface {
	UserAsker() Asker
	PlanConfirmer() PlanConfirmer
}

// Handler holds the user-interaction tool handlers.
type Handler struct{ ctx Context }

// NewHandler creates a Handler.
func NewHandler(ctx Context) *Handler { return &Handler{ctx: ctx} }

// Tools returns Builtin definitions for ask_user and confirm_plan.
func (h *Handler) Tools() []toolutil.Builtin {
	return []toolutil.Builtin{
		{Def: askUserDef, Handle: h.handleAskUser},
		{Def: confirmPlanDef, Handle: h.handleConfirmPlan},
	}
}

var askUserDef = ai.ToolDefinition{
	Name: "ask_user",
	Description: `Ask the user a direct question and wait for their answer.
Use this when the task requires a decision or information that only the user can provide.

REQUIRED: When there are 2–4 known valid answers, you MUST put them in the "choices" array — do NOT embed numbered options in the question string itself.
WRONG: question="What should I do?\n\n1. Deploy now\n2. Review first", choices=[]
RIGHT: question="What should I do?", choices=["Deploy now", "Review first"]

Set allow_freeform to false to restrict the user strictly to those choices.
Set recommended_choice to highlight a suggested option.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"question": map[string]any{
				"type":        "string",
				"description": "The question to present to the user. Must be a plain sentence ending in '?'. Do NOT include numbered or bulleted options — put those in the 'choices' array.",
			},
			"choices": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Predefined answer choices (e.g. [\"Yes\", \"No\"] or [\"Option A\", \"Option B\", \"Option C\"]). Always populate this when there are 2–4 known valid answers.",
			},
			"recommended_choice": map[string]any{
				"type":        "string",
				"description": "The choice you recommend; must exactly match one of the choices",
			},
			"allow_freeform": map[string]any{
				"type":        "boolean",
				"description": "Whether to also accept a custom typed answer (default: true)",
			},
		},
		"required": []string{"question"},
	},
}

var confirmPlanDef = ai.ToolDefinition{
	Name: "confirm_plan",
	Description: `Present the finalised plan to the user and ask whether to proceed with implementation.
Call this ONLY after writing the plan file and completing any review cycles.
Provide a brief 2-3 sentence summary of the plan for the user to review.
The user will choose one of: implement now, implement with autopilot, or reject.
The return value tells you how to proceed:
- "approved": implement immediately in the current conversation turn.
- "approved_with_autopilot": autopilot has been enabled — STOP your response; the autonomous loop will continue.
- "rejected: <reason>": acknowledge the reason and stop; do NOT start implementing.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"summary": map[string]any{
				"type":        "string",
				"description": "Brief 2-3 sentence summary of the plan for the user",
			},
		},
		"required": []string{"summary"},
	},
}

func (h *Handler) handleAskUser(ctx context.Context, args map[string]any) (string, error) {
	question := toolutil.StringArg(args, "question")
	if question == "" {
		return "error: ask_user requires a non-empty question", nil
	}
	choices := toolutil.StringSliceArg(args, "choices")
	recommended := toolutil.StringArg(args, "recommended_choice")
	allowFreeform := true
	if v, ok := args["allow_freeform"].(bool); ok {
		allowFreeform = v
	}
	if !allowFreeform && len(choices) == 0 {
		return "error: ask_user requires at least one choice when allow_freeform is false", nil
	}
	if recommended != "" {
		found := false
		for _, c := range choices {
			if c == recommended {
				found = true
				break
			}
		}
		if !found {
			return fmt.Sprintf("error: recommended_choice %q does not match any provided choice", recommended), nil
		}
	}
	asker := h.ctx.UserAsker()
	if asker == nil {
		return "(ask_user requires interactive mode — run werkler interactively to provide an answer)", nil
	}
	return asker(ctx, question, choices, recommended, allowFreeform)
}

func (h *Handler) handleConfirmPlan(ctx context.Context, args map[string]any) (string, error) {
	confirmer := h.ctx.PlanConfirmer()
	if confirmer == nil {
		return "approved: proceed with implementation", nil
	}
	return confirmer(ctx, toolutil.StringArg(args, "summary"))
}

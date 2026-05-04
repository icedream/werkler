package util

import (
	"context"
	"fmt"
	"time"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/tools/toolutil"
)

// Handler holds the utility tool handlers (calculate, sleep).
// These tools have no external dependencies beyond the tool arguments.
type Handler struct{}

// NewHandler creates a Handler.
func NewHandler() *Handler { return &Handler{} }

// Tools returns Builtin definitions for calculate and sleep.
func (h *Handler) Tools() []toolutil.Builtin {
	return []toolutil.Builtin{
		{Def: calculateDef, Handle: h.handleCalculate},
		{Def: sleepDef, Handle: h.handleSleep},
	}
}

var calculateDef = ai.ToolDefinition{
	Name: "calculate",
	Description: `Evaluate a mathematical expression and return the result.
Use this for arithmetic, unit conversions, and any calculation you would normally
estimate. Supports: +, -, *, /, % (remainder), bitwise &/|/^/<</>>/&^.
Unary minus and plus are supported. Integer literals: 0xff, 0b1010, 0o17.
Constants: pi, e, phi, sqrt2, ln2.
Functions: sqrt, cbrt, abs, floor, ceil, round, trunc, exp, exp2, log, log2,
log10, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, pow, hypot,
mod, min, max. Note: ^ is bitwise XOR; use pow(x,y) for exponentiation.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"expression": map[string]any{
				"type":        "string",
				"description": `Mathematical expression to evaluate (e.g. "sqrt(2) * pi", "2**8" is invalid — use pow(2,8))`,
			},
		},
		"required": []string{"expression"},
	},
}

var sleepDef = ai.ToolDefinition{
	Name: "sleep",
	Description: `Pause execution for a duration or until a specific time.
Use this ONLY when you genuinely need to wait (e.g. polling for a file,
waiting for a background process, or an explicit user request to delay).
Avoid calling this unless strictly necessary.
Specify either "seconds" (float, max 600) OR "until" (RFC3339 timestamp, max 600s ahead).`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"seconds": map[string]any{
				"type":        "number",
				"description": "Seconds to sleep (max 600)",
			},
			"until": map[string]any{
				"type":        "string",
				"description": "RFC3339 timestamp to sleep until (max 600 s in the future)",
			},
		},
	},
}

const maxSleepSeconds = 600.0

func (h *Handler) handleCalculate(_ context.Context, args map[string]any) (string, error) {
	expr := toolutil.StringArg(args, "expression")
	if expr == "" {
		return "error: expression is required", nil
	}
	v, err := evalExpression(expr)
	if err != nil {
		return fmt.Sprintf("error: %s", err), nil
	}
	return fmt.Sprintf("%s = %s", expr, formatResult(v)), nil
}

func (h *Handler) handleSleep(ctx context.Context, args map[string]any) (string, error) {
	var dur time.Duration
	if until := toolutil.StringArg(args, "until"); until != "" {
		t, err := time.Parse(time.RFC3339, until)
		if err != nil {
			return fmt.Sprintf("error: invalid 'until' timestamp %q: %s", until, err), nil
		}
		dur = time.Until(t)
	} else if v, ok := args["seconds"]; ok {
		secs, ok2 := toolutil.ToFloat64(v)
		if !ok2 || secs < 0 {
			return "error: 'seconds' must be a non-negative number", nil
		}
		dur = time.Duration(secs * float64(time.Second))
	} else {
		return "error: specify either 'seconds' or 'until'", nil
	}
	if dur <= 0 {
		return "0s elapsed (target time already passed)", nil
	}
	if dur > maxSleepSeconds*time.Second {
		dur = maxSleepSeconds * time.Second
	}
	start := time.Now()
	select {
	case <-ctx.Done():
		return fmt.Sprintf("sleep cancelled after %s", time.Since(start).Round(time.Millisecond)), nil
	case <-time.After(dur):
		return fmt.Sprintf("slept %s", dur.Round(time.Millisecond)), nil
	}
}

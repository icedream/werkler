package modeleval

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/icedream/werkler/internal/ai"
)

// ToolResult records a single tool call made during a scenario and the
// scripted response that was returned to the model.
type ToolResult struct {
	CallID string
	Name   string
	Args   map[string]any
	Output string // what the scripted ToolHandler returned
}

// Turn captures one full round-trip: the model's response and every tool
// result injected back into the conversation. The final turn (when the model
// stops calling tools or MaxTurns is exhausted) has an empty ToolResults slice.
type Turn struct {
	ModelResponse ai.Message
	ToolResults   []ToolResult
	Elapsed       time.Duration
}

// ScenarioCase describes a multi-turn agentic test. The model is given an
// initial prompt and a set of tools. It calls tools zero or more times;
// ToolHandler returns scripted (deterministic) responses so no real
// side-effects occur. TraceCheck asserts properties over the complete recorded
// trace once the scenario ends.
type ScenarioCase struct {
	// Name is a short kebab-case identifier used for filtering.
	Name string
	// Description explains what the scenario validates.
	Description string

	// Messages is the initial conversation to send on the first turn.
	Messages []ai.Message
	// Tools is the set of tool definitions the model may call.
	Tools []ai.ToolDefinition

	// ToolHandler maps (toolName, args) → (output, err).
	// Return ("some text", nil) for any response the model should see,
	// including simulated errors like "error: EPERM".
	// Return ("", err) only for infrastructure failures that abort the run.
	ToolHandler func(name string, args map[string]any) (string, error)

	// MaxTurns is the upper bound on model→tool round-trips. 0 means 10.
	MaxTurns int

	// TraceCheck asserts properties over the full trace after the scenario
	// concludes (final text answer or turn limit reached).
	TraceCheck func(turns []Turn) error

	// Repeat runs the scenario this many times and reports the pass rate.
	// 0 and 1 both mean a single run.
	Repeat int
}

// ScenarioResult holds the outcome of running one ScenarioCase (possibly
// multiple times when Repeat > 1).
type ScenarioResult struct {
	Case      *ScenarioCase
	Runs      []ScenarioRunResult
	PassCount int
}

// ScenarioRunResult is the outcome of a single execution of a ScenarioCase.
type ScenarioRunResult struct {
	Turns  []Turn
	Err    error
	Passed bool
}

// PassRate returns the fraction of runs that passed (0.0–1.0).
func (r *ScenarioResult) PassRate() float64 {
	if len(r.Runs) == 0 {
		return 0
	}
	return float64(r.PassCount) / float64(len(r.Runs))
}

// RunScenario executes a ScenarioCase (possibly multiple times) against the
// given client and returns the aggregated result.
func RunScenario(ctx context.Context, client ai.Completer, sc *ScenarioCase) *ScenarioResult {
	repeats := sc.Repeat
	if repeats < 1 {
		repeats = 1
	}
	result := &ScenarioResult{Case: sc, Runs: make([]ScenarioRunResult, 0, repeats)}
	for range repeats {
		rr := runScenarioOnce(ctx, client, sc)
		result.Runs = append(result.Runs, rr)
		if rr.Passed {
			result.PassCount++
		}
	}
	return result
}

func runScenarioOnce(ctx context.Context, client ai.Completer, sc *ScenarioCase) ScenarioRunResult {
	maxTurns := sc.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 10
	}

	// Working copy of the conversation history.
	messages := make([]ai.Message, len(sc.Messages))
	copy(messages, sc.Messages)

	var turns []Turn

	for turn := 0; turn < maxTurns; turn++ {
		start := time.Now()
		response, err := client.Complete(ctx, messages, sc.Tools)
		elapsed := time.Since(start)
		if err != nil {
			return ScenarioRunResult{
				Turns: turns,
				Err:   fmt.Errorf("turn %d: Complete failed: %w", turn+1, err),
			}
		}

		// Append the assistant message to history.
		messages = append(messages, response)

		// If the model made no tool calls this turn, the scenario is done.
		if len(response.ToolCalls) == 0 {
			turns = append(turns, Turn{ModelResponse: response, Elapsed: elapsed})
			break
		}

		// Dispatch every tool call through the scripted handler.
		var toolResults []ToolResult
		for _, tc := range response.ToolCalls {
			output, handlerErr := sc.ToolHandler(tc.Name, tc.Arguments)
			if handlerErr != nil {
				return ScenarioRunResult{
					Turns: turns,
					Err:   fmt.Errorf("turn %d: tool %q handler error: %w", turn+1, tc.Name, handlerErr),
				}
			}

			toolResults = append(toolResults, ToolResult{
				CallID: tc.ID,
				Name:   tc.Name,
				Args:   tc.Arguments,
				Output: output,
			})

			// Inject the tool result into the conversation history.
			messages = append(messages, ai.Message{
				Role:       "tool",
				Content:    output,
				ToolCallID: tc.ID,
			})
		}

		turns = append(turns, Turn{
			ModelResponse: response,
			ToolResults:   toolResults,
			Elapsed:       elapsed,
		})
	}

	// Run the trace assertion.
	if sc.TraceCheck != nil {
		if checkErr := sc.TraceCheck(turns); checkErr != nil {
			return ScenarioRunResult{Turns: turns, Err: checkErr}
		}
	}

	return ScenarioRunResult{Turns: turns, Passed: true}
}

// RunAllScenarios executes all provided scenario cases and returns results.
func RunAllScenarios(ctx context.Context, client ai.Completer, cases []*ScenarioCase) []*ScenarioResult {
	results := make([]*ScenarioResult, len(cases))
	for i, sc := range cases {
		results[i] = RunScenario(ctx, client, sc)
	}
	return results
}

// --- Trace check helpers ---

// allToolCalls returns a flat, chronologically ordered slice of every
// ToolResult recorded across all turns.
func allToolCalls(turns []Turn) []ToolResult {
	var out []ToolResult
	for _, t := range turns {
		out = append(out, t.ToolResults...)
	}
	return out
}

// TraceContainsToolCall returns a TraceCheck that fails unless the named tool
// was called at least once.
func TraceContainsToolCall(toolName string) func([]Turn) error {
	return func(turns []Turn) error {
		for _, tr := range allToolCalls(turns) {
			if tr.Name == toolName {
				return nil
			}
		}
		return fmt.Errorf("expected tool %q to be called but it never was", toolName)
	}
}

// TraceNeverCalls returns a TraceCheck that fails if the named tool was called
// at any point.
func TraceNeverCalls(toolName string) func([]Turn) error {
	return func(turns []Turn) error {
		for _, tr := range allToolCalls(turns) {
			if tr.Name == toolName {
				return fmt.Errorf("tool %q was called but should never be called", toolName)
			}
		}
		return nil
	}
}

// TraceToolCallOrder returns a TraceCheck that fails unless the first
// occurrence of firstTool precedes the first occurrence of secondTool.
func TraceToolCallOrder(firstTool, secondTool string) func([]Turn) error {
	return func(turns []Turn) error {
		alls := allToolCalls(turns)
		firstIdx, secondIdx := -1, -1
		for i, tr := range alls {
			if firstIdx == -1 && tr.Name == firstTool {
				firstIdx = i
			}
			if secondIdx == -1 && tr.Name == secondTool {
				secondIdx = i
			}
			if firstIdx != -1 && secondIdx != -1 {
				break
			}
		}
		if firstIdx == -1 {
			return fmt.Errorf("tool %q was never called", firstTool)
		}
		if secondIdx == -1 {
			return fmt.Errorf("tool %q was never called", secondTool)
		}
		if firstIdx >= secondIdx {
			return fmt.Errorf("expected %q (first at call #%d) before %q (first at call #%d)",
				firstTool, firstIdx+1, secondTool, secondIdx+1)
		}
		return nil
	}
}

// TraceToolCallArgs returns a TraceCheck that fails unless at least one call
// to toolName satisfies pred.
func TraceToolCallArgs(toolName string, pred func(args map[string]any) bool) func([]Turn) error {
	return func(turns []Turn) error {
		for _, tr := range allToolCalls(turns) {
			if tr.Name == toolName && pred(tr.Args) {
				return nil
			}
		}
		return fmt.Errorf("no call to %q satisfied the argument predicate", toolName)
	}
}

// TraceNeverCallsWithArgs returns a TraceCheck that fails if ANY call to
// toolName satisfies pred. Use this to prohibit dangerous argument patterns.
func TraceNeverCallsWithArgs(toolName string, pred func(args map[string]any) bool, reason string) func([]Turn) error {
	return func(turns []Turn) error {
		for _, tr := range allToolCalls(turns) {
			if tr.Name == toolName && pred(tr.Args) {
				return fmt.Errorf("tool %q was called with forbidden args (%s): %v", toolName, reason, tr.Args)
			}
		}
		return nil
	}
}

// AllTraceChecks returns a TraceCheck that runs every supplied check and
// aggregates all failures into a single error (does not short-circuit).
func AllTraceChecks(checks ...func([]Turn) error) func([]Turn) error {
	return func(turns []Turn) error {
		var errs []string
		for _, c := range checks {
			if err := c(turns); err != nil {
				errs = append(errs, err.Error())
			}
		}
		if len(errs) > 0 {
			return fmt.Errorf("%s", strings.Join(errs, "; "))
		}
		return nil
	}
}

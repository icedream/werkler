// Package modeleval provides a model evaluation harness for testing AI model
// compatibility with werkler's tool definitions and system prompt. Run test
// cases against any OpenAI-compatible endpoint to validate tool calling
// behaviour before deploying a new model.
package modeleval

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/icedream/werkler/internal/ai"
)

// TestCase describes a single evaluation scenario.
type TestCase struct {
	// Name is a short kebab-case identifier used for filtering.
	Name string
	// Description explains what the test validates.
	Description string
	// Messages is the full conversation to send, including system prompt.
	// Use BuildMessages to construct these from werkler's real system prompt.
	Messages []ai.Message
	// Tools is the set of tool definitions available to the model.
	Tools []ai.ToolDefinition
	// Check validates the model's response. Return a non-nil error to fail.
	Check func(resp ai.Message) error
	// Repeat runs the case this many times and reports the pass rate.
	// 0 and 1 both mean a single run.
	Repeat int
}

// Result holds the outcome of running one TestCase (possibly multiple times).
type Result struct {
	Case      *TestCase
	Runs      []RunResult
	PassCount int
}

// RunResult is the outcome of a single execution of a TestCase.
type RunResult struct {
	Passed   bool
	Err      error
	Response ai.Message
	Elapsed  time.Duration
}

// PassRate returns the fraction of runs that passed (0.0–1.0).
func (r *Result) PassRate() float64 {
	if len(r.Runs) == 0 {
		return 0
	}
	return float64(r.PassCount) / float64(len(r.Runs))
}

// Run executes a single TestCase against the given client.
func Run(ctx context.Context, client ai.Completer, tc *TestCase) *Result {
	repeats := tc.Repeat
	if repeats < 1 {
		repeats = 1
	}
	result := &Result{Case: tc, Runs: make([]RunResult, 0, repeats)}
	for range repeats {
		rr := runOnce(ctx, client, tc)
		result.Runs = append(result.Runs, rr)
		if rr.Passed {
			result.PassCount++
		}
	}
	return result
}

func runOnce(ctx context.Context, client ai.Completer, tc *TestCase) RunResult {
	start := time.Now()
	resp, err := client.Complete(ctx, tc.Messages, tc.Tools)
	elapsed := time.Since(start)
	if err != nil {
		return RunResult{Passed: false, Err: fmt.Errorf("API error: %w", err), Elapsed: elapsed}
	}
	checkErr := tc.Check(resp)
	return RunResult{
		Passed:   checkErr == nil,
		Err:      checkErr,
		Response: resp,
		Elapsed:  elapsed,
	}
}

// RunAll executes all provided test cases and returns their results.
func RunAll(ctx context.Context, client ai.Completer, cases []*TestCase) []*Result {
	results := make([]*Result, len(cases))
	for i, tc := range cases {
		results[i] = Run(ctx, client, tc)
	}
	return results
}

// --- Check helpers ---

// CheckHasContent passes when the response contains non-empty text.
func CheckHasContent() func(ai.Message) error {
	return func(resp ai.Message) error {
		if strings.TrimSpace(resp.Content) == "" {
			return fmt.Errorf("expected text content but got empty response (tool_calls=%d)", len(resp.ToolCalls))
		}
		return nil
	}
}

// CheckNotEmpty passes when the response has either text content or at least
// one tool call. Catches the silent-empty-response bug (tokens generated but
// neither content nor tool calls surfaced).
func CheckNotEmpty() func(ai.Message) error {
	return func(resp ai.Message) error {
		if resp.Content == "" && len(resp.ToolCalls) == 0 {
			return fmt.Errorf("response is completely empty: no content and no tool calls")
		}
		return nil
	}
}

// CheckToolCall passes when the response contains a call to a tool whose name
// matches nameContains (case-insensitive substring).
func CheckToolCall(nameContains string) func(ai.Message) error {
	return func(resp ai.Message) error {
		for _, tc := range resp.ToolCalls {
			if strings.Contains(strings.ToLower(tc.Name), strings.ToLower(nameContains)) {
				return nil
			}
		}
		if len(resp.ToolCalls) == 0 {
			return fmt.Errorf("expected tool call containing %q but got no tool calls (content=%q)", nameContains, resp.Content)
		}
		names := make([]string, len(resp.ToolCalls))
		for i, tc := range resp.ToolCalls {
			names[i] = tc.Name
		}
		return fmt.Errorf("expected tool call containing %q but got: %s", nameContains, strings.Join(names, ", "))
	}
}

// CheckToolCallArg passes when there is a tool call matching nameContains whose
// decoded arguments contain a field whose value includes argValueContains
// (case-insensitive).
func CheckToolCallArg(nameContains, argValueContains string) func(ai.Message) error {
	return func(resp ai.Message) error {
		for _, tc := range resp.ToolCalls {
			if !strings.Contains(strings.ToLower(tc.Name), strings.ToLower(nameContains)) {
				continue
			}
			// Cheaply scan the raw argument JSON for the expected value.
			raw := fmt.Sprintf("%v", tc.Arguments)
			if strings.Contains(strings.ToLower(raw), strings.ToLower(argValueContains)) {
				return nil
			}
			return fmt.Errorf("tool %q called but argument value %q not found in args: %v", tc.Name, argValueContains, tc.Arguments)
		}
		return fmt.Errorf("no tool call matching %q found", nameContains)
	}
}

// CheckResponseContains passes when the response content contains substr
// (case-insensitive). failMsg is included in the error when the check fails.
func CheckResponseContains(substr, failMsg string) func(ai.Message) error {
	return func(resp ai.Message) error {
		if strings.Contains(strings.ToLower(resp.Content), strings.ToLower(substr)) {
			return nil
		}
		return fmt.Errorf("%s: %q not found in response (content=%q)", failMsg, substr, resp.Content)
	}
}

// All returns a check that passes only when every provided check passes.
func All(checks ...func(ai.Message) error) func(ai.Message) error {
	return func(resp ai.Message) error {
		for _, c := range checks {
			if err := c(resp); err != nil {
				return err
			}
		}
		return nil
	}
}

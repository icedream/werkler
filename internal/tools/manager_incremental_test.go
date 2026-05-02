// Package tools tests for incremental tool execution support.
package tools

import (
	"testing"
	"time"
)

// TestCallToolAsync tests that CallToolAsync returns a channel that receives results.
func TestCallToolAsync(t *testing.T) {
	m := New(nil, nil, nil)
	ctx := t.Context()

	resultCh := m.CallToolAsync(ctx, "calculate", map[string]any{"seconds": 1})

	// Should receive a result within timeout
	select {
	case result := <-resultCh:
		if result.Error != nil {
			t.Errorf("Expected no error, got: %v", result.Error)
		}
		if result.Output == "" {
			t.Error("Expected non-empty output")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Timeout waiting for tool result")
	}
}

// TestCallToolsParallel tests that CallToolsParallel executes tools concurrently.
func TestCallToolsParallel(t *testing.T) {
	m := New(nil, nil, nil)
	ctx := t.Context()

	requests := []ToolCallRequest{
		{ID: "tool1", Name: "calculate", Args: map[string]any{"expression": "not_a_valid_expression"}},
		{ID: "tool2", Name: "calculate", Args: map[string]any{"expression": "not_a_valid_expression"}},
		{ID: "tool3", Name: "calculate", Args: map[string]any{"expression": "not_a_valid_expression"}},
	}

	chMap := m.CallToolsParallel(ctx, requests)

	// Wait for all results
	results, err := m.WaitToolResults(ctx, chMap, 5)
	if err != nil {
		t.Fatalf("WaitToolResults failed: %v", err)
	}

	// Check results
	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}

	for id, result := range results {
		if result == nil {
			t.Errorf("Result for %q is nil", id)
			continue
		}
		if result.Error != nil {
			t.Errorf("Expected no error for %q, got: %v", id, result.Error)
		}
		if result.Output == "" {
			t.Errorf("Expected non-empty output for %q", id)
		}
	}
}

// TestToolCallRequest tests ToolCallRequest struct.
func TestToolCallRequest(t *testing.T) {
	req := ToolCallRequest{
		ID:   "test-id",
		Name: "test_tool",
		Args: map[string]any{"key": "value"},
	}

	if req.ID != "test-id" {
		t.Errorf("Expected ID 'test-id', got '%s'", req.ID)
	}
	if req.Name != "test_tool" {
		t.Errorf("Expected Name 'test_tool', got '%s'", req.Name)
	}
	if req.Args["key"] != "value" {
		t.Errorf("Expected Args['key'] = 'value', got '%v'", req.Args["key"])
	}
}

// TestToolResult tests ToolResult struct.
func TestToolResult(t *testing.T) {
	result := ToolResult{
		ToolCallID: "call-123",
		Name:       "test_tool",
		Arguments:  map[string]any{"arg": "value"},
		Output:     "test output",
		Error:      nil,
		Completed:  true,
	}

	if result.ToolCallID != "call-123" {
		t.Errorf("Expected ToolCallID 'call-123', got '%s'", result.ToolCallID)
	}
	if result.Name != "test_tool" {
		t.Errorf("Expected Name 'test_tool', got '%s'", result.Name)
	}
	if result.Output != "test output" {
		t.Errorf("Expected Output 'test output', got '%s'", result.Output)
	}
	if result.Error != nil {
		t.Errorf("Expected no error, got %v", result.Error)
	}
	if !result.Completed {
		t.Error("Expected Completed to be true")
	}
}

// TestToolResultChan tests ToolResultChan type.
func TestToolResultChan(t *testing.T) {
	ch := make(ToolResultChan, 1)

	result := ToolResult{
		Name: "test",
	}

	// Send and close
	select {
	case ch <- result:
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout sending to channel")
	}
	close(ch)

	// Receive
	received := <-ch
	if received.Name != "test" {
		t.Errorf("Expected Name 'test', got '%s'", received.Name)
	}
}

// TestWaitToolResultsWithTimeout tests WaitToolResults with timeout.
func TestWaitToolResultsWithTimeout(t *testing.T) {
	m := New(nil, nil, nil)
	ctx := t.Context()

	// Create a tool that will take a long time
	requests := []ToolCallRequest{
		{ID: "tool1", Name: "sleep", Args: map[string]any{"seconds": 5}},
	}

	chMap := m.CallToolsParallel(ctx, requests)

	// Wait with very short timeout - should timeout
	_, err := m.WaitToolResults(ctx, chMap, 1)
	if err == nil {
		t.Error("Expected timeout or error, got nil")
	}
	// We don't check the exact error because it might be timeout or context cancelled
}

// TestWaitToolResultsAllComplete tests WaitToolResults when all tools complete.
func TestWaitToolResultsAllComplete(t *testing.T) {
	m := New(nil, nil, nil)
	ctx := t.Context()

	// Create tools that complete quickly
	requests := []ToolCallRequest{
		{ID: "tool1", Name: "calculate", Args: map[string]any{"expression": "not_a_valid_expression"}},
		{ID: "tool2", Name: "calculate", Args: map[string]any{"expression": "not_a_valid_expression"}},
	}

	chMap := m.CallToolsParallel(ctx, requests)

	// Wait for all results
	results, err := m.WaitToolResults(ctx, chMap, 5)
	if err != nil {
		t.Fatalf("WaitToolResults failed: %v", err)
	}

	// All should complete
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	for id, result := range results {
		if result == nil {
			t.Errorf("Result for %q is nil", id)
			continue
		}
		if result.Error != nil {
			t.Errorf("Expected no error for %q, got: %v", id, result.Error)
		}
		if !result.Completed {
			t.Errorf("Expected Completed=true for %q", id)
		}
	}
}

// TestWaitToolResultsWithError tests WaitToolResults when one tool fails.
func TestWaitToolResultsWithError(t *testing.T) {
	m := New(nil, nil, nil)
	ctx := t.Context()

	// Create one valid tool and one invalid
	requests := []ToolCallRequest{
		{ID: "tool1", Name: "calculate", Args: map[string]any{"expression": "not_a_valid_expression"}},
		{ID: "tool2", Name: "calculate", Args: map[string]any{}},
	}

	chMap := m.CallToolsParallel(ctx, requests)

	// Wait for all results
	results, err := m.WaitToolResults(ctx, chMap, 5)
	if err != nil {
		t.Fatalf("WaitToolResults failed: %v", err)
	}

	// Check results
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// The invalid tool should have an error message in output
	if result, ok := results["tool2"]; ok {
		if result == nil {
			t.Errorf("Result for tool2 is nil")
		} else {
			if result.Error != nil {
				t.Errorf("Expected no error for tool2, got: %v", result.Error)
			}
			if result.Output == "" {
				t.Errorf("Expected non-empty output for tool2")
			}
			t.Logf("Tool2 output: %s", result.Output)
		}
	}
}

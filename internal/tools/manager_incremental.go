// Package tools wraps an MCP ToolManager and augments it with built-in tools
// (process execution, etc.) that live entirely within werkler.
//
// This file adds asynchronous tool execution support for the incremental context
// streaming architecture.
//
// Responsibilities:
//   - Provide async tool execution via CallToolAsync
//   - Return results incrementally via channels
//   - Support parallel tool execution
//
// The async approach enables:
//  1. Tool results to be streamed back immediately as they complete
//  2. Multiple independent tools to execute concurrently
//  3. Better integration with the incremental context streaming loop
package tools

import (
	"context"
	"sync"
	"time"
)

// ToolResult represents a single tool execution result.
type ToolResult struct {
	ToolCallID string
	Name       string
	Arguments  map[string]any
	Output     string
	Error      error
	Completed  bool
}

// ToolResultChan is a channel that receives tool execution results.
type ToolResultChan chan ToolResult

// CallToolAsync executes a tool asynchronously and returns a channel that will
// receive the result when complete. This enables parallel tool execution and
// incremental result streaming.
//
// The returned channel is buffered and closed when the tool execution completes.
// Use a timeout on the channel receive to avoid blocking indefinitely.
//
// Example usage:
//
//	ch := m.CallToolAsync(ctx, "file_read", map[string]any{"path": "/tmp/test.txt"})
//	result := <-ch // blocks until completion or timeout
//
// The original CallTool method remains available for synchronous execution.
func (m *Manager) CallToolAsync(ctx context.Context, name string, args map[string]any) ToolResultChan {
	ch := make(chan ToolResult, 1)

	go func() {
		result := ToolResult{
			ToolCallID: "",
			Name:       name,
			Arguments:  args,
			Completed:  true,
		}

		// Execute the tool synchronously (can be modified to support async execution)
		// For now, we keep synchronous execution but return results via channel
		// Call CallToolWithParts directly to handle nil wrapped manager
		output, _, err := m.CallToolWithParts(ctx, name, args)

		result.Output = output
		result.Error = err

		// Send result and close channel
		select {
		case ch <- result:
		default:
			// Channel is closed or blocked
		}
		close(ch)
	}()

	return ch
}

// CallToolsParallel executes multiple tools concurrently and returns a map of
// result channels. This enables true parallel execution of independent tools.
//
// The returned channels are buffered and closed when tool execution completes.
// Use a timeout on each channel receive to avoid blocking indefinitely.
//
// Example usage:
//
//	chMap := m.CallToolsParallel(ctx, []ToolCallRequest{
//	    {Name: "file_read", Args: map[string]any{"path": "/tmp/a.txt"}},
//	    {Name: "file_read", Args: map[string]any{"path": "/tmp/b.txt"}},
//	    {Name: "file_read", Args: map[string]any{"path": "/tmp/c.txt"}},
//	})
//	// Wait for all results with timeout
//	for _, ch := range chMap {
//	    result := <-ch
//	    if result.Error != nil {
//	        // Handle error
//	    }
//	    // Process result.Output
//	}
func (m *Manager) CallToolsParallel(ctx context.Context, requests []ToolCallRequest) map[string]ToolResultChan {
	chMap := make(map[string]ToolResultChan)

	for _, req := range requests {
		chMap[req.ID] = m.CallToolAsync(ctx, req.Name, req.Args)
	}

	return chMap
}

// ToolCallRequest represents a single tool call for parallel execution.
type ToolCallRequest struct {
	ID   string
	Name string
	Args map[string]any
}

// WaitToolResults waits for all tool results with a timeout and returns a map
// of results. This is a convenience function for parallel tool execution.
//
// Returns error if any tool fails or if timeout is reached.
func (m *Manager) WaitToolResults(ctx context.Context, chMap map[string]ToolResultChan, timeoutSecs int) (map[string]*ToolResult, error) {
	type result struct {
		id   string
		res  *ToolResult
		err  error
		done bool
	}

	results := make(map[string]*result)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for id, ch := range chMap {
		wg.Add(1)
		go func(toolID string, resultCh ToolResultChan) {
			defer wg.Done()

			select {
			case res, ok := <-resultCh:
				if !ok {
					return // Channel closed
				}
				mu.Lock()
				results[toolID] = &result{
					id:   toolID,
					res:  &res,
					err:  res.Error,
					done: true,
				}
				mu.Unlock()
			case <-ctx.Done():
				mu.Lock()
				results[toolID] = &result{
					id:   toolID,
					done: true,
					err:  ctx.Err(),
				}
				mu.Unlock()
			}
		}(id, ch)
	}

	// Wait for all goroutines with timeout
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// All completed
		break
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(time.Duration(timeoutSecs) * time.Second):
		// Timeout
		return nil, context.DeadlineExceeded
	}

	// Check for errors
	for _, r := range results {
		if r.err != nil {
			return nil, r.err
		}
	}

	// Return successful results
	successful := make(map[string]*ToolResult)
	for _, r := range results {
		successful[r.id] = r.res
	}

	return successful, nil
}

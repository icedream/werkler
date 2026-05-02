// Package sessioncontext provides session-level context management for incremental tool execution.
// This maintains a persistent context object in memory and only sends incremental
// updates to the LLM on each request.
package sessioncontext

import (
	"time"
)

// Message represents a single message in the conversation.
type Message struct {
	Role       string
	Content    string
	Parts      []ImagePart
	Reasoning  string
	ToolCallID string
	ToolCalls  []ToolCall
}

// ToolDefinition describes a callable tool for the AI.
type ToolDefinition struct {
	Name        string
	Description string
	InputSchema any
}

// ToolCall represents a tool invocation requested by the assistant.
type ToolCall struct {
	ID        string
	Name      string
	Arguments map[string]any
	Status    string
	Result    string
	Error     error
	Idx       *int
}

// ImagePart holds an image to include in a user message.
type ImagePart struct {
	URL      string
	Data     []byte
	Name     string
	MIMEType string
}

// ModelInfo identifies a model available from a provider.
type ModelInfo struct {
	Model    string
	Provider string
	Input    int
	Output   int
	Context  ContextWindow
	Cost     Cost
}

// ContextWindow contains context-related information.
type ContextWindow struct {
	MaxTokens int
}

// Cost holds token consumption statistics.
type Cost struct {
	Input      float64
	Output     float64
	CacheRead  float64
	CacheWrite float64
	Total      float64
}

// Context represents the persistent state of an agent session.
// This is maintained in memory and only incremental updates are sent to the LLM.
type Context struct {
	// SystemPrompt is the system-level instructions for the AI.
	SystemPrompt string

	// Messages contain the conversation history (user/assistant/toolResult only).
	// Tool calls are NOT included here - they're handled separately.
	Messages []Message

	// Tools defines available tools for this context.
	Tools []ToolDefinition

	// Model information for this context.
	Model ModelInfo

	// ThinkingLevel controls the AI's thinking behavior ("low", "medium", "high", "off").
	ThinkingLevel string

	// SessionID provides a unique identifier for cache-aware backends.
	SessionID string

	// Streaming state
	IsStreaming    bool
	StreamingModel *StreamingModel

	// Pending tool calls that have been requested but not yet executed.
	PendingToolCalls []ToolCall

	// Tool results that have been executed but not yet added to messages.
	ToolResults []ToolResult

	// Metadata for context tracking
	LastUpdate time.Time
}

// StreamingModel tracks the current streaming request.
type StreamingModel struct {
	CallID     string
	StartTime  time.Time
	TokenCount int
}

// ToolResult represents a tool execution result.
type ToolResult struct {
	CallID string
	Name   string
	Result string
	Error  error
}

// NewContext creates a new context with the provided initial state.
func NewContext(systemPrompt string, messages []Message, tools []ToolDefinition, model ModelInfo) *Context {
	return &Context{
		SystemPrompt:  systemPrompt,
		Messages:      messages,
		Tools:         tools,
		Model:         model,
		ThinkingLevel: "off",
		LastUpdate:    time.Now(),
	}
}

// Clone creates a deep copy of the context.
func (c *Context) Clone() *Context {
	cloned := &Context{
		SystemPrompt:  c.SystemPrompt,
		Model:         c.Model,
		ThinkingLevel: c.ThinkingLevel,
		SessionID:     c.SessionID,
		IsStreaming:   c.IsStreaming,
	}

	// Deep copy messages
	if len(c.Messages) > 0 {
		cloned.Messages = make([]Message, len(c.Messages))
		copy(cloned.Messages, c.Messages)
	}

	// Deep copy tools
	if len(c.Tools) > 0 {
		cloned.Tools = make([]ToolDefinition, len(c.Tools))
		copy(cloned.Tools, c.Tools)
	}

	// Clone streaming model
	if c.StreamingModel != nil {
		cloned.StreamingModel = &StreamingModel{
			CallID:     c.StreamingModel.CallID,
			StartTime:  c.StreamingModel.StartTime,
			TokenCount: c.StreamingModel.TokenCount,
		}
	}

	// Deep copy pending tool calls
	if len(c.PendingToolCalls) > 0 {
		cloned.PendingToolCalls = make([]ToolCall, len(c.PendingToolCalls))
		copy(cloned.PendingToolCalls, c.PendingToolCalls)
	}

	// Deep copy tool results
	if len(c.ToolResults) > 0 {
		cloned.ToolResults = make([]ToolResult, len(c.ToolResults))
		copy(cloned.ToolResults, c.ToolResults)
	}

	cloned.LastUpdate = time.Now()
	return cloned
}

// AddMessage adds a message to the context.
// This is the incremental update mechanism - only new messages are added.
func (c *Context) AddMessage(msg Message) {
	c.Messages = append(c.Messages, msg)
	c.LastUpdate = time.Now()
}

// AddToolResult adds a tool result to be executed.
func (c *Context) AddToolResult(result ToolResult) {
	c.ToolResults = append(c.ToolResults, result)
	c.LastUpdate = time.Now()
}

// AddToolCall adds a pending tool call.
func (c *Context) AddToolCall(call ToolCall) {
	c.PendingToolCalls = append(c.PendingToolCalls, call)
	c.LastUpdate = time.Now()
}

// MarkToolAsPending marks a tool call as pending execution.
func (c *Context) MarkToolAsPending(callID string) {
	for i := range c.PendingToolCalls {
		if c.PendingToolCalls[i].ID == callID {
			c.PendingToolCalls[i].Status = "pending"
			c.LastUpdate = time.Now()
			return
		}
	}
}

// MarkToolAsExecuting marks a tool call as being executed.
func (c *Context) MarkToolAsExecuting(callID string) {
	for i := range c.PendingToolCalls {
		if c.PendingToolCalls[i].ID == callID {
			c.PendingToolCalls[i].Status = "executing"
			c.LastUpdate = time.Now()
			return
		}
	}
}

// MarkToolAsCompleted marks a tool call as completed with a result.
func (c *Context) MarkToolAsCompleted(callID string, result string, err error) {
	for i := range c.PendingToolCalls {
		if c.PendingToolCalls[i].ID == callID {
			c.PendingToolCalls[i].Status = "completed"
			c.PendingToolCalls[i].Result = result
			c.PendingToolCalls[i].Error = err
			c.LastUpdate = time.Now()
			return
		}
	}
}

// MarkToolAsFailed marks a tool call as failed.
func (c *Context) MarkToolAsFailed(callID string, err error) {
	for i := range c.PendingToolCalls {
		if c.PendingToolCalls[i].ID == callID {
			c.PendingToolCalls[i].Status = "failed"
			c.PendingToolCalls[i].Error = err
			c.LastUpdate = time.Now()
			return
		}
	}
}

// ResetPendingToolCalls clears all pending tool calls.
func (c *Context) ResetPendingToolCalls() {
	c.PendingToolCalls = nil
	c.LastUpdate = time.Now()
}

// ResetToolResults clears all tool results.
func (c *Context) ResetToolResults() {
	c.ToolResults = nil
	c.LastUpdate = time.Now()
}

// ClearMessages clears the messages array while keeping other state.
func (c *Context) ClearMessages() {
	c.Messages = nil
	c.LastUpdate = time.Now()
}

// AppendMessages appends new messages to the context.
func (c *Context) AppendMessages(newMessages []Message) {
	c.Messages = append(c.Messages, newMessages...)
	c.LastUpdate = time.Now()
}

// GetLastMessage returns the most recent message.
func (c *Context) GetLastMessage() *Message {
	if len(c.Messages) == 0 {
		return nil
	}
	return &c.Messages[len(c.Messages)-1]
}

// GetRecentMessages returns the last N user/assistant messages.
func (c *Context) GetRecentMessages(n int) []Message {
	var filtered []Message
	for i := len(c.Messages) - 1; i >= 0 && len(filtered) < n; i-- {
		msg := c.Messages[i]
		if msg.Role == "user" || msg.Role == "assistant" || msg.Role == "tool" {
			filtered = append(filtered, msg)
		}
	}
	// Reverse to maintain order
	for i, j := 0, len(filtered)-1; i < j; i, j = i+1, j-1 {
		filtered[i], filtered[j] = filtered[j], filtered[i]
	}
	return filtered
}

// GetContextForLLM prepares the context for the LLM by filtering out tool calls
// and returning only user/assistant/toolResult messages.
func (c *Context) GetContextForLLM() []Message {
	// Filter to keep only messages suitable for LLM input
	// (user, assistant, and tool result messages)
	llmMessages := make([]Message, 0, len(c.Messages))
	for _, msg := range c.Messages {
		if msg.Role == "user" || msg.Role == "assistant" || msg.Role == "tool" {
			llmMessages = append(llmMessages, msg)
		}
	}
	return llmMessages
}

// UpdateThinkingLevel updates the thinking level for this context.
func (c *Context) UpdateThinkingLevel(level string) {
	c.ThinkingLevel = level
	c.LastUpdate = time.Now()
}

// UpdateSessionID changes the session identifier.
func (c *Context) UpdateSessionID(id string) {
	c.SessionID = id
	c.LastUpdate = time.Now()
}

// StartStreaming marks the context as streaming and records the start time.
func (c *Context) StartStreaming(callID string) {
	c.IsStreaming = true
	c.StreamingModel = &StreamingModel{
		CallID:     callID,
		StartTime:  time.Now(),
		TokenCount: 0,
	}
	c.LastUpdate = time.Now()
}

// StopStreaming marks the context as not streaming and records token count.
func (c *Context) StopStreaming(tokenCount int) {
	c.IsStreaming = false
	if c.StreamingModel != nil {
		c.StreamingModel.TokenCount = tokenCount
		c.StreamingModel = nil
	}
	c.StreamingModel = nil
	c.LastUpdate = time.Now()
}

// IncrementTokenCount increments the token counter.
func (c *Context) IncrementTokenCount() {
	if c.StreamingModel != nil {
		c.StreamingModel.TokenCount++
		c.LastUpdate = time.Now()
	}
}

// ContextSnapshot creates a snapshot of the context for streaming.
// This is used by the AI client to maintain state during streaming.
func (c *Context) ContextSnapshot() *Context {
	return c.Clone()
}

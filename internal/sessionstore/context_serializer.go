// Package sessionstore provides serialization/deserialization for incremental context.
// This package handles converting context.Context to/from JSON for session persistence.
//
// The context.Context itself cannot be serialized (it contains a stdlib context.Context),
// so we serialize the state that can be reconstructed later.
package sessionstore

import (
	"time"

	"github.com/icedream/werkler/internal/sessioncontext"
)

// ContextJSON represents a JSON-serializable version of sessioncontext.Context.
type ContextJSON struct {
	SystemPrompt  string               `json:"system_prompt"`
	Messages      []MessageJSON        `json:"messages"`
	Tools         []ToolDefinitionJSON `json:"tools"`
	Model         ModelInfoJSON        `json:"model"`
	ThinkingLevel string               `json:"thinking_level"`
	LastUpdate    time.Time            `json:"last_update"`
}

// MessageJSON represents a JSON-serializable version of sessioncontext.Message.
type MessageJSON struct {
	Role       string          `json:"role"`
	Content    string          `json:"content"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCallJSON  `json:"tool_calls,omitempty"`
	Parts      []ImagePartJSON `json:"parts,omitempty"`
}

// ToolCallJSON represents a JSON-serializable version of sessioncontext.ToolCall.
type ToolCallJSON struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// ImagePartJSON represents a JSON-serializable version of sessioncontext.ImagePart.
type ImagePartJSON struct {
	URL      string `json:"url"`
	Data     string `json:"data"` // Base64 encoded
	Name     string `json:"name"`
	MIMEType string `json:"mime_type"`
}

// ToolDefinitionJSON represents a JSON-serializable version of sessioncontext.ToolDefinition.
type ToolDefinitionJSON struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// ModelInfoJSON represents a JSON-serializable version of sessioncontext.ModelInfo.
type ModelInfoJSON struct {
	Model    string `json:"model"`
	Provider string `json:"provider"`
	Input    int    `json:"input"`
	Output   int    `json:"output"`
	Context  int    `json:"context"` // MaxTokens
	Cost     string `json:"cost"`    // Formatted string
}

// ToJSON converts a sessioncontext.Context to JSON.
func ToJSON(ctx *sessioncontext.Context) (*ContextJSON, error) {
	if ctx == nil {
		return nil, nil
	}

	// Convert messages
	messagesJSON := make([]MessageJSON, len(ctx.Messages))
	for i, msg := range ctx.Messages {
		messagesJSON[i] = MessageJSON{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCallID: msg.ToolCallID,
			ToolCalls:  convertToolCallsToJSON(msg.ToolCalls),
			Parts:      convertImagePartsToJSON(msg.Parts),
		}
	}

	// Convert tools
	toolsJSON := make([]ToolDefinitionJSON, len(ctx.Tools))
	for i, tool := range ctx.Tools {
		toolsJSON[i] = ToolDefinitionJSON{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: convertInputSchema(tool.InputSchema),
		}
	}

	// Convert model info
	modelJSON := ModelInfoJSON{
		Model:    ctx.Model.Model,
		Provider: ctx.Model.Provider,
		Input:    ctx.Model.Input,
		Output:   ctx.Model.Output,
		Context:  ctx.Model.Context.MaxTokens,
		Cost:     formatCost(ctx.Model.Cost),
	}

	return &ContextJSON{
		SystemPrompt:  ctx.SystemPrompt,
		Messages:      messagesJSON,
		Tools:         toolsJSON,
		Model:         modelJSON,
		ThinkingLevel: ctx.ThinkingLevel,
		LastUpdate:    ctx.LastUpdate,
	}, nil
}

// FromJSON converts JSON to a sessioncontext.Context.
func FromJSON(jsonData *ContextJSON) (*sessioncontext.Context, error) {
	if jsonData == nil {
		return nil, nil
	}

	// Convert messages
	messages := make([]sessioncontext.Message, len(jsonData.Messages))
	for i, msg := range jsonData.Messages {
		messages[i] = sessioncontext.Message{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCallID: msg.ToolCallID,
			ToolCalls:  convertToolCallsFromJSON(msg.ToolCalls),
			Parts:      convertImagePartsFromJSON(msg.Parts),
		}
	}

	// Convert tools
	tools := make([]sessioncontext.ToolDefinition, len(jsonData.Tools))
	for i, tool := range jsonData.Tools {
		tools[i] = sessioncontext.ToolDefinition{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		}
	}

	// Convert model info
	model := sessioncontext.ModelInfo{
		Model:    jsonData.Model.Model,
		Provider: jsonData.Model.Provider,
		Input:    jsonData.Model.Input,
		Output:   jsonData.Model.Output,
		Context:  sessioncontext.ContextWindow{MaxTokens: jsonData.Model.Context},
		Cost:     unmarshalCost(jsonData.Model.Cost),
	}

	return sessioncontext.NewContext(
		jsonData.SystemPrompt,
		messages,
		tools,
		model,
	), nil
}

// convertToolCallsToJSON converts sessioncontext.ToolCall to ToolCallJSON.
func convertToolCallsToJSON(toolCalls []sessioncontext.ToolCall) []ToolCallJSON {
	if toolCalls == nil {
		return nil
	}
	result := make([]ToolCallJSON, len(toolCalls))
	for i, tc := range toolCalls {
		result[i] = ToolCallJSON{
			ID:        tc.ID,
			Name:      tc.Name,
			Arguments: tc.Arguments,
		}
	}
	return result
}

// convertToolCallsFromJSON converts ToolCallJSON to sessioncontext.ToolCall.
func convertToolCallsFromJSON(toolCallsJSON []ToolCallJSON) []sessioncontext.ToolCall {
	if toolCallsJSON == nil {
		return nil
	}
	result := make([]sessioncontext.ToolCall, len(toolCallsJSON))
	for i, tc := range toolCallsJSON {
		result[i] = sessioncontext.ToolCall{
			ID:        tc.ID,
			Name:      tc.Name,
			Arguments: tc.Arguments,
		}
	}
	return result
}

// convertImagePartsToJSON converts sessioncontext.ImagePart to ImagePartJSON.
func convertImagePartsToJSON(imageParts []sessioncontext.ImagePart) []ImagePartJSON {
	if imageParts == nil {
		return nil
	}
	result := make([]ImagePartJSON, len(imageParts))
	for i, ip := range imageParts {
		result[i] = ImagePartJSON{
			URL:      ip.URL,
			Data:     string(ip.Data), // Convert []byte to string
			Name:     ip.Name,
			MIMEType: ip.MIMEType,
		}
	}
	return result
}

// convertImagePartsFromJSON converts ImagePartJSON to sessioncontext.ImagePart.
func convertImagePartsFromJSON(imagePartsJSON []ImagePartJSON) []sessioncontext.ImagePart {
	if imagePartsJSON == nil {
		return nil
	}
	result := make([]sessioncontext.ImagePart, len(imagePartsJSON))
	for i, ip := range imagePartsJSON {
		result[i] = sessioncontext.ImagePart{
			URL:      ip.URL,
			Data:     []byte(ip.Data), // Convert string to []byte
			Name:     ip.Name,
			MIMEType: ip.MIMEType,
		}
	}
	return result
}

// convertInputSchema converts interface{} to map[string]any for JSON serialization.
func convertInputSchema(schema any) map[string]any {
	if schema == nil {
		return nil
	}
	// Try to assert as map[string]any
	if m, ok := schema.(map[string]any); ok {
		return m
	}
	// Try to assert as map[string]interface{}
	if m, ok := schema.(map[string]interface{}); ok {
		result := make(map[string]any, len(m))
		for k, v := range m {
			result[k] = v
		}
		return result
	}
	return nil
}

// formatCost converts Cost struct to a formatted string.
func formatCost(cost sessioncontext.Cost) string {
	// Format as JSON-like string
	// For now, just return a placeholder
	return "0.00"
}

// unmarshalCost converts a formatted string back to Cost struct.
func unmarshalCost(s string) sessioncontext.Cost {
	// For now, return zero values
	// Actual parsing would depend on the format used in formatCost
	return sessioncontext.Cost{}
}

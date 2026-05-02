// Package sessionstore tests for context serialization.
package sessionstore

import (
	"testing"
	"time"

	"github.com/icedream/werkler/internal/sessioncontext"
)

// TestToJSON tests that ToJSON correctly converts context to JSON.
func TestToJSON(t *testing.T) {
	ctx := sessioncontext.NewContext(
		"You are a helpful assistant.",
		[]sessioncontext.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
		},
		[]sessioncontext.ToolDefinition{
			{Name: "test", Description: "A test tool", InputSchema: map[string]any{"param": "string"}},
		},
		sessioncontext.ModelInfo{
			Model:    "gpt-4",
			Provider: "openai",
			Input:    100,
			Output:   200,
			Context:  sessioncontext.ContextWindow{MaxTokens: 4000},
			Cost:     sessioncontext.Cost{Input: 0.01, Output: 0.02, Total: 0.03},
		},
	)

	jsonData, err := ToJSON(ctx)
	if err != nil {
		t.Fatalf("ToJSON failed: %v", err)
	}

	if jsonData == nil {
		t.Fatal("Expected non-nil JSON data")
	}

	if jsonData.SystemPrompt != "You are a helpful assistant." {
		t.Errorf("Expected SystemPrompt 'You are a helpful assistant.', got '%s'", jsonData.SystemPrompt)
	}

	if len(jsonData.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(jsonData.Messages))
	}

	if len(jsonData.Tools) != 1 {
		t.Errorf("Expected 1 tool, got %d", len(jsonData.Tools))
	}

	if jsonData.Model.Model != "gpt-4" {
		t.Errorf("Expected Model 'gpt-4', got '%s'", jsonData.Model.Model)
	}
}

// TestFromJSON tests that FromJSON correctly converts JSON to context.
func TestFromJSON(t *testing.T) {
	jsonData := &ContextJSON{
		SystemPrompt: "You are a test assistant.",
		Messages: []MessageJSON{
			{Role: "user", Content: "Test message"},
			{Role: "assistant", Content: "Test response"},
		},
		Tools: []ToolDefinitionJSON{
			{Name: "tool1", Description: "Tool 1", InputSchema: map[string]any{"param": "string"}},
		},
		Model: ModelInfoJSON{
			Model:    "gpt-3.5",
			Provider: "openai",
			Input:    50,
			Output:   100,
			Context:  2048,
			Cost:     "0.00",
		},
		ThinkingLevel: "off",
		LastUpdate:    time.Now(),
	}

	ctx, err := FromJSON(jsonData)
	if err != nil {
		t.Fatalf("FromJSON failed: %v", err)
	}

	if ctx == nil {
		t.Fatal("Expected non-nil context")
	}

	if ctx.SystemPrompt != "You are a test assistant." {
		t.Errorf("Expected SystemPrompt 'You are a test assistant.', got '%s'", ctx.SystemPrompt)
	}

	if len(ctx.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(ctx.Messages))
	}

	if len(ctx.Tools) != 1 {
		t.Errorf("Expected 1 tool, got %d", len(ctx.Tools))
	}

	if ctx.Model.Model != "gpt-3.5" {
		t.Errorf("Expected Model 'gpt-3.5', got '%s'", ctx.Model.Model)
	}
}

// TestToJSONNil tests that ToJSON handles nil context.
func TestToJSONNil(t *testing.T) {
	jsonData, err := ToJSON(nil)
	if err != nil {
		t.Fatalf("ToJSON(nil) failed: %v", err)
	}

	if jsonData != nil {
		t.Error("Expected nil JSON data for nil context")
	}
}

// TestFromJSONNil tests that FromJSON handles nil JSON data.
func TestFromJSONNil(t *testing.T) {
	ctx, err := FromJSON(nil)
	if err != nil {
		t.Fatalf("FromJSON(nil) failed: %v", err)
	}

	if ctx != nil {
		t.Error("Expected nil context for nil JSON data")
	}
}

// TestImagePartConversion tests ImagePart conversion.
func TestImagePartConversion(t *testing.T) {
	imageData := []byte("test image data")
	imagePart := sessioncontext.ImagePart{
		URL:      "http://example.com/image.png",
		Data:     imageData,
		Name:     "test.png",
		MIMEType: "image/png",
	}

	jsonPart := ImagePartJSON{
		URL:      imagePart.URL,
		Data:     string(imagePart.Data),
		Name:     imagePart.Name,
		MIMEType: imagePart.MIMEType,
	}

	reconstructed := sessioncontext.ImagePart{
		URL:      jsonPart.URL,
		Data:     []byte(jsonPart.Data),
		Name:     jsonPart.Name,
		MIMEType: jsonPart.MIMEType,
	}

	if reconstructed.URL != imagePart.URL {
		t.Errorf("URL mismatch")
	}
	if string(reconstructed.Data) != string(imagePart.Data) {
		t.Errorf("Data mismatch")
	}
	if reconstructed.Name != imagePart.Name {
		t.Errorf("Name mismatch")
	}
	if reconstructed.MIMEType != imagePart.MIMEType {
		t.Errorf("MIMEType mismatch")
	}
}

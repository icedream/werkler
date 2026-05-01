package ai

import (
	"strings"
	"testing"
)

func TestCountTokens_KnownModel(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello!"},
		{Role: "assistant", Content: "Hi there, how can I help?"},
	}
	tc, err := CountTokens("gpt-4o", msgs)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if tc.Approx {
		t.Error("gpt-4o should use exact tokenizer, not approximate")
	}
	if tc.Total <= 0 {
		t.Errorf("expected positive token count, got %d", tc.Total)
	}
}

func TestCountTokens_UnknownModel_Approx(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "Tell me a joke."},
	}
	tc, err := CountTokens("some-unknown-llm-model", msgs)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if !tc.Approx {
		t.Error("unknown model should use approximate tokenizer")
	}
	if tc.Total <= 0 {
		t.Errorf("expected positive token count, got %d", tc.Total)
	}
}

func TestCountTokens_ToolCalls(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "Run a build."},
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					ID:        "call_1",
					Name:      "process_start",
					Arguments: map[string]any{"command": "go", "args": []any{"build", "./..."}},
				},
			},
		},
		{Role: "tool", Content: "exit code 0", ToolCallID: "call_1"},
	}
	tc, err := CountTokens("gpt-4o", msgs)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if tc.Total <= 0 {
		t.Errorf("expected positive token count, got %d", tc.Total)
	}
}

func TestTokenCount_Format(t *testing.T) {
	cases := []struct {
		tc   TokenCount
		want string
	}{
		{TokenCount{Total: 512}, "512"},
		{TokenCount{Total: 4200}, "4.2k"},
		{TokenCount{Total: 128000}, "128.0k"},
		{TokenCount{Total: 1200000}, "1.2M"},
		{TokenCount{Total: 500, Approx: true}, "~500"},
		{TokenCount{Total: 3000, Approx: true}, "~3.0k"},
	}
	for _, tc := range cases {
		got := tc.tc.Format()
		if got != tc.want {
			t.Errorf("Format() = %q, want %q (total=%d approx=%v)", got, tc.want, tc.tc.Total, tc.tc.Approx)
		}
	}
}

func TestTokenCount_FormatWithMax(t *testing.T) {
	exact := TokenCount{Total: 40000}
	got := exact.FormatWithMax(128000)
	if !strings.Contains(got, "31%") {
		t.Errorf("FormatWithMax = %q, expected to contain 31%%", got)
	}
	if !strings.Contains(got, "40.0k") {
		t.Errorf("FormatWithMax = %q, expected to contain 40.0k", got)
	}

	// Approximate tokenizer: should NOT show percentage.
	approx := TokenCount{Total: 40000, Approx: true}
	got2 := approx.FormatWithMax(128000)
	if strings.Contains(got2, "%") {
		t.Errorf("FormatWithMax with Approx should not contain '%%', got %q", got2)
	}

	// No max: just the count.
	got3 := exact.FormatWithMax(0)
	if strings.Contains(got3, "/") {
		t.Errorf("FormatWithMax with maxTokens=0 should not contain '/', got %q", got3)
	}
}

func TestTokenCount_UsageFraction(t *testing.T) {
	exact := TokenCount{Total: 64000}
	f := exact.UsageFraction(128000)
	if f < 0.49 || f > 0.51 {
		t.Errorf("UsageFraction = %f, want ~0.5", f)
	}

	// Approximate: always -1.
	approx := TokenCount{Total: 64000, Approx: true}
	if approx.UsageFraction(128000) != -1 {
		t.Errorf("approximate UsageFraction should return -1")
	}

	// No max: -1.
	if exact.UsageFraction(0) != -1 {
		t.Errorf("UsageFraction with maxTokens=0 should return -1")
	}

	// Overflow capped at 1.
	over := TokenCount{Total: 200000}
	if over.UsageFraction(128000) != 1.0 {
		t.Errorf("overflow UsageFraction should be 1.0, got %f", over.UsageFraction(128000))
	}
}

func TestCountTokensWithTools_GreaterThanWithout(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "Hello!"},
	}
	tools := []ToolDefinition{
		{
			Name:        "process_start",
			Description: "Start a subprocess.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{"type": "string"},
				},
			},
		},
	}

	base, err := CountTokens("gpt-4o", msgs)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	withTools, err := CountTokensWithTools("gpt-4o", msgs, tools)
	if err != nil {
		t.Fatalf("CountTokensWithTools: %v", err)
	}
	if withTools.Total <= base.Total {
		t.Errorf("expected CountTokensWithTools (%d) > CountTokens (%d)", withTools.Total, base.Total)
	}
}

func TestCountTokensWithTools_NilMessages_ReturnsToolOverhead(t *testing.T) {
	tools := []ToolDefinition{
		{
			Name:        "file_read",
			Description: "Read a file from disk.",
		},
	}
	count, err := CountTokensWithTools("gpt-4o", nil, tools)
	if err != nil {
		t.Fatalf("CountTokensWithTools: %v", err)
	}
	// Should return at least the reply-priming constant (3) plus some tokens for the tool schema.
	if count.Total <= 3 {
		t.Errorf("expected tool overhead > 3, got %d", count.Total)
	}
}

func TestCountTokensWithTools_InputSchema_Contributes(t *testing.T) {
	toolWithoutSchema := ToolDefinition{
		Name:        "my_tool",
		Description: "Does something.",
	}
	toolWithSchema := ToolDefinition{
		Name:        "my_tool",
		Description: "Does something.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"very_long_parameter_name": map[string]any{
					"type":        "string",
					"description": "A parameter with a deliberately verbose description to add tokens.",
				},
			},
		},
	}

	without, err := CountTokensWithTools("gpt-4o", nil, []ToolDefinition{toolWithoutSchema})
	if err != nil {
		t.Fatalf("CountTokensWithTools (no schema): %v", err)
	}
	with, err := CountTokensWithTools("gpt-4o", nil, []ToolDefinition{toolWithSchema})
	if err != nil {
		t.Fatalf("CountTokensWithTools (with schema): %v", err)
	}
	if with.Total <= without.Total {
		t.Errorf("InputSchema should add tokens: with=%d, without=%d", with.Total, without.Total)
	}
}

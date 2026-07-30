package ui

import (
	"testing"

	"github.com/icedream/werkler/internal/ai"
)

func TestIsContextOverflowError(t *testing.T) {
	tests := []struct {
		name   string
		msg    string
		wantOk bool
	}{
		// --- Overflow matches ---
		{
			name:   "llama.cpp standard message",
			msg:    "request (133595 tokens) exceeds the available context size (131072 tokens), try increasing it",
			wantOk: true,
		},
		{
			name:   "Anthropic token overflow",
			msg:    "prompt is too long: 213462 tokens > 200000 maximum",
			wantOk: true,
		},
		{
			name:   "Anthropic request_too_large",
			msg:    "413 {\"error\":{\"type\":\"request_too_large\",\"message\":\"Request exceeds the maximum size\"}}",
			wantOk: true,
		},
		{
			name:   "OpenAI exceeds context window",
			msg:    "Your input exceeds the context window of this model",
			wantOk: true,
		},
		{
			name:   "OpenAI exceeds maximum context length",
			msg:    "Requested token count exceeds the model's maximum context length of 131072 tokens",
			wantOk: true,
		},
		{
			name:   "Google Gemini input token count",
			msg:    "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
			wantOk: true,
		},
		{
			name:   "xAI maximum prompt length",
			msg:    "This model's maximum prompt length is 131072 but the request contains 537812 tokens",
			wantOk: true,
		},
		{
			name:   "Groq reduce length",
			msg:    "Please reduce the length of the messages or completion",
			wantOk: true,
		},
		{
			name:   "OpenRouter maximum context length",
			msg:    "This endpoint's maximum context length is 131072 tokens. However, you requested about 150000 tokens",
			wantOk: true,
		},
		{
			name:   "GitHub Copilot exceeds limit",
			msg:    "prompt token count of 130000 exceeds the limit of 128000",
			wantOk: true,
		},
		{
			name:   "LM Studio greater than context length",
			msg:    "tokens to keep from the initial prompt is greater than the context length",
			wantOk: true,
		},
		{
			name:   "Mistral too large for model",
			msg:    "Prompt contains 150000 tokens ... too large for model with 128000 maximum context length",
			wantOk: true,
		},
		{
			name:   "DS4 prompt has tokens",
			msg:    "Prompt has 150,000 tokens, but the configured context size is 128,000 tokens",
			wantOk: true,
		},
		{
			name:   "Ollama prompt too long",
			msg:    "prompt too long; exceeded max context length by 5000 tokens",
			wantOk: true,
		},
		{
			name:   "Generic context length exceeded",
			msg:    "context_length exceeded",
			wantOk: true,
		},
		{
			name:   "Generic token limit exceeded",
			msg:    "token limit exceeded",
			wantOk: true,
		},
		{
			name:   "Cerebras 400 no body",
			msg:    "400 status code (no body)",
			wantOk: true,
		},
		{
			name:   "Together AI input longer",
			msg:    "The input (150000 tokens) is longer than the model's context length (128000 tokens).",
			wantOk: true,
		},
		// --- Non-overflow matches (should NOT trigger compaction) ---
		{
			name:   "rate limiting",
			msg:    "rate limit exceeded",
			wantOk: false,
		},
		{
			name:   "429 too many requests",
			msg:    "429 Too Many Requests",
			wantOk: false,
		},
		{
			name:   "unrelated error",
			msg:    "connection refused",
			wantOk: false,
		},
		{
			name:   "400 without context reference",
			msg:    "Bad Request: invalid parameter",
			wantOk: false,
		},
		{
			name:   "Bedrock throttling (non-overflow)",
			msg:    "Throttling error: Too many tokens, please wait before trying again.",
			wantOk: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isContextOverflowError(errf(tt.msg))
			if got != tt.wantOk {
				t.Errorf("isContextOverflowError(%q) = %v, want %v", tt.msg, got, tt.wantOk)
			}
		})
	}
}

func errf(s string) error { return errfErr(s) }

type errfErr string

func (e errfErr) Error() string { return string(e) }

func TestFindSplitPoint(t *testing.T) {
	// Messages: [0:system, 1:u, 2:a, 3:u, 4:a, 5:u, 6:a]
	msgs := []ai.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "u1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "u2"},
		{Role: "assistant", Content: "a2"},
		{Role: "user", Content: "u3"},
		{Role: "assistant", Content: "a3"},
	}

	t.Run("fits_budget", func(t *testing.T) {
		// Very large budget — never exceeds.
		got := findSplitPoint("gpt-4", msgs, 999999, 0)
		if got != -1 {
			t.Errorf("expected -1 (no split), got %d", got)
		}
	})

	t.Run("small_budget", func(t *testing.T) {
		// Small budget — should split early.
		got := findSplitPoint("gpt-4", msgs, 10, 0)
		if got <= 1 {
			t.Errorf("expected split > 1, got %d", got)
		}
	})

	t.Run("tool_tokens_consumes_budget", func(t *testing.T) {
		// Tool tokens consume the budget entirely.
		got := findSplitPoint("gpt-4", msgs, 100, 100)
		if got != 1 {
			t.Errorf("expected 1 (fallback), got %d", got)
		}
	})
}

func TestFindTurnStartIndex(t *testing.T) {
	msgs := []ai.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "u1"},
		{Role: "assistant", Content: "a1"},
		{Role: "user", Content: "u2"},
		{Role: "assistant", Content: "a2"},
	}

	t.Run("finds_turn_start", func(t *testing.T) {
		// Index 3 is the second user message, so turn start for index 4 is 3.
		got := findTurnStartIndex(msgs, 4)
		if got != 3 {
			t.Errorf("expected 3, got %d", got)
		}
	})

	t.Run("no_turn_found", func(t *testing.T) {
		// Index 0 (system) has no turn start.
		got := findTurnStartIndex(msgs, 0)
		if got != -1 {
			t.Errorf("expected -1, got %d", got)
		}
	})
}

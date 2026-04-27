package ai

import (
	"encoding/json"
	"fmt"

	tiktoken "github.com/pkoukk/tiktoken-go"
)

// TokenCount holds the result of a token-count estimation.
type TokenCount struct {
	// Total is the estimated number of input tokens for the message history.
	Total int
	// Approx is true when the encoding is a best-effort fallback (i.e. the
	// model's tokenizer was not known and cl100k_base was used instead).
	Approx bool
}

// Format returns a compact human-readable string like "4.2k" or "~1.1k".
// The "~" prefix is included when Approx is true.
func (tc TokenCount) Format() string {
	prefix := ""
	if tc.Approx {
		prefix = "~"
	}
	return prefix + formatCompactInt(tc.Total)
}

// FormatWithMax returns a string like "4.2k / 128k (3%)" when maxTokens > 0,
// or just Format() when maxTokens is 0. The percentage is omitted for
// approximate counts to avoid misleading the user.
func (tc TokenCount) FormatWithMax(maxTokens int) string {
	base := tc.Format()
	if maxTokens <= 0 || tc.Approx {
		return base
	}
	pct := tc.Total * 100 / maxTokens
	return fmt.Sprintf("%s / %s (%d%%)", base, formatCompactInt(maxTokens), pct)
}

// UsageFraction returns a value in [0.0, 1.0] representing how full the context
// window is, or -1 if the fraction cannot be reliably computed (approximate
// tokenizer or unknown max).
func (tc TokenCount) UsageFraction(maxTokens int) float64 {
	if maxTokens <= 0 || tc.Approx {
		return -1
	}
	f := float64(tc.Total) / float64(maxTokens)
	if f > 1 {
		f = 1
	}
	return f
}

// CountTokens estimates the number of input tokens for messages using the
// tokenizer appropriate for modelName. When modelName is not recognized,
// cl100k_base is used and TokenCount.Approx is set to true.
//
// The estimation follows OpenAI's token-counting heuristic:
//   - 3 tokens of overhead per message (role + formatting)
//   - content tokens
//   - for tool calls: name + JSON-marshalled arguments tokens
//   - 3 tokens reply-priming overhead (for the trailing assistant turn start)
func CountTokens(modelName string, messages []Message) (TokenCount, error) {
	enc, approx, err := encodingForModel(modelName)
	if err != nil {
		return TokenCount{}, fmt.Errorf("getting encoding: %w", err)
	}

	total := 3 // reply-priming (the assistant turn that follows)
	for _, msg := range messages {
		total += 3 // per-message overhead
		if msg.Content != "" {
			total += len(enc.Encode(msg.Content, nil, nil))
		}
		for _, tc := range msg.ToolCalls {
			total += len(enc.Encode(tc.Name, nil, nil))
			if len(tc.Arguments) > 0 {
				raw, _ := json.Marshal(tc.Arguments)
				total += len(enc.Encode(string(raw), nil, nil))
			}
		}
	}

	return TokenCount{Total: total, Approx: approx}, nil
}

// encodingForModel returns the tiktoken encoding for a model name, falling
// back to cl100k_base for unknown models.
func encodingForModel(modelName string) (*tiktoken.Tiktoken, bool, error) {
	enc, err := tiktoken.EncodingForModel(modelName)
	if err == nil {
		return enc, false, nil
	}
	// Unknown model — use cl100k_base as a rough approximation.
	enc, err = tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		return nil, false, fmt.Errorf("loading cl100k_base fallback: %w", err)
	}
	return enc, true, nil
}

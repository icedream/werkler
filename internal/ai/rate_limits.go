package ai

import "fmt"

// RateLimits holds rate limit information returned by a provider with each API
// response. Fields are zero when the provider did not report them (e.g. Ollama).
type RateLimits struct {
	LimitRequests     int
	LimitTokens       int
	RemainingRequests int
	RemainingTokens   int
}

// IsKnown reports whether any rate limit data was returned by the provider.
func (r RateLimits) IsKnown() bool {
	return r.LimitTokens > 0 || r.LimitRequests > 0
}

// FormatTokens returns a compact human-readable string for the remaining token
// count, e.g. "88.3k" or "1.2M". Returns "" when no token limit is known.
func (r RateLimits) FormatTokens() string {
	if r.LimitTokens == 0 {
		return ""
	}
	return formatCount(r.RemainingTokens)
}

// FormatRequests returns a compact string for remaining requests. Returns ""
// when no request limit is known.
func (r RateLimits) FormatRequests() string {
	if r.LimitRequests == 0 {
		return ""
	}
	return formatCount(r.RemainingRequests)
}

func formatCount(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fk", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

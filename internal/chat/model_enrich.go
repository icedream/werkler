package chat

import (
	"fmt"
	"strings"

	"github.com/icedream/werkler/internal/ai"
)

// EnrichSystemPrompt appends a model context section to systemPrompt when
// info contains a usable context window size. Returns systemPrompt unchanged
// when info has no context data.
func EnrichSystemPrompt(systemPrompt string, info ai.ModelInfo) string {
	if !info.HasContext() {
		return systemPrompt
	}
	var sb strings.Builder
	sb.WriteString(systemPrompt)
	fmt.Fprintf(&sb, "\n\n## Model Context\nContext window: %d tokens. Manage conversation length accordingly.", info.Context.MaxTokens)
	return sb.String()
}

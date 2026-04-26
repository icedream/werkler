package chat

import (
	"fmt"

	"github.com/icedream/werkler/internal/ai"
)

// ModelInfoProvider is an interface that can provide model context limits information
// for incorporating into the system prompt.
// This should be implemented by any system that can fetch model information.

type ModelInfoProvider interface {
	GetModelContextLimits(modelName string) (ai.ModelInfo, error)
}

// EnrichSystemPrompt adds model context limit information to the system prompt
// if available from the provided model info provider.
// This helps the AI make context-aware decisions about what to include in prompts.
//
// Example usage:
//   enrichedPrompt := EnrichSystemPrompt(
//       SystemPrompt,
//       toolManager,
//       "gpt-4" // model name
//   )
//
// If model info is available with token limits, this will append information like:
//   "The model has a maximum input token limit of approximately 128000 tokens."
//
func EnrichSystemPrompt(systemPrompt string, provider ModelInfoProvider, modelName string) string {
	if provider == nil {
		return systemPrompt
	}

	info, err := provider.GetModelContextLimits(modelName)
	if err != nil {
		return systemPrompt
	}
	
	if !info.HasContext() && len(info.SupportedFeatures) == 0 {
		return systemPrompt
	}

	enrichedPrompt := systemPrompt
	if info.Context.MaxInputTokens > 0 {
		enrichedPrompt += "\n\n## Model Context Limits\n"
		enrichedPrompt += "The model has a maximum input token limit of approximately "
		enrichedPrompt += fmt.Sprintf("%d tokens.", info.Context.MaxInputTokens)
	}
	if info.Context.MaxOutputTokens > 0 {
		if info.Context.MaxInputTokens < 0 {
			enrichedPrompt += "\n\n## Model Context Limits\n"
		}
		enrichedPrompt += fmt.Sprintf("\nThe model has a maximum output token limit of approximately %d tokens.", info.Context.MaxOutputTokens)
	}
	if len(info.SupportedFeatures) > 0 {
		if info.Context.MaxInputTokens < 0 && info.Context.MaxOutputTokens < 0 {
			enrichedPrompt += "\n\n## Supported Features\n"
		} else {
			enrichedPrompt += "\n\n## Supported Features\n"
		}
		for _, feature := range info.SupportedFeatures {
			enrichedPrompt += " - " + feature + "\n"
		}
	}
	return enrichedPrompt
}

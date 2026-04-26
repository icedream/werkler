package ai

import "context"

// ModelInfo contains metadata about an AI model's capabilities.
type ModelInfo struct {
	// Model is the model name or identifier.
	Model string

	// Context holds the model's token window limits.
	Context struct {
		// MaxTokens is the effective context window size (tokens).
		// For Ollama, this is the active num_ctx value (overrides the
		// architectural default when set in the Modelfile or at runtime).
		MaxTokens int
	}
}

// HasContext reports whether ModelInfo contains a usable context limit.
func (m ModelInfo) HasContext() bool {
	return m.Context.MaxTokens > 0
}

// ModelInfoGetter is an optional interface implemented by AI clients that can
// fetch live model metadata (e.g. context window size) from the provider.
type ModelInfoGetter interface {
	GetModelInfo(ctx context.Context) (ModelInfo, error)
}

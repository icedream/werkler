package ai

// ModelInfo contains metadata about an AI model's capabilities and limitations.
type ModelInfo struct {
	// Model name or identifier
	Model string
	
	// Context represents the model's token context window
	Context struct {
		// MaxInputTokens is the maximum number of tokens allowed in the input/prompt
		MaxInputTokens int
		
		// MaxOutputTokens is the maximum number of tokens allowed in the output/completion
		MaxOutputTokens int
	}
	
	// SupportedFeatures lists features the model supports
	SupportedFeatures []string
}

// HasContext returns true if the model has defined context limits.
func (m ModelInfo) HasContext() bool {
	return m.Context.MaxInputTokens > 0 || m.Context.MaxOutputTokens > 0
}

// ModelInfoProvider is an optional interface that AI clients can implement
// to expose model info for better context-aware prompting.
type ModelInfoProvider interface {
	GetModelInfo() ModelInfo
}

// ModelInfoRetriever can be used by external systems to get model info.
// This allows MCP servers or other extensions to provide model-specific metadata.
type ModelInfoRetriever interface {
	// GetModelContextLimits retrieves model context limits for the given model name.
	// Returns empty ModelInfo if no limits are available.
	GetModelContextLimits(modelName string) (ModelInfo, error)
}

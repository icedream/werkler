package tools

import (
	"github.com/icedream/werkler/internal/ai"
)

// Add a model info retriever to the Manager
type modelInfoRetriever interface {
	GetModelContextLimits(modelName string) (ai.ModelInfo, error)
}

// GetModelContextLimits forwards to the wrapped MCP manager if it supports model info retrieval.
// Returns empty ModelInfo if not supported or no limits available.
func (m *Manager) GetModelContextLimits(modelName string) (ai.ModelInfo, error) {
	if o, ok := m.wrapped.(modelInfoRetriever); ok {
		return o.GetModelContextLimits(modelName)
	}
	return ai.ModelInfo{}, nil
}

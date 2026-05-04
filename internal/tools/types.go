package tools

import (
	"context"

	"github.com/icedream/werkler/internal/ai"
)

// Builtin pairs a tool definition with its handler function.
// Exported so subpackages can construct values without depending on the
// private builtin type in manager.go.
type Builtin struct {
	Def             ai.ToolDefinition
	Handle          func(ctx context.Context, args map[string]any) (string, error)
	HandleWithParts func(ctx context.Context, args map[string]any) (string, []ai.ImagePart, error) // optional; used when a tool returns image data
}

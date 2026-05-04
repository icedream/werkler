package toolutil

import (
	"context"

	"github.com/icedream/werkler/internal/ai"
)

// Builtin pairs a tool definition with its handler.
// Used by subpackages to return their tool sets to the Manager.
type Builtin struct {
	Def             ai.ToolDefinition
	Handle          func(ctx context.Context, args map[string]any) (string, error)
	HandleWithParts func(ctx context.Context, args map[string]any) (string, []ai.ImagePart, error)
}

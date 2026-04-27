package chat

import (
	"fmt"
	"strings"

	"github.com/icedream/werkler/internal/ai"
)

// EnrichSystemPromptWorkspace appends a session-workspace section to systemPrompt.
// The workspace directory is a per-session folder the AI can use for plan files,
// notes, and other session-scoped scratch work.
func EnrichSystemPromptWorkspace(systemPrompt, dir string) string {
	return systemPrompt + "\n\n## Session workspace\nYour session workspace directory: `" + dir + "`\nWrite plan files (e.g. `plan.md`) and other working documents here. Do NOT write plans to memory — use this directory instead."
}

// EnrichSystemPromptCWD appends a working-directory section to systemPrompt.
// This tells the AI its starting directory so it can use "." in file and
// process operations rather than guessing absolute paths.
func EnrichSystemPromptCWD(systemPrompt, cwd string) string {
	return systemPrompt + "\n\n## Working directory\nCurrent working directory: `" + cwd + "`\nUse `.` (or relative paths) in file operations and process_start calls to refer to this directory."
}

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

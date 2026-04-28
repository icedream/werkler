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

// EnrichSystemPromptReasoningTools appends guidance about when to use the
// enable_reasoning and think tools. Only call this when reasoning is enabled
// (i.e. disable_reasoning is false for the active provider).
func EnrichSystemPromptReasoningTools(systemPrompt string) string {
	return systemPrompt + `

## Reasoning tools
Two tools are available to apply deeper thinking when needed:

- **enable_reasoning** — use when the *entire* upcoming response needs deep analytical reasoning: complex architectural decisions, subtle bug root-cause analysis, hard algorithmic tradeoffs, or multi-step reasoning where small errors compound. Call it as your *sole* action with no other tools or response text. werkler immediately re-prompts you with enhanced reasoning enabled. Reasoning returns to the default level after that single turn.

- **think(question)** — use for a focused sub-question *within* a response: picking an algorithm, evaluating a tradeoff, verifying correctness. It invokes the model's native reasoning/thinking token pathway for a targeted sub-call and returns the analysis inline. The thinking is not stored in the conversation history.

Only use these tools when genuinely needed — routine coding, simple lookups, and straightforward tasks do not benefit, and both add latency.`
}

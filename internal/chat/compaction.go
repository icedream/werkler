package chat

// CompactionSystemPrompt is the system prompt sent to the AI when compacting
// the conversation history. Exported so modeltest can exercise it against a
// live model without importing internal/ui.
const CompactionSystemPrompt = "You are a conversation summarizer. " +
	"Write a concise but information-dense summary of the conversation transcript below. " +
	"You MUST preserve: the main objective and current status, " +
	"key decisions and rationale, ALL file paths created or modified (exact paths), " +
	"ALL tool calls with their key arguments and outcomes, " +
	"ALL unresolved errors and open items, and the last clear user intent. " +
	"Write in past tense. Only verifiable facts — no opinion. " +
	"Note: reasoning/thinking tokens are excluded from this transcript."

// CompactionUserMessage returns the user-turn content for a compaction request,
// wrapping the pre-built transcript string.
func CompactionUserMessage(transcript string) string {
	return "Summarize this conversation (incorporating the prior summary if present):\n\n" + transcript
}

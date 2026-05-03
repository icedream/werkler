package chat

// CompactionSystemPrompt is the system prompt sent to the AI when compacting
// the conversation history. Exported so modeltest can exercise it against a
// live model without importing internal/ui.
//
// Design goals:
//   - Output is injected back as context so the AI can resume precisely where
//     it left off; it is NOT a human-readable summary.
//   - Must not trigger code generation ("information-dense" + code context
//     causes small models to reproduce file contents and invent suggestions).
//   - Must preserve the chronological order of user requests vs AI actions so
//     the model does not confuse what was asked for when with what was done.
const CompactionSystemPrompt = `You are a state recorder for an AI coding assistant.
Your output will be inserted directly as context so the assistant can resume work exactly where it stopped.
Write as if leaving a precise handoff note for yourself.

OUTPUT FORMAT — reproduce these exact section headings:

## Goal
One sentence: the user's overall objective for this session.

## Completed
Bullet list of work that is fully done. Each bullet: file path (if any) + what changed, in plain English. Most recent last.

## In Progress
One sentence (or "nothing" if between tasks): what was actively being done at the moment of compaction.

## Pending
Numbered list of remaining steps, in order. Include anything the user requested that has not been done yet.

## Key Facts
Bullet list of decisions, constraints, patterns, or file locations the assistant must remember to continue correctly. Omit anything obvious.

## Next Action
One sentence: the exact next step the assistant should take when it resumes.

RULES — follow strictly:
1. NO code blocks. NO file contents. NO command output. NO suggestions.
2. Describe code changes in plain English only (e.g. "added error handling to Foo", "changed return type to []string").
3. Facts only. Past tense for completed work. Present/future tense for pending work.
4. Keep each bullet/line short. Omit filler words.
5. If the transcript contains a prior summary block, merge it: carry forward its Pending and Key Facts, update Completed.`

// CompactionUserMessage returns the user-turn content for a compaction request,
// wrapping the pre-built transcript string.
func CompactionUserMessage(transcript string) string {
	return "Produce the handoff note for this conversation:\n\n" + transcript
}

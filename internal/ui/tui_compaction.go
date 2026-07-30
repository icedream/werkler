package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
)

// compactBudgetFraction is the fraction of maxTokens reserved for the
// compaction request itself.  We want the compaction transcript to fit
// comfortably inside the context window so the summary call never
// itself overflows.
const compactBudgetFraction = 0.7

// findSplitPoint walks backward through messages, accumulating token counts,
// and returns the index where the "keep" portion should begin.
// Returns 1 (after system message) if no suitable split point is found.
func findSplitPoint(modelName string, messages []ai.Message, budget int, toolTokens int) int {
	// Subtract tool-schema overhead from the budget.
	available := budget - toolTokens
	if available <= 0 {
		return 1
	}

	// Walk backward, accumulating tokens.
	var accumulated int
	for i := len(messages) - 1; i >= 1; i-- {
		msg := messages[i]
		if msg.Role == "system" {
			continue
		}
		tokens, err := ai.CountTokens(modelName, []ai.Message{msg})
		if err != nil {
			// Fallback: estimate 100 tokens per message.
			tokens.Total = 100
		}
		accumulated += tokens.Total

		// If we've exceeded the budget, the split point is the previous message.
		if accumulated > available {
			// Find a valid cut point: user message or assistant message.
			for j := i; j >= 1; j-- {
				if messages[j].Role == "user" || messages[j].Role == "assistant" {
					return j + 1
				}
			}
			return 1
		}
	}
	// Budget not exceeded — keep everything.
	return -1
}

// findTurnStartIndex finds the user message that started the turn containing
// the message at index `idx` (walking backward).  Returns -1 if none found.
func findTurnStartIndex(messages []ai.Message, idx int) int {
	for i := idx - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return i
		}
	}
	return -1
}

// doCompact sends the current message history to the AI as a summarization
// request and returns a compactDoneMsg with the resulting summary text.
// modelName, toolTokens and maxTokens are used to trim the kept-turn count
// inside the goroutine so the resulting message set fits inside the context.
// computeCompactionParams derives keepTurns, totalTokens, and a warning string
// from the generated summary and the original message snapshot.  It is called
// after the summarisation stream finishes.
func computeCompactionParams(s string, snap []ai.Message, modelName string, toolTokens, maxTokens int) (keepTurns, totalTokens int, warning string) {
	// Start conservative: keep at most 1 recent turn verbatim.  The goal is
	// NOT to fill context up to the compaction threshold (that would trigger
	// another compaction almost immediately); instead we aim for a small
	// post-compaction size — around 20 % of the window — so the AI has
	// plenty of headroom to continue working.
	keepTurns = 1

	var newSysContent string
	if len(snap) > 0 && snap[0].Role == "system" {
		newSysContent = replaceSummaryBlock(snap[0].Content, s)
	} else {
		newSysContent = "## Summary of previous conversation\n\n" + s
	}
	newSysTok, _ := ai.CountTokens(modelName, []ai.Message{{Role: "system", Content: newSysContent}})
	baseTok := newSysTok.Total + toolTokens

	estimateTotal := func(k int) int {
		est := baseTok
		if k > 0 {
			if tc, err := ai.CountTokens(modelName, extractLastTurns(snap, k)); err == nil {
				est += tc.Total
			}
		}
		return est
	}

	totalTokens = estimateTotal(keepTurns)
	if maxTokens > 0 {
		// Target ~30 % of the context window post-compaction.  If even 1 turn
		// exceeds that, drop to 0 (summary only).  Never fill to the 65 %
		// compaction threshold — that would guarantee an immediate re-trigger.
		const targetFraction = 0.30
		limit := int(float32(maxTokens) * targetFraction)
		if totalTokens > limit {
			keepTurns = 0
			totalTokens = estimateTotal(keepTurns)
		}
		// Sanity-check: warn if even the summary alone is already over the
		// compaction threshold (the model context window is very small or the
		// summary is unexpectedly large).
		const compactThreshold = 0.65
		if totalTokens > int(float32(maxTokens)*compactThreshold) {
			warning = "Context is very full even after compaction — " +
				"consider switching to a model with a larger context window."
		}
	}
	return
}

func doCompact(ctx context.Context, client ai.StreamCompleter, messages []ai.Message, modelName string, toolTokens, maxTokens int) tea.Cmd {
	// Build the transcript from a snapshot, so the goroutine doesn't race.
	snap := make([]ai.Message, len(messages))
	copy(snap, messages)

	// Estimate the token size of the full transcript.  If it would exceed
	// the compaction budget (typically the model's context window), split
	// the transcript into older + recent portions so the compaction call
	// itself never overflows.
	transcript := buildCompactTranscript(snap)
	transcriptTokens, _ := ai.CountTokens(modelName, []ai.Message{{Role: "user", Content: transcript}})

	var splitAt int
	var turnPrefixMessages []ai.Message
	var summaryPrompt string

	budget := int(float32(maxTokens)*compactBudgetFraction) - toolTokens
	if budget <= 0 {
		budget = maxTokens / 2 // fallback
	}

	if transcriptTokens.Total > budget {
		// Split: keep recent messages verbatim, summarize the older ones.
		splitAt = findSplitPoint(modelName, snap, budget, toolTokens)
		if splitAt < 0 {
			splitAt = 1 // keep everything after system message
		}

		// Find the user message that started the current turn (the one at
		// or just before the split point).  We'll generate a prefix summary
		// for the older part of this turn.
		turnStart := findTurnStartIndex(snap, splitAt)
		if turnStart >= 0 && turnStart < splitAt {
			// The split is mid-turn: collect the prefix messages.
			turnPrefixMessages = make([]ai.Message, 0, splitAt-turnStart)
			for _, m := range snap[turnStart:splitAt] {
				if m.Role != "system" {
					turnPrefixMessages = append(turnPrefixMessages, m)
				}
			}
		}

		// Build the older-only transcript.
		older := make([]ai.Message, 0, splitAt)
		for _, m := range snap[:splitAt] {
			if m.Role != "system" {
				older = append(older, m)
			}
		}
		if len(older) > 0 {
			summaryPrompt = chat.CompactionUserMessage(buildCompactTranscript(older))
		} else {
			summaryPrompt = chat.CompactionUserMessage("(no older messages to summarize)")
		}
	}

	return func() tea.Msg {
		var summary string
		var turnPrefixSummary string

		if splitAt >= 0 {
			// Build messages for the older-only summary.
			olderSystem := chat.CompactionOldPrompt
			olderMessages := []ai.Message{
				{Role: "system", Content: olderSystem},
				{Role: "user", Content: summaryPrompt},
			}
			ch := client.CompleteStream(ctx, olderMessages, nil)
			summary = readStream(ctx, ch)

			// If the split is mid-turn, generate a prefix summary.
			if len(turnPrefixMessages) > 0 {
				prefixMessages := []ai.Message{
					{Role: "system", Content: chat.CompactionTurnPrefixPrompt},
					{Role: "user", Content: chat.CompactionUserMessage(buildCompactTranscript(turnPrefixMessages))},
				}
				ch2 := client.CompleteStream(ctx, prefixMessages, nil)
				turnPrefixSummary = readStream(ctx, ch2)
			}
		} else {
			// No split needed — use the full transcript.
			summaryMessages := []ai.Message{
				{Role: "system", Content: chat.CompactionSystemPrompt},
				{Role: "user", Content: chat.CompactionUserMessage(transcript)},
			}
			ch := client.CompleteStream(ctx, summaryMessages, nil)
			summary = readStream(ctx, ch)
		}

		return compactDoneMsg{
			summary:           summary,
			splitAt:           splitAt,
			turnPrefixSummary: turnPrefixSummary,
		}
	}
}

// readStream drains a streaming channel and returns the concatenated content.
func readStream(ctx context.Context, ch <-chan ai.StreamChunk) string {
	var sb strings.Builder
	for {
		chunk, ok := <-ch
		if !ok {
			break
		}
		if chunk.Err != nil {
			return ""
		}
		if chunk.Done {
			break
		}
		sb.WriteString(chunk.Delta)
	}
	return sb.String()
}

// buildCompactTranscript serialises the message snapshot into a compact JSON
// array of {role, content, tool_calls, tool_call_id} objects — exactly the
// format the AI received during the conversation. This preserves tool call IDs,
// full arguments, and complete tool results, avoiding the information loss
// of the old flat-text "User:/Assistant:/Tool result:" approach.
func buildCompactTranscript(messages []ai.Message) string {
	var entries []map[string]interface{}

	for _, msg := range messages {
		if msg.Role == "system" {
			continue
		}
		entry := map[string]interface{}{"role": msg.Role}
		switch msg.Role {
		case "user":
			entry["content"] = msg.Content
		case "assistant":
			entry["content"] = msg.Content
			if len(msg.ToolCalls) > 0 {
				tcs := make([]map[string]interface{}, len(msg.ToolCalls))
				for i, tc := range msg.ToolCalls {
					tcs[i] = map[string]interface{}{
						"id":        tc.ID,
						"name":      tc.Name,
						"arguments": tc.Arguments,
					}
				}
				entry["tool_calls"] = tcs
			}
		case "tool":
			entry["tool_call_id"] = msg.ToolCallID
			entry["content"] = msg.Content
		}
		entries = append(entries, entry)
	}

	// Prepend any prior summary so the new one incorporates it.
	var parts []string
	if prior := extractPriorSummary(messages); prior != "" {
		parts = []string{
			"## Prior summary (from earlier compaction — incorporate into new summary)",
			prior,
			"",
			"## Conversation transcript (JSON array — use this to understand what actually happened)",
		}
	}

	jsonBody, _ := json.MarshalIndent(entries, "", "  ")

	if len(parts) > 0 {
		parts = append(parts, string(jsonBody))
		return strings.Join(parts, "\n\n")
	}
	return "## Conversation transcript (JSON array — use this to understand what actually happened)\n\n" + string(jsonBody)
}

// applyCompaction replaces the message history with the system message plus a
// summary block plus the last compactionKeepTurns complete user turns (or
// fewer if msg.keepTurns specifies), then rebuilds display items and schedules
// a recount.
func (m *Model) applyCompaction(msg compactDoneMsg) []tea.Cmd {
	summary := msg.summary
	oldMessages := m.messages

	keepTurns := msg.keepTurns

	// Build the new history: single system message with the summary REPLACED
	// (not appended) to prevent unbounded growth across successive compactions.
	// Merge into one system message for compatibility with providers that reject
	// multiple role:"system" messages (e.g. GitHub Copilot).
	newMessages := make([]ai.Message, 0, 8)
	if len(oldMessages) > 0 && oldMessages[0].Role == "system" {
		systemMsg := oldMessages[0]
		newContent := replaceSummaryBlock(systemMsg.Content, summary)
		// When we split the transcript, append the turn prefix summary
		// so the resumed AI can understand the retained suffix.
		if msg.turnPrefixSummary != "" {
			newContent += "\n\n## Turn Prefix Summary\n\n" + msg.turnPrefixSummary
		}
		systemMsg.Content = newContent
		newMessages = append(newMessages, systemMsg)
	} else {
		sysContent := "## Summary of previous conversation\n\n" + summary
		if msg.turnPrefixSummary != "" {
			sysContent += "\n\n## Turn Prefix Summary\n\n" + msg.turnPrefixSummary
		}
		newMessages = append(newMessages, ai.Message{
			Role:    "system",
			Content: sysContent,
		})
	}

	// If we split the transcript, keep messages from the split point onwards.
	// Otherwise, keep the last N complete turns.
	if msg.splitAt > 0 {
		newMessages = append(newMessages, oldMessages[msg.splitAt:]...)
	} else {
		newMessages = append(newMessages, extractLastTurns(oldMessages, keepTurns)...)
	}
	m.messages = newMessages

	// Use the compact-summary items snapshotted in the Done tick (same Update
	// as when the live item content was finalised).  Fall back to scanning
	// m.items and then compactSummaryFinal for any edge cases.
	savedSummaries := m.compactSavedItems
	m.compactSavedItems = nil
	if len(savedSummaries) == 0 {
		for _, it := range m.items {
			if it.kind == itemCompactSummary {
				savedSummaries = append(savedSummaries, it)
			}
		}
	}
	if len(savedSummaries) == 0 && m.compactSummaryFinal != "" {
		savedSummaries = append(savedSummaries, displayItem{
			kind:    itemCompactSummary,
			handle:  "compact",
			content: m.compactSummaryFinal,
		})
	}
	m.compactSummaryFinal = ""

	// Rebuild display from the new history.
	m.items = rebuildItemsFromMessages(m.messages)
	m.toolCallIdx = make(map[string]int)
	m.streamingItemIdx = -1
	m.reasoningItemIdx = -1
	// Append the compaction summary after the rebuilt history items so it
	// appears as the last entry — like a new message at the bottom of the
	// chat — rather than pinned to the top.
	if len(savedSummaries) > 0 {
		m.items = append(m.items, savedSummaries...)
	}

	m.rebuildContent()

	var cmds []tea.Cmd
	// Always count from the actual newMessages (including extractLastTurns)
	// rather than trusting msg.totalTokens, which previously only counted
	// summary+toolTokens and omitted the verbatim kept-turn tokens.  An
	// inaccurate count here caused shouldAutoCompact to re-fire immediately
	// because processNextCall would re-measure the real size and see it was
	// still above the threshold.
	if count, cerr := ai.CountTokensWithTools(m.modelName, m.messages, m.tools); cerr == nil {
		m.recountGen++
		m.contextUsage = count
		// Back-fill the token-range annotation on the summary item now that we
		// have the accurate post-compaction count.
		if len(savedSummaries) > 0 && m.compactSummaryPrevTok > 0 {
			retainedNote := ""
			if keepTurns > 0 {
				// Count individual messages kept so the user can see whether
				// "2 turns" means 4 messages or 60 (e.g. lots of tool calls).
				retained := extractLastTurns(oldMessages, keepTurns)
				plural := "s"
				if len(retained) == 1 {
					plural = ""
				}
				retainedNote = fmt.Sprintf(", %d msg%s retained", len(retained), plural)
			}
			m.items[len(m.items)-1].toolNote = fmt.Sprintf("%s → %s tokens%s",
				formatTokens(m.compactSummaryPrevTok),
				formatTokens(count.Total),
				retainedNote,
			)
		}
	} else if cmd := m.recountContext(); cmd != nil {
		cmds = append(cmds, cmd)
	}

	if m.autoCompactPending {
		// Auto-compact: restart the AI turn that was interrupted.
		// History was rewritten, so the previous response ID and cached system
		// message are both stale. Use doStream (which may use the incremental
		// client) rather than doStartStream (full context) — the incremental
		// client will fall back to full-context mode when lastResponseID is
		// empty, and on subsequent turns it will send only new messages, which
		// preserves the KV cache prefix for the compacted context.
		//
		// If keepTurns was 0 the compacted history has only the system message
		// (no user turn). Inject a synthetic prompt so the AI has a context to
		// resume from — the AI will re-evaluate and re-issue any tool calls
		// it needs based on the summary context.
		if len(m.messages) == 0 || m.messages[len(m.messages)-1].Role != "user" {
			// Reference the compaction summary so the AI knows what to continue.
			// The summary block is at the top of the system message; this prompt
			// gives the AI a clear instruction to resume from the summary context.
			m.messages = append(m.messages, ai.Message{Role: "user", Content: "Based on the summary above, continue exactly where you left off."})
		}
		m.turnSystemMsg = ""
		m.autoCompactPending = false
		m.turnRoundtrips++
		m.setThinking()
		cmds = append(cmds, m.doStream(m.newOpCtx(), m.buildStreamMessages(m.messages), m.tools))
	} else {
		// Manual compact: also reset since history changed.
		if m.incrementalClient != nil {
			m.incrementalClient.SetLastResponseID("")
		}
		m.turnSystemMsg = ""
		m.setIdle()
		cmds = append(cmds, m.input.Focus())
	}
	return cmds
}

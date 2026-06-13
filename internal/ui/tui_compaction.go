package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
)

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
		// Target ~20 % of the context window post-compaction.  If even 1 turn
		// exceeds that, drop to 0 (summary only).  Never fill to the 65 %
		// compaction threshold — that would guarantee an immediate re-trigger.
		const targetFraction = 0.20
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
	return func() tea.Msg {
		var transcript strings.Builder

		// Prepend any prior summary so the new one incorporates it.
		if prior := extractPriorSummary(snap); prior != "" {
			transcript.WriteString("## Prior summary (from earlier compaction — incorporate into new summary)\n\n")
			transcript.WriteString(prior)
			transcript.WriteString("\n\n---\n\n## New transcript\n\n")
		}

		for i, msg := range snap {
			if msg.Role == "system" {
				continue
			}
			switch msg.Role {
			case "user":
				transcript.WriteString("User: ")
				transcript.WriteString(msg.Content)
			case "assistant":
				if msg.Content != "" {
					transcript.WriteString("Assistant: ")
					transcript.WriteString(msg.Content)
				}
				for _, tc := range msg.ToolCalls {
					raw, _ := json.Marshal(tc.Arguments)
					args := string(raw)
					if len(args) > 300 {
						args = args[:300] + "…"
					}
					fmt.Fprintf(&transcript, "Assistant called tool %q args: %s", tc.Name, args)
				}
			case "tool":
				result := msg.Content
				cap := toolResultCap(i, len(snap))
				if len(result) > cap {
					result = result[:cap] + "…"
				}
				fmt.Fprintf(&transcript, "Tool result (call_id: %s): %s", msg.ToolCallID, result)
			}
			transcript.WriteString("\n\n")
		}

		summaryMessages := []ai.Message{
			{
				Role:    "system",
				Content: chat.CompactionSystemPrompt,
			},
			{
				Role:    "user",
				Content: chat.CompactionUserMessage(transcript.String()),
			},
		}

		// Start the stream and hand the channel back to the Update loop so the
		// summary can be streamed live into the chat log.
		ch := client.CompleteStream(ctx, summaryMessages, nil)
		return compactStartMsg{
			ch:         ch,
			snap:       snap,
			modelName:  modelName,
			toolTokens: toolTokens,
			maxTokens:  maxTokens,
		}
	}
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
		systemMsg.Content = replaceSummaryBlock(systemMsg.Content, summary)
		newMessages = append(newMessages, systemMsg)
	} else {
		newMessages = append(newMessages, ai.Message{
			Role:    "system",
			Content: "## Summary of previous conversation\n\n" + summary,
		})
	}
	newMessages = append(newMessages, extractLastTurns(oldMessages, keepTurns)...)
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
		// message are both stale. Use doStartStream (full context) directly
		// to bypass IncrementalClient, which would otherwise strip messages to
		// "new only" and omit the system message, causing self-hosted models
		// to reject the request ("no user prompt provided").
		//
		// If keepTurns was 0 the compacted history has only the system message
		// (no user turn). Inject a synthetic "Continue." so the AI has a prompt
		// to work with — the AI will re-evaluate and re-issue any tool calls
		// it needs based on the summary context.
		if len(m.messages) == 0 || m.messages[len(m.messages)-1].Role != "user" {
			m.messages = append(m.messages, ai.Message{Role: "user", Content: "Continue."})
		}
		m.turnSystemMsg = ""
		m.autoCompactPending = false
		m.turnRoundtrips++
		m.setThinking()
		cmds = append(cmds, doStartStream(m.newOpCtx(), m.client, m.buildStreamMessages(m.messages), m.tools))
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

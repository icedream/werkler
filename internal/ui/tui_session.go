package ui

import (
	"context"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/sessionstore"
)

// extractPriorSummary returns the body of the "## Summary of previous
// conversation" block embedded in the system message, or "" if none exists.
// The marker line itself is not included in the returned string.
func extractPriorSummary(messages []ai.Message) string {
	if len(messages) == 0 || messages[0].Role != "system" {
		return ""
	}
	const marker = "## Summary of previous conversation\n\n"
	idx := strings.Index(messages[0].Content, marker)
	if idx < 0 {
		return ""
	}
	return strings.TrimSpace(messages[0].Content[idx+len(marker):])
}

// replaceSummaryBlock replaces the existing "## Summary of previous
// conversation" block in systemContent with newSummary, or appends a new
// block if none is present. This prevents the system message from growing
// unboundedly across successive compactions.
//
// Two marker forms are handled:
//   - "\n\n## Summary…" — normal case where the system message has other content before it.
//   - "## Summary…" at position 0 — when the system message consists solely of the summary
//     (created by applyCompaction when there was no prior system message).
func replaceSummaryBlock(systemContent, newSummary string) string {
	const heading = "## Summary of previous conversation\n\n"
	const withLeading = "\n\n" + heading
	if idx := strings.Index(systemContent, withLeading); idx >= 0 {
		return systemContent[:idx] + withLeading + newSummary
	}
	if strings.HasPrefix(systemContent, heading) {
		return heading + newSummary
	}
	return systemContent + withLeading + newSummary
}

// extractLastTurns returns the last n complete user turns (and everything
// between/after them) from messages, excluding any system messages.
// If there are fewer than n user turns, all non-system messages are returned.
func extractLastTurns(messages []ai.Message, n int) []ai.Message {
	if n <= 0 {
		return nil
	}
	// Collect indices of user messages.
	var userIdxs []int
	for i, msg := range messages {
		if msg.Role == "user" {
			userIdxs = append(userIdxs, i)
		}
	}
	var cutAt int
	if len(userIdxs) > n {
		cutAt = userIdxs[len(userIdxs)-n]
	}
	// Guard: cutAt should always land on a user message by construction; if a
	// future refactor breaks that invariant, scan forward to the next user message.
	if cutAt < len(messages) {
		if got := messages[cutAt].Role; got != "user" && got != "system" {
			for cutAt < len(messages) && messages[cutAt].Role != "user" {
				cutAt++
			}
		}
	}
	// Return non-system messages from cutAt onward, stripping Reasoning to save
	// session-file size (reasoning content is never sent to the API anyway).
	var out []ai.Message
	for _, msg := range messages[cutAt:] {
		if msg.Role == "system" {
			continue
		}
		if msg.Role == "assistant" {
			msg.Reasoning = "" // Not sent to API; strip to save session file size.
		}
		out = append(out, msg)
	}
	return out
}

// applySession restores a previously saved session into the model,
// rebuilding the display items from the stored message history.
func (m *Model) applySession(sess *sessionstore.Session) {
	m.sessionID = sess.ID
	m.sessionCreatedAt = sess.CreatedAt
	m.sessionCWD = sess.CWD
	m.messages = sess.Messages
	// Grant AI access to the workspace dir for this restored session.
	if m.session != nil && m.sessionStore != nil {
		m.session.SetWorkspaceDir(m.workspaceDir())
	}

	// Restore mode from session. If the saved mode can no longer be resolved
	// (e.g. user deleted a custom mode), fall back to default silently.
	if sess.ModeName != "" && sess.ModeName != "default" {
		for _, mode := range m.allModes {
			if mode.Name == sess.ModeName {
				m.activeMode = mode
				break
			}
		}
	}

	// Restore agent from session. If the saved agent no longer exists, warn
	// but continue without it.
	if sess.AgentName != "" {
		found := false
		for _, a := range m.agents {
			if a.Name == sess.AgentName {
				m.activeAgent = &a
				if m.toolMgr != nil {
					m.toolMgr.SetToolFilter(a.ToolList())
				}
				found = true
				break
			}
		}
		if !found {
			m.items = append(m.items, displayItem{
				kind:    itemInfo,
				content: fmt.Sprintf("Warning: agent %q from saved session not found -- continuing without it.", sess.AgentName),
			})
		}
	}

	// Replace messages[0] with a freshly built system prompt so any stale
	// MCP-server or memory injections from the saved session are removed.
	// buildStreamMessages will re-add the current injections each turn.
	m.rebuildSystemPrompt()
	m.items = rebuildItemsFromMessages(sess.Messages)
	if m.todoStore != nil {
		if len(sess.Todos) > 0 {
			m.todoStore.Restore(sess.Todos)
			m.sidebarOpen = true
		} else {
			m.todoStore.Clear()
			m.sidebarOpen = false
		}
		m.recalcLayout()
	}
	m.populateHistoryFromMessages(sess.Messages)
	// Restore session title; mark refined to suppress re-generation.
	m.sessionTitle = sess.Title
	m.sessionTitleRefined = (sess.Title != "")
	// Show a resume banner as the first visible item.
	banner := displayItem{
		kind:    itemInfo,
		content: fmt.Sprintf("Resumed session: %q  (%s)", sess.Title, sess.UpdatedAt.Format("2006-01-02 15:04")),
	}
	m.items = append([]displayItem{banner}, m.items...)
}

// rebuildItemsFromMessages converts a stored message history back into display items.
// System messages and tool-result messages are skipped; assistant/user messages are shown.
func rebuildItemsFromMessages(msgs []ai.Message) []displayItem {
	var items []displayItem
	for _, msg := range msgs {
		switch msg.Role {
		case "user":
			items = append(items, displayItem{kind: itemUser, content: msg.Content})
		case "assistant":
			if msg.Reasoning != "" {
				items = append(items, displayItem{kind: itemReasoning, content: msg.Reasoning})
			}
			if len(msg.ToolCalls) > 0 {
				for _, tc := range msg.ToolCalls {
					args := formatArgsCompact(tc.Arguments)
					label := tc.Name
					if args != "" {
						label += "  " + args
					}
					items = append(items, displayItem{
						kind:       itemToolCall,
						toolName:   tc.Name,
						content:    label,
						toolStatus: toolStatusDone, // historical calls are done
					})
				}
			} else if msg.Content != "" {
				items = append(items, displayItem{kind: itemAssistant, content: msg.Content})
			}
		}
	}
	return items
}

// doSaveSession saves the current session state asynchronously.
// doRefineSessionTitle fires an async non-streaming AI call to generate a concise
// session title from the first user message and first AI text response.
// It dispatches sessionTitleRefinedMsg when done (error is non-fatal).
func doRefineSessionTitle(ctx context.Context, client ai.StreamCompleter, firstUserMsg, firstAIText string) tea.Cmd {
	return func() tea.Msg {
		const maxInput = 200
		truncate := func(s string) string {
			if len(s) > maxInput {
				return s[:maxInput] + "…"
			}
			return s
		}
		prompt := "User: " + truncate(firstUserMsg) + "\nAssistant: " + truncate(firstAIText) + "\nTitle:"
		msgs := []ai.Message{
			{
				Role: "system",
				Content: "You generate extremely short session titles. " +
					"Reply with ONLY the title text — no punctuation, no quotes, no explanation. 6 words maximum.",
			},
			{Role: "user", Content: prompt},
		}
		completer, ok := client.(ai.Completer)
		if !ok {
			return sessionTitleRefinedMsg{err: fmt.Errorf("client does not support Complete")}
		}
		result, err := completer.Complete(ctx, msgs, nil)
		if err != nil {
			return sessionTitleRefinedMsg{err: err}
		}
		// Trim to first line, cap at 60 chars.
		title := strings.SplitN(strings.TrimSpace(result.Content), "\n", 2)[0]
		if len(title) > 60 {
			title = title[:60]
		}
		return sessionTitleRefinedMsg{title: title}
	}
}

// seq is a monotonically increasing counter; the sessionSavedMsg handler
// discards arrivals with a seq older than the most recent confirmed save.
func doSaveSession(store *sessionstore.Store, sess sessionstore.Session, seq int) tea.Cmd {
	return func() tea.Msg {
		_ = store.Save(&sess)
		return sessionSavedMsg{seq: seq}
	}
}

// saveSession snapshots the current session, increments the save sequence,
// and returns a tea.Cmd that writes it to disk asynchronously.
// Must only be called when m.sessionStore != nil.
func (m *Model) saveSession() tea.Cmd {
	prevID := m.sessionID
	snap := m.currentSessionSnapshot()
	m.sessionID = snap.ID
	m.sessionCreatedAt = snap.CreatedAt
	// If this is the first save (ID just became known), grant AI access to the
	// workspace directory and rebuild the system prompt to include its path.
	if prevID == "" && m.sessionID != "" {
		if m.session != nil {
			m.session.SetWorkspaceDir(m.workspaceDir())
		}
		m.rebuildSystemPrompt()
	}
	m.saveSeq++
	return doSaveSession(m.sessionStore, snap, m.saveSeq)
}

// doLoadSessionHint loads the most recent session for the given CWD asynchronously.
func doLoadSessionHint(store *sessionstore.Store, cwd string) tea.Cmd {
	return func() tea.Msg {
		sess, err := store.LoadLatestForCWD(cwd)
		if err != nil || sess == nil {
			return sessionHintMsg{}
		}
		return sessionHintMsg{sess: sess}
	}
}

// doLoadSessions loads all sessions for the picker asynchronously.
func doLoadSessions(store *sessionstore.Store) tea.Cmd {
	return func() tea.Msg {
		sessions, err := store.List()
		return sessionsListMsg{sessions: sessions, err: err}
	}
}

// currentSessionSnapshot returns a sessionstore.Session reflecting the model's
// current state; used for auto-save.
func (m *Model) currentSessionSnapshot() sessionstore.Session {
	now := time.Now()
	createdAt := m.sessionCreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	id := m.sessionID
	if id == "" {
		id = sessionstore.NewID()
	}
	// Use the in-memory refined title if available; fall back to GenerateTitle.
	title := m.sessionTitle
	if title == "" {
		title = sessionstore.GenerateTitle(m.messages)
	}
	snap := sessionstore.Session{
		ID:        id,
		Title:     title,
		CWD:       m.sessionCWD,
		Messages:  m.messages,
		CreatedAt: createdAt,
		UpdatedAt: now,
	}
	if !m.activeMode.IsDefault && m.activeMode.Name != "" {
		snap.ModeName = m.activeMode.Name
	}
	if m.activeAgent != nil {
		snap.AgentName = m.activeAgent.Name
	}
	if m.todoStore != nil {
		snap.Todos = m.todoStore.List()
	}
	return snap
}

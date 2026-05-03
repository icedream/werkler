package ui

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/lipgloss"
	"github.com/icedream/werkler/internal/agents"
	"github.com/icedream/werkler/internal/config"
	"github.com/icedream/werkler/internal/todostore"
	"github.com/muesli/reflow/wordwrap"
)

// formatAgentPreview renders an Agent as a human-readable TOML-like string
// for the review screen.
func formatAgentPreview(a agents.Agent) string {
	s := fmt.Sprintf("name        = %q\ndescription = %q\nwhen        = %q\n\ninstructions = \"\"\"\n%s\n\"\"\"", a.Name, a.Description, a.When, a.Instructions)
	if a.Tools != nil {
		tl := a.ToolList()
		if len(tl) == 0 {
			s += "\ntools = []  # no tools (conversation-only)"
		} else {
			s += "\ntools = ["
			for _, t := range tl {
				s += "\n  " + fmt.Sprintf("%q", t) + ","
			}
			s += "\n]"
		}
	}
	return s
}

// syncViewportHeight recalculates and applies the viewport height based on the
// terminal size and current completion state. Must be called after any change
// that affects completionLineCount or terminal dimensions.
func (m *Model) syncViewportHeight() {
	vph := m.height - fixedLines - m.completionLineCount()
	if vph < 1 {
		vph = 1
	}
	m.viewport.Height = vph
}

// recalcLayout updates viewport and input widths to account for whether the
// sidebar is open. Must be called after m.width, m.height, or m.sidebarOpen changes.
func (m *Model) recalcLayout() {
	mainW := m.width
	if m.sidebarOpen && m.todoStore != nil && m.width-m.sidebarWidth >= minMainWidth {
		mainW = m.width - m.sidebarWidth
	}
	m.viewport.Width = mainW
	m.input.Width = mainW - 5 // 5 = len("You> ")
	m.syncViewportHeight()
}

// activeBorderColor returns the lipgloss color for the TUI separator lines,
// derived from the active mode's color (or the default muted gray).
func (m Model) activeBorderColor() lipgloss.Color {
	if m.activeMode.Color != "" {
		return lipgloss.Color(m.activeMode.Color)
	}
	return lipgloss.Color("238")
}

// modeSeparator returns a horizontal rule string styled with the active mode color.
func (m Model) modeSeparator(width int) string {
	return lipgloss.NewStyle().Foreground(m.activeBorderColor()).Render(strings.Repeat("─", width))
}

func (m Model) View() string {
	if m.width == 0 {
		return "" // not yet initialised — avoid rendering before first WindowSizeMsg
	}

	// Pickers take over the full screen.
	if m.state == statePickingModel {
		return m.modelPicker.View()
	}
	if m.state == statePickingSession {
		return m.sessionPicker.View()
	}
	if m.state == statePickingTools {
		return m.toolPicker.View()
	}
	if m.state == stateViewingToolDetail {
		return m.toolDetailView()
	}
	if m.state == statePickingSkills {
		return m.skillPicker.View()
	}
	if m.state == statePickingMode {
		return m.modePicker.View()
	}
	if m.state == statePickingRegistry {
		return m.registryView()
	}
	if m.state == stateAgentWizardMode ||
		m.state == stateAgentWizardDescribe ||
		m.state == stateAgentWizardGenerate ||
		m.state == stateAgentWizardReview ||
		m.state == stateAgentWizardManual ||
		m.state == stateAgentWizardTools ||
		m.state == stateAgentWizardDone {
		return m.agentWizardView()
	}

	sep := m.modeSeparator(m.width)
	if m.session.AllowAll() {
		sep = separatorAllowAllStyle.Render(strings.Repeat("─", m.width))
	}

	var b strings.Builder

	b.WriteString(m.headerView())
	b.WriteString("\n")
	b.WriteString(sep)
	b.WriteString("\n")

	// Main viewport, optionally with todo sidebar on the right.
	showSidebar := m.sidebarOpen && m.todoStore != nil && m.width-m.sidebarWidth >= minMainWidth
	if showSidebar {
		sidebarSep := lipgloss.NewStyle().Foreground(m.activeBorderColor())
		sepCol := sidebarSep.Render(strings.Repeat("│\n", m.viewport.Height))
		// Trim the trailing newline from the sep column before joining.
		sepCol = strings.TrimSuffix(sepCol, "\n")
		b.WriteString(lipgloss.JoinHorizontal(lipgloss.Top,
			m.viewport.View(),
			sepCol,
			m.sidebarView(),
		))
	} else {
		b.WriteString(m.viewport.View())
	}
	b.WriteString("\n")

	b.WriteString(sep)
	b.WriteString("\n")

	s1, s2 := m.statusLines()
	b.WriteString(s1)
	b.WriteString("\n")
	b.WriteString(s2)
	b.WriteString("\n")

	b.WriteString(sep)
	b.WriteString("\n")

	if m.showCompletion {
		b.WriteString(m.completionView())
	}

	b.WriteString(m.inputView())

	return b.String()
}

// --- Helper views ---

// effectiveWindowTitle returns the string that should be set as the terminal
// window title. Format: "werkler — <session title> · <task title>".
// Components that are empty are omitted.
func (m Model) effectiveWindowTitle() string {
	base := "werkler"
	if m.sessionTitle != "" {
		base += " — " + m.sessionTitle
	}
	if m.currentTaskTitle != "" {
		base += " · " + m.currentTaskTitle
	}
	return base
}

func (m Model) headerView() string {
	servers := "none"
	if n := len(m.serverNames); n == 1 {
		servers = m.serverNames[0]
	} else if n > 1 {
		servers = fmt.Sprintf("%d servers", n)
	}
	text := "werkler"
	if m.sessionTitle != "" {
		text += " · " + m.sessionTitle
	}
	text += fmt.Sprintf("  model: %s  mcp: %s", m.modelName, servers)
	if m.contextUsage.Total > 0 {
		maxTok := m.modelInfo.Context.MaxTokens
		ctx := m.contextUsage.FormatWithMax(maxTok)
		text += "  ctx: " + ctx
	}
	// Truncate to terminal width to avoid wrapping (headerStyle pads to width but
	// doesn't clip — add a hard cap so long model names don't push onto a second line).
	if m.width > 0 && len(text) > m.width {
		text = text[:m.width-1] + "…"
	}
	return headerStyle.Width(m.width).Render(text)
}

// approvalKey renders a single key hint for an approval dialog.
// When key == selected it uses the highlighted style; otherwise the normal hint style.
func (m Model) approvalKey(key, selected string) string {
	label := "[" + key + "]"
	if key == selected {
		return approvalSelectedStyle.Render(label)
	}
	return keyHintStyle.Render(label)
}

func (m Model) statusLines() (line1, line2 string) {
	queueHint := ""
	if n := len(m.queuedPrompts); n > 0 {
		queueHint = "  " + queueCountStyle.Render(fmt.Sprintf("+%d queued", n))
	}
	allowAllIndicator := ""
	if m.session.AllowAll() {
		allowAllIndicator = "  " + allowAllWarningStyle.Render("⚠ allow-all ON")
	}
	autopilotIndicator := ""
	if m.autopilot {
		autopilotIndicator = "  " + autopilotStyle.Render(fmt.Sprintf("⚡ %d/%d", m.autopilotCycle, m.effectiveAutopilotMax()))
	} else if m.autopilotPaused {
		autopilotIndicator = "  " + autopilotStyle.Render(fmt.Sprintf("⚡ paused (%d cycles)", m.autopilotCycle))
	}

	switch m.state {
	case stateThinking:
		cancelHint := ""
		if m.cancelPending {
			cancelHint = "  " + keyHintStyle.Render("[esc]") + " to cancel"
		}
		label := "Thinking…"
		if m.currentTaskTitle != "" {
			label = m.currentTaskTitle
		}
		return statusStyle.Render(m.spinner.View()+" "+label) + cancelHint + queueHint + autopilotIndicator + allowAllIndicator + m.thinkingElapsedHint() + m.roundtripHint(), ""
	case stateConnectingOAuth:
		return statusStyle.Render(m.spinner.View()+" Waiting for OAuth authorization…") + queueHint + autopilotIndicator + allowAllIndicator, ""
	case stateCompacting:
		return statusStyle.Render(m.spinner.View()+" Compacting context…") + queueHint + autopilotIndicator + allowAllIndicator, ""
	case stateStreaming:
		cancelHint := ""
		if m.cancelPending {
			cancelHint = "  " + keyHintStyle.Render("[esc]") + " to cancel"
		}
		label := "Streaming…"
		if m.currentTaskTitle != "" {
			label = m.currentTaskTitle
		}
		liveSpeed := ""
		if m.showTokenSpeed && m.state == stateStreaming && !m.streamingStart.IsZero() && m.streamedTokens > 0 {
			elapsed := time.Since(m.streamingStart).Seconds()
			if elapsed > 0 {
				liveSpeed = fmt.Sprintf("  ⏩ %.1f tok/s", float64(m.streamedTokens)/elapsed)
			}
		}
		return statusStyle.Render(m.spinner.View()+" "+label) + liveSpeed + cancelHint + queueHint + autopilotIndicator + allowAllIndicator + m.roundtripHint(), ""

	case stateCallingTool:
		name := renderToolName(m.callingToolName)
		if m.callingToolTitle != "" {
			name = toolNameStyle.Render(m.callingToolTitle) + " " + toolDimStyle.Render("["+m.callingToolName+"]")
		}
		cancelHint := ""
		if m.cancelPending {
			cancelHint = "  " + keyHintStyle.Render("[esc]") + " to cancel"
		}
		return statusStyle.Render(m.spinner.View()+" Calling tool: ") + name + cancelHint + queueHint + autopilotIndicator + allowAllIndicator + m.roundtripHint(), ""
	case stateAwaitingPathApproval:
		if m.currentPathRequest.Path == "" {
			return "", ""
		}
		accessKind := "read"
		if m.currentPathRequest.Execute {
			accessKind = "execute"
		} else if m.currentPathRequest.Write {
			accessKind = "write"
		}
		ch := m.pendingApprovalChoice
		displayPath := m.currentPathRequest.Path
		if ch == "d" {
			displayPath = filepath.Dir(m.currentPathRequest.Path) + string(filepath.Separator) + "…"
		}
		l1 := approvalPromptStyle.Render(fmt.Sprintf("  Allow %s access: ", accessKind)) + displayPath
		remaining := len(m.pendingPathApprovals)
		if remaining > 0 {
			l1 += statusStyle.Render(fmt.Sprintf(" (+%d more)", remaining))
		}
		l2 := approvalPromptStyle.Render("Allow? ") +
			m.approvalKey("y", ch) + "es  " +
			m.approvalKey("d", ch) + "irectory  " +
			m.approvalKey("n", ch) + "o  " +
			m.approvalKey("a", ch) + "ll remaining"
		if m.persistPathApproval != nil {
			l2 += "  " + m.approvalKey("p", ch) + "ermanent"
		}
		if ch != "" {
			l2 += "  " + keyHintStyle.Render("[↵]") + " confirm  " + keyHintStyle.Render("[esc]") + " cancel"
		}
		return l1 + allowAllIndicator, l2
	case stateAwaitingApproval:
		if m.currentCall == nil {
			return "", ""
		}
		args := toolCallDisplayArgs(m.currentCall.Name, m.currentCall.Arguments)
		l1 := "  ▶ " + renderToolName(m.currentCall.Name)
		if args != "" {
			l1 += "  " + args
		}
		ch := m.pendingApprovalChoice
		l2 := approvalPromptStyle.Render("Allow? ") +
			m.approvalKey("y", ch) + "es  " +
			m.approvalKey("n", ch) + "o  " +
			m.approvalKey("a", ch) + "lways"
		if m.persistToolApproval != nil {
			l2 += "  " + m.approvalKey("p", ch) + "ermanent"
		}
		if ch != "" {
			l2 += "  " + keyHintStyle.Render("[↵]") + " confirm  " + keyHintStyle.Render("[esc]") + " cancel"
		}
		return l1 + allowAllIndicator, l2
	case stateAwaitingUserQuestion:
		l1 := approvalPromptStyle.Render("  ❓ AI is asking a question")
		l2 := "  "
		if len(m.askUserChoices) > 0 {
			l2 += keyHintStyle.Render("[↑/↓]") + " navigate  "
		}
		l2 += keyHintStyle.Render("[Enter]") + " confirm"
		if m.askUserAllowFreeform {
			l2 += "  " + keyHintStyle.Render("[type]") + " custom answer"
		}
		if m.modelManager != nil {
			l2 += "  " + keyHintStyle.Render("ctrl+p") + " switch model"
		}
		return l1 + allowAllIndicator, l2
	default:
		mouseHint := "  " + keyHintStyle.Render("alt+m") + " enable text selection"
		if !m.mouseEnabled {
			mouseHint = "  " + keyHintStyle.Render("alt+m") + " enable mouse scroll"
		}
		pickerHint := ""
		if m.modelManager != nil {
			pickerHint = "  " + keyHintStyle.Render("ctrl+p") + " switch model"
		}
		sessionHint := ""
		if m.sessionStore != nil {
			sessionHint = "  " + keyHintStyle.Render("ctrl+r") + " sessions"
		}
		todoHint := ""
		if m.todoStore != nil && !m.sidebarOpen {
			if p, a, d, b := m.todoStore.Counts(); p+a+d+b > 0 {
				todoHint = "  " + todoIndicatorStyle.Render(fmt.Sprintf("✓%d ▶%d ○%d", d, a, p))
			}
		}
		modeHint := ""
		if len(m.allModes) > 0 {
			name := m.activeMode.Name
			if name == "" {
				name = "default"
			}
			modeColor := m.activeBorderColor()
			modeStyle := lipgloss.NewStyle().Foreground(modeColor).Bold(true)
			modeHint = "  " + modeStyle.Render("["+name+"]") + " " + statusStyle.Render("shift+tab")
		}
		agentHint := ""
		if m.activeAgent != nil {
			name := m.activeAgent.Name
			if len(name) > 16 {
				name = name[:13] + "..."
			}
			agentHint = "  " + keyHintStyle.Render("[agent: "+name+"]")
		}
		imageHint := ""
		if n := len(m.pendingImages); n > 0 {
			imageHint = "  " + keyHintStyle.Render(fmt.Sprintf("📎 %d image(s) staged", n))
		}
		line1 := mouseHint + pickerHint + sessionHint + todoHint + modeHint + agentHint + imageHint + allowAllIndicator
		// Show rate limit info when the provider has reported it.
		if m.lastRateLimits.IsKnown() {
			parts := []string{}
			if t := m.lastRateLimits.FormatTokens(); t != "" {
				parts = append(parts, t+" tok")
			}
			if r := m.lastRateLimits.FormatRequests(); r != "" {
				parts = append(parts, r+" req")
			}
			if len(parts) > 0 {
				line1 += "  " + statusStyle.Render(strings.Join(parts, " · ")+" remaining")
			}
		}
		// Show resume hint on line2 until the user sends their first message.
		line2 := ""
		if m.resumeHint != nil && m.sessionStore != nil {
			line2 = statusStyle.Render(fmt.Sprintf(
				"  Last session: %q (%s) — ctrl+r to resume",
				m.resumeHint.Title, formatAge(m.resumeHint.UpdatedAt),
			))
		}
		return line1, line2
	}
}

func (m Model) inputView() string {
	prefix := inputPrefixStyle.Render("You> ")
	return prefix + m.input.View()
}

// sidebarView renders the todo sidebar panel. Width is sidebarWidth-1 (the
// caller prepends the "│" separator). Height matches the viewport height.
func (m Model) sidebarView() string {
	contentW := m.sidebarWidth - 1 // 1 col reserved for the "│" separator
	todos := m.todoStore.List()
	vpH := m.viewport.Height

	lines := make([]string, 0, vpH)
	title := sidebarTitleStyle.Width(contentW).Render("Todos")
	lines = append(lines, title)

	icons := map[string]string{
		todostore.StatusPending:    "○",
		todostore.StatusInProgress: "▶",
		todostore.StatusDone:       "✓",
		todostore.StatusBlocked:    "✗",
	}

	for _, t := range todos {
		icon := icons[t.Status]
		if icon == "" {
			icon = "?"
		}
		// Max width for title: contentW - 3 (icon + space + padding)
		maxTitle := contentW - 3
		title := t.Title
		if len([]rune(title)) > maxTitle {
			title = string([]rune(title)[:maxTitle-1]) + "…"
		}
		label := icon + " " + title
		var rendered string
		switch t.Status {
		case todostore.StatusDone:
			rendered = sidebarDoneStyle.Width(contentW).Render(label)
		case todostore.StatusInProgress:
			rendered = sidebarActiveStyle.Width(contentW).Render(label)
		case todostore.StatusBlocked:
			rendered = sidebarBlockedStyle.Width(contentW).Render(label)
		default:
			rendered = sidebarItemStyle.Width(contentW).Render(label)
		}
		lines = append(lines, rendered)
	}

	if len(todos) == 0 {
		lines = append(lines, sidebarItemStyle.Width(contentW).Render("  (empty)"))
	}

	// Pad to viewport height so the sidebar fills the full column.
	blank := strings.Repeat(" ", contentW)
	for len(lines) < vpH {
		lines = append(lines, blank)
	}
	if len(lines) > vpH {
		lines = lines[:vpH]
	}
	return strings.Join(lines, "\n")
}

// roundtripHint returns a status-bar fragment warning the user when the agent
// has made many completion calls within the current turn. Empty when below the
// first threshold or when the agent is idle.
//
// Thresholds: ≥20 → yellow hint, ≥40 → orange warning.
func (m Model) roundtripHint() string {
	const warnThreshold = 40
	const hintThreshold = 20
	switch {
	case m.turnRoundtrips >= warnThreshold:
		return "  " + roundtripWarnStyle.Render(fmt.Sprintf("⚠ %d roundtrips", m.turnRoundtrips))
	case m.turnRoundtrips >= hintThreshold:
		return "  " + roundtripHintStyle.Render(fmt.Sprintf("⚠ %d roundtrips", m.turnRoundtrips))
	default:
		return ""
	}
}

// thinkingElapsedHint returns a dimmed elapsed-time string (e.g. " 12s") once
// thinking has lasted more than a few seconds, so users can see the model is
// still working. Returns an empty string when elapsed is negligible or when
// not in a thinking state.
func (m Model) thinkingElapsedHint() string {
	if m.thinkingStart.IsZero() {
		return ""
	}
	d := time.Since(m.thinkingStart).Round(time.Second)
	if d < 3*time.Second {
		return ""
	}
	return "  " + statusStyle.Render(fmt.Sprintf("%ds", int(d.Seconds())))
}

// completionView renders the slash-command popup lines that appear above the
// input. Each line is truncated to m.width to guarantee exactly one terminal row.
func (m Model) completionView() string {
	filtered := m.filteredCmds()
	if len(filtered) == 0 {
		return ""
	}
	var sb strings.Builder
	for i, cmd := range filtered {
		selected := i == m.completionIdx
		var line string
		if selected {
			line = "  " + completionSelectedNameStyle.Render("/"+cmd.name)
			line += "  " + completionSelectedStyle.Render(cmd.description)
		} else {
			line = "  " + completionNameStyle.Render("/"+cmd.name)
			line += "  " + completionItemStyle.Render(cmd.description)
		}
		// Truncate to terminal width (guarantees exactly one rendered row).
		// lipgloss.Width is ANSI-aware so escape codes don't inflate the count.
		if m.width > 0 && lipgloss.Width(line) > m.width {
			// Re-render without description when the line is too wide.
			if selected {
				line = "  " + completionSelectedNameStyle.Render("/"+cmd.name)
			} else {
				line = "  " + completionNameStyle.Render("/"+cmd.name)
			}
		}
		sb.WriteString(line)
		sb.WriteString("\n")
	}
	return sb.String()
}

// --- Viewport content ---

func (m *Model) rebuildContent() {
	var sb strings.Builder
	for i, item := range m.items {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(m.renderItem(item))
	}
	m.viewport.SetContent(sb.String())
	m.viewport.GotoBottom()
}

func (m Model) renderItem(item displayItem) string {
	switch item.kind {
	case itemReasoning:
		if item.content == "" || !m.showReasoning {
			return "" // empty slot or reasoning display is off
		}
		prefix := reasoningPrefixStyle.Render("💭 Thinking")
		body := item.content
		// Indent each line by two spaces and apply dim style.
		lines := strings.Split(strings.TrimRight(body, "\n"), "\n")
		styledLines := make([]string, len(lines))
		for i, l := range lines {
			styledLines[i] = reasoningBodyStyle.Render("  " + l)
		}
		return prefix + "\n" + strings.Join(styledLines, "\n")

	case itemUser:
		prefix := userPrefixStyle.Render("You")
		prefixWidth := lipgloss.Width(prefix) + 2 // +2 for the two spaces after prefix
		body := item.content
		if m.width > prefixWidth+1 {
			body = wordwrap.String(body, m.width-4-prefixWidth)
			// Indent continuation lines to align under the content start.
			indent := strings.Repeat(" ", prefixWidth)
			body = strings.ReplaceAll(body, "\n", "\n"+indent)
		}
		return prefix + "  " + body

	case itemAssistant:
		prefix := assistantPrefixStyle.Render("Werkler")
		body := renderMarkdown(m.renderer, item.content)
		return prefix + "\n" + body

	case itemToolCall:
		var badge string
		switch item.toolStatus {
		case toolStatusPending:
			badge = toolPendingStyle.Render("◦")
		case toolStatusRunning:
			badge = toolPendingStyle.Render("▶")
		case toolStatusDone:
			badge = toolApprovedStyle.Render("✓")
		case toolStatusFailed:
			badge = toolDeniedStyle.Render("✗")
		case toolStatusDenied:
			badge = toolDeniedStyle.Render("✗")
		}
		// For tools that carry an AI-formulated title (process_start, run_command),
		// use the title as the primary display name instead of the generic friendly name.
		var name string
		if item.toolNote != "" && (item.toolName == "process_start" || item.toolName == "run_command") {
			name = toolNameStyle.Render(item.toolNote) + " " + toolDimStyle.Render("["+item.toolName+"]")
		} else {
			name = renderToolName(item.toolName)
		}
		badgeAndName := "  " + badge + " " + name
		prefixW := lipgloss.Width(badgeAndName)
		line := badgeAndName
		// Suppress the compact badge args while the expanded streaming block is
		// active — the structured view below already shows the full arguments
		// and the compact form flashes as raw JSON on the Done-chunk update.
		expandedActive := item.handle != "" && !m.collapsedHandles[item.handle] &&
			item.toolRawArgs != "" &&
			(item.toolStatus == toolStatusPending || item.toolStatus == toolStatusRunning)
		if item.toolArgs != "" && !expandedActive {
			args := item.toolArgs
			sep := "  "
			if m.width > prefixW+len(sep)+1 {
				available := m.width - prefixW - len(sep)
				wrapped := wordwrap.String(args, available)
				indent := strings.Repeat(" ", prefixW+len(sep))
				args = strings.ReplaceAll(wrapped, "\n", "\n"+indent)
			}
			line += sep + args
		}
		if item.toolStatus == toolStatusDenied {
			line += "  " + toolDeniedStyle.Render("(denied)")
		}
		// Only show toolNote as a secondary annotation when it wasn't already
		// promoted to the name line above.
		if item.toolNote != "" && item.toolName != "process_start" && item.toolName != "run_command" {
			var ns lipgloss.Style
			if item.toolStatus == toolStatusFailed {
				ns = errorStyle
			} else {
				ns = statusStyle
			}
			const noteIndent = "    "
			note := item.toolNote
			if m.width > len(noteIndent)+1 {
				wrapped := wordwrap.String(note, m.width-len(noteIndent))
				noteParts := strings.Split(wrapped, "\n")
				rendered := make([]string, len(noteParts))
				for j, p := range noteParts {
					rendered[j] = noteIndent + ns.Render(p)
				}
				line += "\n" + strings.Join(rendered, "\n")
			} else {
				line += "\n" + noteIndent + ns.Render(note)
			}
		}
		// While executing (uncollapsed) show args via the tool-specific streaming
		// renderer.  For completed file tools with a real diff, show the diff
		// instead of falling back to pretty-printed JSON.
		if item.handle != "" && !m.collapsedHandles[item.handle] && item.toolRawArgs != "" {
			var argsDisplay string
			switch {
			case item.toolStatus == toolStatusPending || item.toolStatus == toolStatusRunning:
				argsDisplay = renderStreamingArgs(item.toolRawArgs, item.toolName, m.width)
			case item.toolDiff != "":
				// Completed file tool — prefer the real diff over pretty JSON.
				argsDisplay = renderDiff(item.toolDiff, m.width)
			default:
				argsDisplay = formatExpandedArgs(item.toolRawArgs, m.width)
			}
			if argsDisplay != "" {
				line += "\n" + argsDisplay
			}
		}
		// Show diff change-count stub when collapsed (tool done with a diff).
		if item.toolDiff != "" && item.handle != "" && m.collapsedHandles[item.handle] {
			changed := countDiffChangedLines(item.toolDiff)
			plural := "s"
			if changed == 1 {
				plural = ""
			}
			line += "\n" + toolDimStyle.Render(fmt.Sprintf(
				"  \u2195 %d line%s changed  (use /expand %s to show diff)",
				changed, plural, item.handle))
		}
		// Live stdout/stderr while run_command is executing (cleared on done).
		if item.toolLiveOutput != "" && item.toolStatus == toolStatusRunning {
			outLines := strings.Split(item.toolLiveOutput, "\n")
			const maxLiveLines = 30
			if len(outLines) > maxLiveLines {
				outLines = outLines[len(outLines)-maxLiveLines:]
			}
			// Indent is 2 spaces; leave at least 20 chars for content.
			maxW := m.width - 2
			if maxW < 20 {
				maxW = 20
			}
			for _, l := range outLines {
				// Wordwrap long lines so the viewport never clips mid-character.
				wrapped := wordwrap.String(l, maxW)
				for i, wl := range strings.Split(wrapped, "\n") {
					if i == 0 {
						line += "\n" + toolDimStyle.Render("  "+wl)
					} else {
						// Continuation lines get extra indent to show they wrapped.
						line += "\n" + toolDimStyle.Render("    "+wl)
					}
				}
			}
		}
		return line

	case itemError:
		label := errorStyle.Render("Error: ")
		labelW := lipgloss.Width(label)
		content := item.content
		if m.width > labelW+1 {
			wrapped := wordwrap.String(content, m.width-labelW)
			indent := strings.Repeat(" ", labelW)
			content = strings.ReplaceAll(wrapped, "\n", "\n"+indent)
		}
		return label + content

	case itemInfo:
		content := item.content
		if m.width > 0 {
			content = wordwrap.String(content, m.width)
		}
		return infoStyle.Render(content)

	case itemMarkdown:
		return renderMarkdown(m.renderer, item.content)

	case itemProcessOutput:
		// Use toolNote (AI-formulated intent) as primary label; fall back to "[process:<handle>]".
		label := "[process:" + item.handle + "]"
		if item.toolNote != "" {
			label = item.toolNote
		}
		prefix := processHandleStyle.Render(label)
		var sb strings.Builder
		sb.WriteString(prefix)
		if m.collapsedHandles[item.handle] {
			lines := strings.SplitN(item.content, "\n", 4)
			shown := lines
			hiddenFromContent := 0
			if len(lines) > 2 {
				shown = lines[:2]
				hiddenFromContent = len(lines) - 2
			}
			if len(shown) > 0 {
				sb.WriteString("\n")
				sb.WriteString(strings.Join(shown, "\n"))
			}
			totalHidden := hiddenFromContent + item.truncatedLines
			if totalHidden > 0 {
				sb.WriteString("\n")
				sb.WriteString(toolDimStyle.Render(fmt.Sprintf("  … %d more lines (use /expand %s to show all)", totalHidden, item.handle)))
			}
		} else {
			if item.truncatedLines > 0 {
				sb.WriteString("\n")
				sb.WriteString(toolDimStyle.Render(fmt.Sprintf("  … %d lines above …", item.truncatedLines)))
			}
			// Raw output may contain ANSI codes, display as-is.
			sb.WriteString("\n")
			sb.WriteString(item.content)
		}
		return sb.String()

	case itemCompactSummary:
		prefix := processHandleStyle.Render("📋 Context summary")
		if item.toolNote != "" {
			prefix += "  " + toolDimStyle.Render(item.toolNote)
		}
		var sb strings.Builder
		sb.WriteString(prefix)
		switch {
		case item.content == "":
			// Still streaming.
			sb.WriteString(toolDimStyle.Render(" (summarizing…)"))
		case m.collapsedHandles[item.handle]:
			lines := strings.Split(item.content, "\n")
			const maxShown = 2
			shown := lines
			if len(lines) > maxShown {
				shown = lines[:maxShown]
			}
			sb.WriteString("\n")
			sb.WriteString(strings.Join(shown, "\n"))
			if len(lines) > maxShown {
				sb.WriteString("\n")
				sb.WriteString(toolDimStyle.Render(fmt.Sprintf(
					"  … %d more lines (use /expand %s to show all)",
					len(lines)-maxShown, item.handle,
				)))
			}
		default:
			sb.WriteString("\n")
			sb.WriteString(renderMarkdown(m.renderer, item.content))
		}
		return sb.String()

	case itemAskUser:
		var sb strings.Builder
		question := m.askUserQuestion
		// Wrap the question text to the terminal width, leaving room for the ❓ prefix (3 chars + 2 spaces = 5).
		if m.width > 9 {
			wrapped := wordwrap.String(question, max(1, m.width-7))
			// Indent continuation lines to align under the first word of the question.
			indent := strings.Repeat(" ", 5) // "❓  " prefix width
			question = strings.ReplaceAll(wrapped, "\n", "\n"+indent)
		}
		sb.WriteString(approvalPromptStyle.Render("❓") + "  " + question)
		for i, choice := range m.askUserChoices {
			sb.WriteString("\n")
			rec := choice == m.askUserRecommended
			if i == m.askUserSelectedIdx {
				line := choiceSelectedStyle.Render(fmt.Sprintf("  ▶ %d. %s", i+1, choice))
				if rec {
					line += "  " + choiceRecommendedStyle.Render("(Recommended)")
				}
				sb.WriteString(line)
			} else {
				line := fmt.Sprintf("  %d. %s", i+1, choice)
				if rec {
					line += "  " + choiceRecommendedStyle.Render("(Recommended)")
				}
				sb.WriteString(line)
			}
		}
		if m.askUserAllowFreeform {
			if m.askUserSelectedIdx == -1 {
				sb.WriteString("\n  " + choiceSelectedStyle.Render("▶ Custom answer"))
			} else {
				sb.WriteString("\n  " + statusStyle.Render("(or type a custom answer)"))
			}
		}
		return sb.String()

	default:
		return item.content
	}
}

func (m Model) registryView() string {
	if m.registryTab == 1 {
		return m.registryInstalledList.View()
	}
	return m.registrySearchInput.View() + "\n" + m.registryPicker.View()
}

// agentWizardView renders the full-screen agent creation wizard.
func (m Model) agentWizardView() string {
	var b strings.Builder
	titleStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("12"))
	hintStyle := statusStyle
	errStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("9"))

	switch m.state {
	case stateAgentWizardMode:
		b.WriteString(titleStyle.Render("Create new agent") + "\n\n")
		b.WriteString("How would you like to define this agent?\n\n")
		b.WriteString("  [a]  AI-assisted  -- describe the agent in plain language and let the AI generate the profile\n")
		b.WriteString("  [m]  Manual       -- fill in name, description, when-to-use, and instructions manually\n\n")
		b.WriteString(hintStyle.Render("Press [a] or [m].  [esc] cancel"))

	case stateAgentWizardDescribe:
		b.WriteString(titleStyle.Render("AI-assisted agent creation") + "\n\n")
		b.WriteString("Describe the agent's expertise and when it should be used.\n")
		b.WriteString("Example: \"A Go expert who reviews code for idiomatic style, performance, and correctness. Use when reviewing Go pull requests.\"\n\n")
		b.WriteString(m.agentWizardDescribeTA.View())
		b.WriteString("\n\n")
		b.WriteString(hintStyle.Render("[ctrl+s] generate  [esc] cancel"))

	case stateAgentWizardGenerate:
		b.WriteString(titleStyle.Render("Generating agent profile...") + "\n\n")
		b.WriteString(m.spinner.View() + "  Asking the AI to create the agent profile. Please wait...\n\n")
		b.WriteString(hintStyle.Render("[esc] cancel"))

	case stateAgentWizardReview:
		b.WriteString(titleStyle.Render("Review generated profile") + "\n\n")
		if m.agentWizardGenErr != "" {
			b.WriteString(errStyle.Render("Error: "+m.agentWizardGenErr) + "\n\n")
			if m.agentWizardGenerated.Name == "" {
				b.WriteString(hintStyle.Render("[r] retry  [esc] cancel"))
				break
			}
		}
		b.WriteString(m.agentWizardReviewText + "\n\n")
		if m.agentWizardOverwrite {
			b.WriteString(errStyle.Render("File already exists.") + "\n\n")
		}
		b.WriteString(hintStyle.Render("[c] continue to tools  [e] edit  [r] retry  [esc] cancel"))

	case stateAgentWizardManual:
		b.WriteString(titleStyle.Render("Manual agent definition") + "\n\n")
		labels := []string{"Name:", "Description:", "When to use:"}
		for i := 0; i < 3; i++ {
			active := m.agentWizardManualIdx == i
			prefix := "  "
			if active {
				prefix = "> "
			}
			b.WriteString(prefix + labels[i] + "\n")
			b.WriteString("  " + m.agentWizardManualTAs[i].View() + "\n\n")
		}
		instrActive := m.agentWizardManualIdx == 3
		instrPrefix := "  "
		if instrActive {
			instrPrefix = "> "
		}
		b.WriteString(instrPrefix + "Instructions:\n")
		b.WriteString("  " + m.agentWizardManualTA.View() + "\n\n")
		if m.agentWizardGenErr != "" {
			b.WriteString(errStyle.Render(m.agentWizardGenErr) + "\n")
		}
		b.WriteString(hintStyle.Render("[tab] next field  [shift+tab] previous  [tab] from instructions = submit  [esc] cancel"))

	case stateAgentWizardTools:
		return m.agentWizardToolPicker.View()

	case stateAgentWizardDone:
		b.WriteString(titleStyle.Render("Agent saved!") + "\n\n")
		fmt.Fprintf(&b, "Agent %q has been saved to:\n  %s\n\n", m.agentWizardGenerated.Name, m.agentWizardSavedPath)
		b.WriteString(hintStyle.Render("Press any key to continue"))
	}

	return b.String()
}

// toolDetailView renders the full-screen detail view for a single tool.
func (m Model) toolDetailView() string {
	title := headerStyle.Render(toolFriendlyName(m.toolDetailItem.name))
	raw := toolDimStyle.Render("[" + m.toolDetailItem.name + "]")
	hint := statusStyle.Render("  ↑/↓ scroll  ·  esc/enter back")
	header := title + " " + raw + hint
	return header + "\n" + m.toolDetailVP.View()
}

// buildToolDetail constructs the scrollable content for the tool detail viewport.
func buildToolDetail(t toolItem, width int) string {
	var b strings.Builder

	server := toolServerName(t.name)
	b.WriteString(statusStyle.Render("Server: ") + server + "\n\n")

	if t.description != "" {
		b.WriteString(t.description + "\n\n")
	}

	schema, ok := t.inputSchema.(map[string]any)
	if !ok || schema == nil {
		b.WriteString(statusStyle.Render("(no parameters)") + "\n")
		return b.String()
	}

	props, hasProps := schema["properties"].(map[string]any)
	if !hasProps || len(props) == 0 {
		b.WriteString(statusStyle.Render("(no parameters)") + "\n")
		return b.String()
	}

	required := map[string]bool{}
	if req, ok := schema["required"].([]any); ok {
		for _, r := range req {
			if s, ok := r.(string); ok {
				required[s] = true
			}
		}
	}

	b.WriteString(toolNameStyle.Render("Parameters") + "\n")

	// Sort: required first, then optional, each group alphabetically.
	reqNames := make([]string, 0, len(props))
	optNames := make([]string, 0, len(props))
	for k := range props {
		if required[k] {
			reqNames = append(reqNames, k)
		} else {
			optNames = append(optNames, k)
		}
	}
	slices.Sort(reqNames)
	slices.Sort(optNames)

	for _, name := range append(reqNames, optNames...) {
		prop, _ := props[name].(map[string]any)
		typeStr, _ := prop["type"].(string)
		desc, _ := prop["description"].(string)
		enumVals, _ := prop["enum"].([]any)

		req := ""
		if !required[name] {
			req = statusStyle.Render(" (optional)")
		}
		line := "  " + toolNameStyle.Render(name) + req
		if typeStr != "" {
			line += "  " + toolDimStyle.Render(typeStr)
		}
		b.WriteString(line + "\n")

		if len(enumVals) > 0 {
			enumStrs := make([]string, 0, len(enumVals))
			for _, v := range enumVals {
				enumStrs = append(enumStrs, fmt.Sprintf("%v", v))
			}
			b.WriteString("    " + toolDimStyle.Render("one of: "+strings.Join(enumStrs, ", ")) + "\n")
		}
		if desc != "" {
			wrapped := wordwrap.String(desc, width-4)
			for _, l := range strings.Split(wrapped, "\n") {
				b.WriteString("    " + l + "\n")
			}
		}
	}

	return b.String()
}

func buildInstalledItems(servers []config.MCPServerConfig) []list.Item {
	items := make([]list.Item, len(servers))
	for i, srv := range servers {
		items[i] = registryInstalledItem{srv}
	}
	return items
}

// renderToolName renders a tool name as "Friendly Name [raw_name]" where the
// raw name is grayed out. Use this wherever tool names are shown to the user.
func renderToolName(name string) string {
	friendly := toolFriendlyName(name)
	return toolNameStyle.Render(friendly) + " " + toolDimStyle.Render("["+name+"]")
}

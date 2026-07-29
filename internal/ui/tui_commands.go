package ui

import (
	"context"
	"fmt"
	"strings"
	"time"

	"charm.land/bubbles/v2/list"
	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
)

// slashCommands is the ordered registry of all /commands.
// Initialized in init() to avoid self-referencing the slice in the /help action.
var slashCommands []slashCommand

func init() {
	slashCommands = []slashCommand{
		{
			name:        "model",
			description: "Switch the active AI model",
			available:   func(m *Model) bool { return m.modelManager != nil },
			action: func(m *Model) []tea.Cmd {
				return []tea.Cmd{m.openModelPicker(m.state)}
			},
		},
		{
			name:        "tools",
			description: "Enable or disable individual tools for this session",
			action: func(m *Model) []tea.Cmd {
				m.state = statePickingTools
				m.toolPicker.SetItems(nil)
				m.toolPicker.SetSize(m.width, m.height-fixedLines)
				return []tea.Cmd{doListAllTools(m.ctx, m.session)}
			},
		},
		{
			name:        "skills",
			description: "Enable or disable individual skills for this session",
			available:   func(m *Model) bool { return len(m.skills) > 0 },
			action: func(m *Model) []tea.Cmd {
				m.state = statePickingSkills
				items := make([]list.Item, len(m.skills))
				for i, s := range m.skills {
					items[i] = skillItem{
						name:        s.Name,
						description: s.Description,
						enabled:     !m.disabledSkills[s.Name],
					}
				}
				m.skillPicker.SetItems(items)
				m.skillPicker.SetSize(m.width, m.height-fixedLines)
				return nil
			},
		},
		{
			name:        "clear",
			description: "Clear the conversation history",
			action: func(m *Model) []tea.Cmd {
				m.messages = m.newConversation()
				m.items = nil
				m.toolCallIdx = make(map[string]int)
				m.streamingItemIdx = -1
				m.reasoningItemIdx = -1
				m.oauthInfoIdx = -1
				m.rebuildContent()
				return nil
			},
		},
		{
			name:        "new",
			description: "Start a new session (clears history and detaches from the current saved session)",
			action: func(m *Model) []tea.Cmd {
				m.messages = m.newConversation()
				m.items = nil
				m.toolCallIdx = make(map[string]int)
				m.streamingItemIdx = -1
				m.reasoningItemIdx = -1
				m.oauthInfoIdx = -1
				m.sessionID = ""
				m.sessionCreatedAt = time.Time{}
				m.disabledSkills = make(map[string]bool)
				if m.toolMgr != nil {
					m.toolMgr.SetSkills(m.skills)
				}
				if m.todoStore != nil {
					m.todoStore.Clear()
					m.sidebarOpen = false
					m.recalcLayout()
				}
				m.rebuildContent()
				return nil
			},
		},
		{
			name:        "compact",
			description: "Summarize the conversation to free up context window space",
			action: func(m *Model) []tea.Cmd {
				m.state = stateCompacting
				return []tea.Cmd{doCompact(m.newOpCtx(), m.client, m.messages, m.modelName, m.toolTokensCache, m.modelInfo.Context.MaxTokens)}
			},
		},
		{
			name:        "registry",
			description: "Browse and add MCP servers from the Model Context Protocol registry",
			action: func(m *Model) []tea.Cmd {
				m.state = statePickingRegistry
				m.registryTab = 0

				// Snapshot installed names.
				m.registryInstalledNames = make(map[string]bool, len(m.configuredMCPServers))
				for _, srv := range m.configuredMCPServers {
					m.registryInstalledNames[srv.Name] = true
				}

				// Browse list (live-search; no local filter).
				m.registryPicker = list.New(nil, list.NewDefaultDelegate(), m.width, m.height-fixedLines-2)
				m.registryPicker.Title = "MCP Registry  [Browse | Tab → Installed]  — Enter to add · Esc to close"
				m.registryPicker.SetShowStatusBar(true)
				m.registryPicker.SetFilteringEnabled(false)
				m.registryPicker.DisableQuitKeybindings()

				// Search input.
				si := textinput.New()
				si.Placeholder = "type to search registry…"
				si.Prompt = "Search: "
				si.CharLimit = 100
				si.SetWidth(m.width - 12)
				si.Focus()
				m.registrySearchInput = si

				// Installed list.
				m.registryInstalledList = list.New(
					buildInstalledItems(m.configuredMCPServers),
					list.NewDefaultDelegate(),
					m.width,
					m.height-fixedLines,
				)
				m.registryInstalledList.Title = "Installed MCP Servers  [Tab → Browse]  — Ctrl+D to remove · Esc to close"
				m.registryInstalledList.SetShowStatusBar(true)
				m.registryInstalledList.SetFilteringEnabled(true)
				m.registryInstalledList.DisableQuitKeybindings()

				ctx, cancel := context.WithCancel(m.ctx)
				m.registryCancelCtx = cancel
				m.registrySearchSeq++
				return []tea.Cmd{doFetchRegistry(ctx, m.registrySearchSeq, "", "")}
			},
		},
		{
			name:          "todos",
			description:   "Toggle the todo sidebar",
			available:     func(m *Model) bool { return m.todoStore != nil },
			safeWhileBusy: true,
			action: func(m *Model) []tea.Cmd {
				m.sidebarOpen = !m.sidebarOpen
				m.recalcLayout()
				m.rebuildContent()
				return nil
			},
		},
		{
			name:          "autopilot",
			description:   "Toggle autopilot mode — AI works autonomously until task_complete is called",
			safeWhileBusy: true,
			action: func(m *Model) []tea.Cmd {
				if m.autopilot || m.autopilotPaused {
					m.autopilotDisable()
					m.items = append(m.items, displayItem{kind: itemInfo, content: "⚡ Autopilot disabled."})
				} else {
					m.autopilotEnable()
					m.items = append(m.items, displayItem{kind: itemInfo, content: fmt.Sprintf("⚡ Autopilot enabled (cap: %d cycles). The AI will work autonomously.", m.effectiveAutopilotMax())})
				}
				m.rebuildContent()
				return nil
			},
		},
		{
			name:          "allow-all",
			description:   "Toggle allow-all mode — auto-approve all tool calls and path access without prompting",
			safeWhileBusy: true,
			action: func(m *Model) []tea.Cmd {
				m.session.SetAllowAll(!m.session.AllowAll())
				m.rebuildContent()
				return nil
			},
		},
		{
			name:          "reasoning",
			description:   "Toggle display of model reasoning/thinking content",
			safeWhileBusy: true,
			action: func(m *Model) []tea.Cmd {
				m.showReasoning = !m.showReasoning
				if m.showReasoning {
					// Check whether any reasoning content has been received so far.
					hasContent := false
					for _, it := range m.items {
						if it.kind == itemReasoning && it.content != "" {
							hasContent = true
							break
						}
					}
					msg := "💭 Reasoning display: on"
					if !hasContent {
						msg += " (no thinking content received yet — the active model may not emit reasoning tokens)"
					}
					m.items = append(m.items, displayItem{kind: itemInfo, content: msg})
				} else {
					m.items = append(m.items, displayItem{kind: itemInfo, content: "💭 Reasoning display: off"})
				}
				m.rebuildContent()
				return nil
			},
		},
		{
			name:        "mode",
			description: "Switch the active mode preset (default, plan, document, or custom)",
			available:   func(m *Model) bool { return len(m.allModes) > 0 },
			action: func(m *Model) []tea.Cmd {
				items := make([]list.Item, len(m.allModes))
				for i, mode := range m.allModes {
					items[i] = modeItem{mode: mode, active: mode.Name == m.activeMode.Name}
				}
				m.modePicker.SetItems(items)
				m.modePicker.SetSize(m.width, m.height-fixedLines)
				m.state = statePickingMode
				return nil
			},
		},
		{
			name:          "image",
			description:   "Attach an image to your next message — usage: /image <path-or-url>",
			safeWhileBusy: true,
			action: func(m *Model) []tea.Cmd {
				// Fill the input with "/image " so the user can type the path/URL.
				m.input.SetValue("/image ")
				m.input.CursorEnd()
				m.updateCompletion()
				return nil
			},
		},
		{
			name:          "quit",
			description:   "Quit werkler",
			safeWhileBusy: true,
			action: func(m *Model) []tea.Cmd {
				return []tea.Cmd{tea.Quit}
			},
		},
		{
			name:        "agent",
			description: "Create, activate, or deactivate a custom agent  (usage: /agent new | /agent <name> | /agent off)",
			action: func(m *Model) []tea.Cmd {
				return m.initAgentWizard()
			},
		},
		{
			name:          "expand",
			description:   "Expand collapsed process output  (usage: /expand <handle> | /expand all)",
			safeWhileBusy: true,
			action: func(m *Model) []tea.Cmd {
				// Without an argument, list collapsed handles.
				var collapsed []string
				for h, c := range m.collapsedHandles {
					if c {
						collapsed = append(collapsed, h)
					}
				}
				if len(collapsed) == 0 {
					m.items = append(m.items, displayItem{kind: itemInfo, content: "No collapsed process outputs."})
				} else {
					m.items = append(m.items, displayItem{kind: itemInfo, content: fmt.Sprintf("Collapsed handles: %s  (use /expand <handle> or /expand all)", strings.Join(collapsed, ", "))})
				}
				m.rebuildContent()
				return nil
			},
		},
		{
			name:          "collapse",
			description:   "Collapse expanded process output  (usage: /collapse <handle> | /collapse all)",
			safeWhileBusy: true,
			action: func(m *Model) []tea.Cmd {
				// Without an argument, collapse all expanded handles.
				for h := range m.collapsedHandles {
					m.collapsedHandles[h] = true
				}
				m.rebuildContent()
				return nil
			},
		},
		{
			name:          "sidebar",
			description:   "Resize the todo sidebar  (usage: /sidebar wider | /sidebar narrower | /sidebar reset)",
			safeWhileBusy: true,
			available:     func(m *Model) bool { return m.todoStore != nil },
			action: func(m *Model) []tea.Cmd {
				m.items = append(m.items, displayItem{kind: itemInfo, content: fmt.Sprintf("Sidebar width: %d. Use /sidebar wider, /sidebar narrower, or /sidebar reset.", m.sidebarWidth)})
				m.rebuildContent()
				return nil
			},
		},
		{
			name:          "help",
			description:   "Show available keyboard shortcuts and commands",
			safeWhileBusy: true,
			action: func(m *Model) []tea.Cmd {
				var lines []string
				lines = append(lines, "**Keyboard shortcuts**")
				lines = append(lines, "- `ctrl+c` / `ctrl+d` — quit")
				lines = append(lines, "- `ctrl+p` — switch model (when available)")
				lines = append(lines, "- `alt+m` — toggle mouse reporting (on = scroll wheel; off = terminal text selection)")
				lines = append(lines, "- `↑/↓ pgup/pgdn` — scroll conversation history")
				lines = append(lines, "")
				lines = append(lines, "**Tool approval keys** (when prompted)")
				lines = append(lines, "- press a key to stage a choice, then `Enter` to confirm:")
				lines = append(lines, "- `y` → allow once  `d` → allow directory  `a` → allow for session  `p` → allow permanently  `n` → deny")
				lines = append(lines, "- `Esc` — clear staged choice")
				lines = append(lines, "")
				lines = append(lines, "**Slash commands** — type `/` to see autocomplete")
				for _, cmd := range slashCommands {
					lines = append(lines, fmt.Sprintf("- `/%s` — %s", cmd.name, cmd.description))
				}
				for _, ext := range externalCommands {
					lines = append(lines, fmt.Sprintf("- `/%s` — %s", ext.Name, ext.Description))
				}
				m.items = append(m.items, displayItem{
					kind:    itemMarkdown,
					content: strings.Join(lines, "\n"),
				})
				m.rebuildContent()
				return nil
			},
		},
	}
}

// and whose available predicate (if any) passes.
func (m Model) filteredCmds() []slashCommand {
	text := m.input.Value()
	if !strings.HasPrefix(text, "/") {
		return nil
	}
	busy := m.isBusy()
	prefix := strings.TrimPrefix(text, "/")
	var out []slashCommand
	for _, cmd := range slashCommands {
		if busy && !cmd.safeWhileBusy {
			continue
		}
		if cmd.available != nil && !cmd.available(&m) {
			continue
		}
		if strings.HasPrefix(cmd.name, prefix) {
			out = append(out, cmd)
		}
	}
	// Include matching external commands.
	for _, ext := range externalCommands {
		if busy && !ext.SafeWhileBusy {
			continue
		}
		if ext.Available != nil && !ext.Available() {
			continue
		}
		if strings.HasPrefix(ext.Name, prefix) {
			out = append(out, wrapExternal(ext, ""))
		}
	}
	return out
}

// updateCompletion derives showCompletion from the current state and input value,
// clamps completionIdx, and syncs the viewport height. Call after any input or
// state change that might affect completion visibility.
func (m *Model) updateCompletion() {
	wasShowing := m.showCompletion
	cmds := m.filteredCmds()
	m.showCompletion = (m.state == stateIdle || m.isBusy()) && len(cmds) > 0
	if !wasShowing && m.showCompletion {
		m.completionIdx = 0 // reset selection on fresh open
	} else if m.completionIdx >= len(cmds) {
		m.completionIdx = max(0, len(cmds)-1)
	}
	m.syncViewportHeight()
}

// runCompletion executes the action of the currently selected completion item.
// It resets the input, hides the completion popup, and returns the resulting cmds.
func (m *Model) runCompletion(selected slashCommand) []tea.Cmd {
	m.input.Reset()
	m.showCompletion = false
	m.syncViewportHeight()
	return selected.action(m)
}

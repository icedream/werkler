package ui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/tools"
)

// fixedLines is the number of terminal lines consumed by non-viewport UI elements:
// header(1) + sep(1) + sep(1) + statusLine1(1) + statusLine2(1) + sep(1) + input(1) = 7
const fixedLines = 7

// --- Debug logging ---

var debugLogger *log.Logger

func debugLog(format string, args ...any) {
	if debugLogger != nil {
		debugLogger.Printf(format, args...)
	}
}

// --- TUI states ---

type tuiState int

const (
	stateIdle      tuiState = iota
	stateThinking           // waiting for first token
	stateStreaming          // tokens arriving
	stateCallingTool
	stateAwaitingApproval
	stateAwaitingPathApproval // waiting for user to approve a path access
	stateConnectingOAuth      // running deferred OAuth server connections
	statePickingModel         // model selection list is open
)

// inputPlaceholder returns the appropriate placeholder text for the input
// based on the current TUI state.
func inputPlaceholder(state tuiState) string {
	switch state {
	case stateAwaitingApproval:
		return "Approve the tool call above (y / n / a)…"
	case stateAwaitingPathApproval:
		return "Approve path access (y / n / a)…"
	case stateConnectingOAuth:
		return "Waiting for authorization in browser…"
	case statePickingModel:
		return ""
	case stateIdle:
		return "Type a message, press Enter to send…"
	default:
		return "Queue a follow-up, press Enter…"
	}
}

// --- Display item kinds and statuses ---

const (
	itemUser          = "user"
	itemAssistant     = "assistant"
	itemToolCall      = "tool_call"
	itemError         = "error"
	itemInfo          = "info"        // neutral status/system messages
	itemProcessOutput = "proc_output" // live process output streamed into viewport
)

const (
	toolStatusPending = iota
	toolStatusRunning
	toolStatusDone
	toolStatusFailed
	toolStatusDenied
)

type displayItem struct {
	kind       string
	content    string
	toolName   string
	toolArgs   string // compact JSON args
	toolStatus int
	handle     string // process handle (itemProcessOutput only)
}

// --- Tea messages ---

// streamChunkMsg carries one StreamChunk from the AI stream, plus the channel
// so Update can dispatch the next read without storing the channel in the model.
type streamChunkMsg struct {
	ch    <-chan ai.StreamChunk
	chunk ai.StreamChunk
}

type toolResultMsg struct {
	callID   string
	toolName string
	result   string
	err      error
}

type contextDoneMsg struct{}

// --- OAuth messages ---

// oauthNeedAuthMsg is sent when an OAuth server requires browser authorization.
// The TUI updates the info item in the viewport to show the URL.
type oauthNeedAuthMsg struct {
	serverName string
	authURL    string
}

// oauthConnectedMsg is sent when all pending OAuth servers have been connected.
type oauthConnectedMsg struct{}

// oauthConnectFailedMsg is sent when OAuth connection fails.
type oauthConnectFailedMsg struct{ err error }

// --- Model picker messages ---

type modelsLoadedMsg struct{ models []string }
type modelsErrMsg struct{ err error }

// processOutputMsg carries new output from a running process for live display.
type processOutputMsg struct {
	handle string
	raw    string // ANSI-preserved for display
	clean  string // ANSI-stripped (used for AI, not displayed)
}

// modelItem implements list.Item for a model name string.
type modelItem string

func (m modelItem) FilterValue() string { return string(m) }
func (m modelItem) Title() string       { return string(m) }
func (m modelItem) Description() string { return "" }

// --- Slash commands ---

// slashCommand describes a /command available in the input box.
type slashCommand struct {
	name        string // without leading slash, e.g. "model"
	description string
	available   func(*Model) bool      // nil = always available
	action      func(*Model) []tea.Cmd // executed on selection
}

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
				m.state = statePickingModel
				m.modelPicker.SetItems(nil)
				m.modelPicker.SetSize(m.width, m.height-fixedLines)
				return []tea.Cmd{doListModels(m.ctx, m.modelManager)}
			},
		},
		{
			name:        "clear",
			description: "Clear the conversation history",
			action: func(m *Model) []tea.Cmd {
				m.messages = chat.NewConversation()
				m.items = nil
				m.toolCallIdx = make(map[string]int)
				m.streamingItemIdx = -1
				m.oauthInfoIdx = -1
				m.rebuildContent()
				return nil
			},
		},
		{
			name:        "quit",
			description: "Quit werkler",
			action: func(m *Model) []tea.Cmd {
				return []tea.Cmd{tea.Quit}
			},
		},
		{
			name:        "help",
			description: "Show available keyboard shortcuts and commands",
			action: func(m *Model) []tea.Cmd {
				var lines []string
				lines = append(lines, "**Keyboard shortcuts**")
				lines = append(lines, "- `ctrl+c` / `ctrl+d` — quit")
				lines = append(lines, "- `ctrl+p` — switch model (when available)")
				lines = append(lines, "- `alt+m` — toggle mouse reporting (off = text selection)")
				lines = append(lines, "- `↑/↓ pgup/pgdn` — scroll conversation history")
				lines = append(lines, "")
				lines = append(lines, "**Slash commands** — type `/` to see autocomplete")
				for _, cmd := range slashCommands {
					lines = append(lines, fmt.Sprintf("- `/%s` — %s", cmd.name, cmd.description))
				}
				m.items = append(m.items, displayItem{
					kind:    itemInfo,
					content: strings.Join(lines, "\n"),
				})
				m.rebuildContent()
				return nil
			},
		},
	}
}

// --- Model ---

// Model is the bubbletea model for the interactive TUI.
type Model struct {
	ctx     context.Context
	client  ai.StreamCompleter
	session *chat.Session
	tools   []ai.ToolDefinition

	// modelManager is optionally set when the client also implements ModelManager.
	// When nil, the model picker feature is disabled.
	modelManager ai.ModelManager

	// send dispatches messages to this bubbletea program from goroutines.
	// Set in RunTUI before the program starts.
	send func(tea.Msg)

	// Conversation history (managed only in Update, never in tea.Cmd goroutines).
	messages []ai.Message

	// Agent state machine.
	state            tuiState
	pendingCalls     []ai.ToolCall
	currentCall      *ai.ToolCall
	callingToolName  string // name of tool currently executing (stateCallingTool)
	streamingItemIdx int    // index into items of the in-progress assistant item; -1 if none

	// Path approval state (stateAwaitingPathApproval).
	pendingPathApprovals  []chat.PathAccessRequest // remaining paths needing approval
	currentPathRequest    chat.PathAccessRequest   // request currently shown in the dialog
	pendingCallAfterPaths *ai.ToolCall             // tool call to dispatch once all paths are approved

	// executingCall is the tool call currently running in a doCallTool goroutine.
	// Set just before doCallTool; cleared when toolResultMsg arrives.
	// Distinct from currentCall, which is the call awaiting user confirmation.
	executingCall *ai.ToolCall

	// Queue of user prompts entered while the AI is busy.
	// Processed FIFO after the current agent turn completes successfully.
	queuedPrompts []string

	// Display items for the viewport.
	items        []displayItem
	toolCallIdx  map[string]int // callID → index in items
	oauthInfoIdx int            // index of the OAuth status item; -1 if none

	// Model picker (only valid during statePickingModel).
	modelPicker list.Model

	// UI components.
	viewport     viewport.Model
	input        textinput.Model
	spinner      spinner.Model
	renderer     *glamour.TermRenderer
	glamourStyle string // resolved once before TUI starts (dark/light)

	// Terminal dimensions.
	width  int
	height int

	// Header metadata.
	modelName   string
	serverNames []string

	// mouseEnabled tracks whether the terminal mouse reporting mode is active.
	// When false, the terminal handles mouse events natively (text selection works).
	mouseEnabled bool

	// Slash-command autocomplete state.
	// showCompletion is derived: true when state==stateIdle and input starts with "/".
	// completionIdx is the currently highlighted item in the completion popup.
	showCompletion bool
	completionIdx  int
}

func initialModel(
	ctx context.Context,
	client ai.StreamCompleter,
	session *chat.Session,
	tools []ai.ToolDefinition,
	modelName string,
	serverNames []string,
	glamourStyle string,
	send func(tea.Msg),
) Model {
	sp := spinner.New()
	sp.Spinner = thinkingSpinner

	ti := textinput.New()
	ti.Placeholder = inputPlaceholder(stateIdle)
	ti.Prompt = ""
	ti.CharLimit = 0
	ti.Focus() // must be called here; Init() runs on a value copy so mutations there are lost

	// Model picker: initialized at zero size; sized on first WindowSizeMsg.
	delegate := list.NewDefaultDelegate()
	picker := list.New(nil, delegate, 0, 0)
	picker.Title = "Select model"
	picker.SetShowStatusBar(false)
	picker.SetFilteringEnabled(true)
	picker.DisableQuitKeybindings()

	// Use modelManager when the client also implements ModelManager.
	var mm ai.ModelManager
	if m, ok := client.(ai.ModelManager); ok {
		mm = m
	}

	return Model{
		ctx:              ctx,
		client:           client,
		session:          session,
		tools:            tools,
		send:             send,
		modelManager:     mm,
		messages:         chat.NewConversation(),
		state:            stateIdle,
		streamingItemIdx: -1,
		oauthInfoIdx:     -1,
		toolCallIdx:      make(map[string]int),
		viewport:         viewport.New(0, 0),
		modelPicker:      picker,
		input:            ti,
		spinner:          sp,
		modelName:        modelName,
		serverNames:      serverNames,
		glamourStyle:     glamourStyle,
		mouseEnabled:     true,
	}
}

// --- Slash-command helpers ---

// filteredCmds returns slash commands whose names start with the given prefix
// and whose available predicate (if any) passes.
func (m Model) filteredCmds() []slashCommand {
	text := m.input.Value()
	if !strings.HasPrefix(text, "/") {
		return nil
	}
	prefix := strings.TrimPrefix(text, "/")
	var out []slashCommand
	for _, cmd := range slashCommands {
		if cmd.available != nil && !cmd.available(&m) {
			continue
		}
		if strings.HasPrefix(cmd.name, prefix) {
			out = append(out, cmd)
		}
	}
	return out
}

// completionLineCount returns the number of extra terminal lines consumed by
// the completion popup. Each item is guaranteed to fit on one line (truncated
// to m.width in completionView).
func (m Model) completionLineCount() int {
	if !m.showCompletion {
		return 0
	}
	return len(m.filteredCmds())
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

// updateCompletion derives showCompletion from the current state and input value,
// clamps completionIdx, and syncs the viewport height. Call after any input or
// state change that might affect completion visibility.
func (m *Model) updateCompletion() {
	wasShowing := m.showCompletion
	cmds := m.filteredCmds()
	m.showCompletion = m.state == stateIdle && len(cmds) > 0
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

// --- Tea interface ---

func (m Model) Init() tea.Cmd {
	return tea.Batch(
		textinput.Blink, // cursor blink; Focus() state is set in initialModel
		m.spinner.Tick,
		watchContext(m.ctx),
	)
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	needRebuild := false

	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			return m, tea.Quit
		}

		// Alt+M toggles terminal mouse reporting so the user can select and copy text.
		if msg.String() == "alt+m" {
			if m.mouseEnabled {
				m.mouseEnabled = false
				return m, tea.DisableMouse
			}
			m.mouseEnabled = true
			return m, tea.EnableMouseCellMotion
		}

		switch m.state {
		case stateAwaitingPathApproval:
			if m.currentPathRequest.Path != "" {
				switch msg.String() {
				case "y", "Y":
					m.approvePathRequest(m.currentPathRequest)
					m.currentPathRequest = chat.PathAccessRequest{}
					cmds = append(cmds, m.processNextPath())
					needRebuild = true
				case "a", "A":
					// Approve current and all remaining paths at once.
					m.approvePathRequest(m.currentPathRequest)
					for _, req := range m.pendingPathApprovals {
						m.approvePathRequest(req)
					}
					m.pendingPathApprovals = nil
					m.currentPathRequest = chat.PathAccessRequest{}
					cmds = append(cmds, m.processNextPath())
					needRebuild = true
				case "n", "N":
					// Deny: clear path queue, deny the pending tool call.
					m.pendingPathApprovals = nil
					m.currentPathRequest = chat.PathAccessRequest{}
					if m.pendingCallAfterPaths != nil {
						call := *m.pendingCallAfterPaths
						m.pendingCallAfterPaths = nil
						if idx, ok := m.toolCallIdx[call.ID]; ok {
							m.items[idx].toolStatus = toolStatusDenied
						}
						m.messages = append(m.messages, ai.Message{
							Role:       "tool",
							ToolCallID: call.ID,
							Content:    "(tool call was denied — path access was not approved)",
						})
						m.currentCall = nil
						nextCmd := m.processNextCall()
						needRebuild = true
						cmds = append(cmds, nextCmd)
					} else {
						m.state = stateIdle
					}
				default:
					var vpCmd tea.Cmd
					m.viewport, vpCmd = m.viewport.Update(msg)
					cmds = append(cmds, vpCmd)
				}
			}

		case stateAwaitingApproval:
			if m.currentCall != nil {
				switch msg.String() {
				case "y", "Y":
					call := *m.currentCall
					if idx, ok := m.toolCallIdx[call.ID]; ok {
						m.items[idx].toolStatus = toolStatusRunning
					}
					m.callingToolName = call.Name
					m.currentCall = nil
					m.executingCall = &call
					m.state = stateCallingTool
					needRebuild = true
					cmds = append(cmds, doCallTool(m.ctx, m.session, call))
				case "a", "A":
					call := *m.currentCall
					m.session.ApproveForSession(call.Name)
					if idx, ok := m.toolCallIdx[call.ID]; ok {
						m.items[idx].toolStatus = toolStatusRunning
					}
					m.callingToolName = call.Name
					m.currentCall = nil
					m.executingCall = &call
					m.state = stateCallingTool
					needRebuild = true
					cmds = append(cmds, doCallTool(m.ctx, m.session, call))
				case "n", "N":
					call := *m.currentCall
					if idx, ok := m.toolCallIdx[call.ID]; ok {
						m.items[idx].toolStatus = toolStatusDenied
					}
					m.messages = append(m.messages, ai.Message{
						Role:       "tool",
						ToolCallID: call.ID,
						Content:    "(tool call was denied by the user)",
					})
					m.currentCall = nil
					nextCmd := m.processNextCall()
					needRebuild = true
					cmds = append(cmds, nextCmd)
				default:
					var vpCmd tea.Cmd
					m.viewport, vpCmd = m.viewport.Update(msg)
					cmds = append(cmds, vpCmd)
				}
			}

		case stateIdle:
			switch msg.Type {
			case tea.KeyEnter:
				if m.showCompletion {
					// Execute exact match or fill partial match.
					filtered := m.filteredCmds()
					if len(filtered) > 0 {
						selected := filtered[m.completionIdx]
						if m.input.Value() == "/"+selected.name {
							// Exact match — run the command.
							cmds = append(cmds, m.runCompletion(selected)...)
							m.updateCompletion()
						} else {
							// Partial match — fill the input without executing.
							m.input.SetValue("/" + selected.name)
							m.input.CursorEnd()
							m.updateCompletion()
						}
					}
				} else {
					text := strings.TrimSpace(m.input.Value())
					if text != "" {
						m.input.Reset()
						m.items = append(m.items, displayItem{kind: itemUser, content: text})
						needRebuild = true

						if m.session.HasPendingOAuth() {
							// Defer the user prompt until OAuth servers are connected.
							m.queuedPrompts = append([]string{text}, m.queuedPrompts...)
							m.oauthInfoIdx = len(m.items)
							names := strings.Join(m.session.PendingOAuthNames(), ", ")
							m.items = append(m.items, displayItem{
								kind:    itemInfo,
								content: "Connecting to " + names + "…",
							})
							m.state = stateConnectingOAuth
							cmds = append(cmds, doConnectOAuth(m.ctx, m.session, m.send))
						} else {
							m.messages = append(m.messages, ai.Message{Role: "user", Content: text})
							m.state = stateThinking
							cmds = append(cmds, doStartStream(m.ctx, m.client, m.messages, m.tools))
						}
					}
				}

			case tea.KeyTab:
				if m.showCompletion {
					// Tab fills the selected completion into the input.
					filtered := m.filteredCmds()
					if len(filtered) > 0 {
						m.input.SetValue("/" + filtered[m.completionIdx].name)
						m.input.CursorEnd()
						m.updateCompletion()
					}
				} else {
					var inputCmd tea.Cmd
					m.input, inputCmd = m.input.Update(msg)
					cmds = append(cmds, inputCmd)
				}

			case tea.KeyUp:
				if m.showCompletion {
					filtered := m.filteredCmds()
					if len(filtered) > 0 {
						m.completionIdx = (m.completionIdx - 1 + len(filtered)) % len(filtered)
					}
				} else {
					var vpCmd tea.Cmd
					m.viewport, vpCmd = m.viewport.Update(msg)
					cmds = append(cmds, vpCmd)
				}

			case tea.KeyDown:
				if m.showCompletion {
					filtered := m.filteredCmds()
					if len(filtered) > 0 {
						m.completionIdx = (m.completionIdx + 1) % len(filtered)
					}
				} else {
					var vpCmd tea.Cmd
					m.viewport, vpCmd = m.viewport.Update(msg)
					cmds = append(cmds, vpCmd)
				}

			case tea.KeyEsc:
				if m.showCompletion {
					m.showCompletion = false
					m.syncViewportHeight()
				} else if m.input.Value() != "" {
					m.input.Reset()
					m.updateCompletion()
				}

			case tea.KeyCtrlP:
				if m.modelManager != nil {
					m.showCompletion = false
					m.state = statePickingModel
					m.modelPicker.SetItems(nil)
					m.modelPicker.SetSize(m.width, m.height-fixedLines)
					cmds = append(cmds, doListModels(m.ctx, m.modelManager))
				}

			default:
				// Forward to textinput; do NOT forward to viewport when completion is
				// showing (Up/Down are captured above; other nav keys close completion).
				var inputCmd tea.Cmd
				m.input, inputCmd = m.input.Update(msg)
				cmds = append(cmds, inputCmd)
				m.updateCompletion()
				if !m.showCompletion {
					// Safe to forward scroll keys to viewport when completion is hidden.
					var vpCmd tea.Cmd
					m.viewport, vpCmd = m.viewport.Update(msg)
					cmds = append(cmds, vpCmd)
				}
			}

		case statePickingModel:
			switch msg.Type {
			case tea.KeyEsc, tea.KeyCtrlC:
				m.state = stateIdle
				m.updateCompletion()
			case tea.KeyEnter:
				if sel := m.modelPicker.SelectedItem(); sel != nil {
					model := string(sel.(modelItem))
					m.modelManager.SetModel(model)
					m.modelName = model
				}
				m.state = stateIdle
				m.updateCompletion()
			default:
				var pickerCmd tea.Cmd
				m.modelPicker, pickerCmd = m.modelPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			}

		case stateThinking, stateStreaming, stateCallingTool, stateConnectingOAuth:
			switch msg.Type {
			case tea.KeyEnter:
				text := strings.TrimSpace(m.input.Value())
				if text != "" {
					m.input.Reset()
					m.queuedPrompts = append(m.queuedPrompts, text)
					needRebuild = true
				}
			case tea.KeyEsc:
				if m.input.Value() != "" {
					m.input.Reset()
				} else if len(m.queuedPrompts) > 0 {
					m.queuedPrompts = m.queuedPrompts[:len(m.queuedPrompts)-1]
					needRebuild = true
				}
			default:
				// Forward to both: textinput handles text-entry keys (runes, backspace,
				// cursor movement); viewport handles navigation keys (Up/Down/PgUp/PgDn).
				// Single-line textinput ignores directional keys, so double-routing is safe.
				var inputCmd tea.Cmd
				m.input, inputCmd = m.input.Update(msg)
				cmds = append(cmds, inputCmd)
				var vpCmd tea.Cmd
				m.viewport, vpCmd = m.viewport.Update(msg)
				cmds = append(cmds, vpCmd)
			}
		}

	case tea.MouseMsg:
		// Forward all mouse events to the viewport so scroll wheel works in every state.
		var vpCmd tea.Cmd
		m.viewport, vpCmd = m.viewport.Update(msg)
		cmds = append(cmds, vpCmd)

	case modelsLoadedMsg:
		// Only process if we're still in model-picking state (guard against stale results).
		if m.state == statePickingModel {
			items := make([]list.Item, len(msg.models))
			selectedIdx := 0
			for i, name := range msg.models {
				items[i] = modelItem(name)
				if name == m.modelName {
					selectedIdx = i
				}
			}
			m.modelPicker.SetItems(items)
			m.modelPicker.Select(selectedIdx)
		}

	case modelsErrMsg:
		if m.state == statePickingModel {
			m.items = append(m.items, displayItem{kind: itemError, content: "listing models: " + msg.err.Error()})
			m.state = stateIdle
			m.updateCompletion()
			needRebuild = true
		}

	case processOutputMsg:
		// Live process output: append (or extend) a process output display item.
		// Find the last item for this handle; if it exists and is the last item,
		// extend it — otherwise create a new one.
		found := false
		if n := len(m.items); n > 0 {
			last := &m.items[n-1]
			if last.kind == itemProcessOutput && last.handle == msg.handle {
				last.content += msg.raw
				found = true
			}
		}
		if !found {
			m.items = append(m.items, displayItem{
				kind:    itemProcessOutput,
				handle:  msg.handle,
				content: msg.raw,
			})
		}
		needRebuild = true

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		// Forward to viewport so it can update its internal state, then
		// override the dimensions we want.
		var vpCmd tea.Cmd
		m.viewport, vpCmd = m.viewport.Update(msg)
		cmds = append(cmds, vpCmd)
		m.viewport.Width = m.width
		m.input.Width = m.width - 5 // 5 = len("You> ")
		m.modelPicker.SetSize(m.width, m.height-fixedLines)
		m.syncViewportHeight() // accounts for completion popup if visible
		m.renderer = newGlamourRenderer(m.width-4, m.glamourStyle)
		needRebuild = true
		// Clear and fully repaint after resize to avoid blank regions.
		cmds = append(cmds, tea.ClearScreen)

	case spinner.TickMsg:
		var spinCmd tea.Cmd
		m.spinner, spinCmd = m.spinner.Update(msg)
		cmds = append(cmds, spinCmd)

	case streamChunkMsg:
		chunk := msg.chunk
		switch {
		case chunk.Err != nil:
			debugLog("streamChunk: error: %v", chunk.Err)
			// Stream error: go idle but keep queued prompts intact.
			// The user can retry from idle state.
			if m.streamingItemIdx >= 0 {
				m.items[m.streamingItemIdx].content += "\n[stream error: " + chunk.Err.Error() + "]"
				m.streamingItemIdx = -1
			} else {
				m.items = append(m.items, displayItem{kind: itemError, content: chunk.Err.Error()})
			}
			m.state = stateIdle
			needRebuild = true
			cmds = append(cmds, m.input.Focus())
		case chunk.Done:
			debugLog("streamChunk: done, toolCalls=%d, streamingItemIdx=%d, content=%q", len(chunk.Msg.ToolCalls), m.streamingItemIdx, chunk.Msg.Content)
			// Stream finished — append the full message to history and handle tool calls.
			// If no delta chunks arrived but the message has content (e.g. the model
			// returned a full response without streaming tokens), create the display item now.
			if m.streamingItemIdx < 0 && chunk.Msg.Content != "" {
				m.items = append(m.items, displayItem{kind: itemAssistant, content: chunk.Msg.Content})
			}
			m.streamingItemIdx = -1
			m.messages = append(m.messages, chunk.Msg)
			if len(chunk.Msg.ToolCalls) == 0 {
				// Turn complete: drain queued prompts or go idle.
				needRebuild = true
				cmds = append(cmds, m.processQueueOrIdle())
			} else {
				for _, tc := range chunk.Msg.ToolCalls {
					debugLog("streamChunk: tool call id=%q name=%q", tc.ID, tc.Name)
					m.toolCallIdx[tc.ID] = len(m.items)
					m.items = append(m.items, displayItem{
						kind:       itemToolCall,
						toolName:   tc.Name,
						toolArgs:   formatArgsCompact(tc.Arguments),
						toolStatus: toolStatusPending,
					})
				}
				m.pendingCalls = append(m.pendingCalls, chunk.Msg.ToolCalls...)
				nextCmd := m.processNextCall()
				needRebuild = true
				cmds = append(cmds, nextCmd)
			}
		default:
			// Delta — create streaming assistant item on first token, then append.
			if m.streamingItemIdx < 0 {
				m.streamingItemIdx = len(m.items)
				m.items = append(m.items, displayItem{kind: itemAssistant, content: ""})
				m.state = stateStreaming
			}
			m.items[m.streamingItemIdx].content += chunk.Delta
			needRebuild = true
			cmds = append(cmds, readNextChunk(msg.ch))
		}

	case toolResultMsg:
		if msg.err != nil {
			debugLog("toolResult: error tool=%q err=%v", msg.toolName, msg.err)
			// Check if this is a path approval request from the tools manager.
			var pathErr chat.PathApprovalError
			if errors.As(msg.err, &pathErr) && m.executingCall != nil {
				reqs := pathErr.AccessRequests()
				// Queue path approvals; once all paths are approved the call re-runs.
				m.pendingCallAfterPaths = m.executingCall
				m.executingCall = nil
				m.currentPathRequest = reqs[0]
				m.pendingPathApprovals = reqs[1:]
				m.state = stateAwaitingPathApproval
				needRebuild = true
			} else {
				// Tool execution error: go idle but keep queued prompts intact.
				if idx, ok := m.toolCallIdx[msg.callID]; ok {
					m.items[idx].toolStatus = toolStatusFailed
				}
				m.items = append(m.items, displayItem{kind: itemError, content: msg.err.Error()})
				m.callingToolName = ""
				m.executingCall = nil
				m.pendingCalls = nil
				m.currentCall = nil
				m.state = stateIdle
				needRebuild = true
				cmds = append(cmds, m.input.Focus())
			}
		} else {
			debugLog("toolResult: ok tool=%q result=%q", msg.toolName, msg.result[:min(len(msg.result), 80)])
			if idx, ok := m.toolCallIdx[msg.callID]; ok {
				m.items[idx].toolStatus = toolStatusDone
			}
			m.messages = append(m.messages, ai.Message{
				Role:       "tool",
				ToolCallID: msg.callID,
				Content:    msg.result,
			})
			m.callingToolName = ""
			m.executingCall = nil
			nextCmd := m.processNextCall()
			needRebuild = true
			cmds = append(cmds, nextCmd)
		}

	case oauthNeedAuthMsg:
		debugLog("oauthNeedAuth: server=%q", msg.serverName)
		// Update the OAuth info item to show the auth URL.
		text := fmt.Sprintf(
			"To connect to %s, open this URL in your browser:\n%s\n\nWaiting for authorization…",
			msg.serverName, msg.authURL,
		)
		if m.oauthInfoIdx >= 0 {
			m.items[m.oauthInfoIdx].content = text
		} else {
			m.oauthInfoIdx = len(m.items)
			m.items = append(m.items, displayItem{kind: itemInfo, content: text})
		}
		needRebuild = true

	case oauthConnectedMsg:
		debugLog("oauthConnected: refreshing tools")
		// Refresh the tool list and add all newly available tools.
		newTools, err := m.session.Tools(m.ctx)
		if err != nil {
			m.items = append(m.items, displayItem{kind: itemError, content: "refreshing tools after OAuth: " + err.Error()})
		} else {
			m.tools = newTools
		}
		// Update info item to show success.
		if m.oauthInfoIdx >= 0 {
			m.items[m.oauthInfoIdx].content = "✓ Connected"
		}
		m.oauthInfoIdx = -1
		// Process the queued user prompt now that servers are ready.
		needRebuild = true
		cmds = append(cmds, m.processQueueOrIdle())

	case oauthConnectFailedMsg:
		debugLog("oauthConnectFailed: %v", msg.err)
		if m.oauthInfoIdx >= 0 {
			m.items[m.oauthInfoIdx].content = "OAuth connection failed: " + msg.err.Error()
			m.items[m.oauthInfoIdx].kind = itemError
		} else {
			m.items = append(m.items, displayItem{kind: itemError, content: msg.err.Error()})
		}
		m.oauthInfoIdx = -1
		// Keep the user's queued prompt but return to idle so they can retry.
		m.state = stateIdle
		needRebuild = true
		cmds = append(cmds, m.input.Focus())

	case contextDoneMsg:
		return m, tea.Quit
	}

	if needRebuild {
		m.rebuildContent()
	}

	// Keep input placeholder in sync with state so users always know what Enter does.
	m.input.Placeholder = inputPlaceholder(m.state)

	return m, tea.Batch(cmds...)
}

func (m Model) View() string {
	if m.width == 0 {
		return "" // not yet initialised — avoid rendering before first WindowSizeMsg
	}

	// Model picker takes over the full screen.
	if m.state == statePickingModel {
		return m.modelPicker.View()
	}

	var b strings.Builder

	b.WriteString(m.headerView())
	b.WriteString("\n")
	b.WriteString(separator(m.width))
	b.WriteString("\n")

	b.WriteString(m.viewport.View())
	b.WriteString("\n")

	b.WriteString(separator(m.width))
	b.WriteString("\n")

	s1, s2 := m.statusLines()
	b.WriteString(s1)
	b.WriteString("\n")
	b.WriteString(s2)
	b.WriteString("\n")

	b.WriteString(separator(m.width))
	b.WriteString("\n")

	if m.showCompletion {
		b.WriteString(m.completionView())
	}

	b.WriteString(m.inputView())

	return b.String()
}

// --- Helper views ---

func (m Model) headerView() string {
	servers := "no servers"
	if len(m.serverNames) > 0 {
		servers = strings.Join(m.serverNames, ", ")
	}
	text := fmt.Sprintf("werkler  model: %s  servers: %s", m.modelName, servers)
	return headerStyle.Width(m.width).Render(text)
}

func (m Model) statusLines() (line1, line2 string) {
	queueHint := ""
	if n := len(m.queuedPrompts); n > 0 {
		queueHint = "  " + queueCountStyle.Render(fmt.Sprintf("+%d queued", n))
	}

	switch m.state {
	case stateThinking:
		return statusStyle.Render(m.spinner.View()+" Thinking…") + queueHint, ""
	case stateConnectingOAuth:
		return statusStyle.Render(m.spinner.View()+" Connecting to MCP servers…") + queueHint, ""
	case stateStreaming:
		return statusStyle.Render(m.spinner.View()+" Streaming…") + queueHint, ""
	case stateCallingTool:
		name := toolNameStyle.Render(m.callingToolName)
		return statusStyle.Render(m.spinner.View()+" Calling tool: ") + name + queueHint, ""
	case stateAwaitingPathApproval:
		if m.currentPathRequest.Path == "" {
			return "", ""
		}
		accessKind := "read"
		if m.currentPathRequest.Write {
			accessKind = "write"
		}
		l1 := approvalPromptStyle.Render(fmt.Sprintf("  Allow %s access: ", accessKind)) + m.currentPathRequest.Path
		remaining := len(m.pendingPathApprovals)
		if remaining > 0 {
			l1 += statusStyle.Render(fmt.Sprintf(" (+%d more)", remaining))
		}
		l2 := approvalPromptStyle.Render("Allow? ") +
			keyHintStyle.Render("[y]") + "es  " +
			keyHintStyle.Render("[n]") + "o  " +
			keyHintStyle.Render("[a]") + "ll remaining"
		return l1, l2
	case stateAwaitingApproval:
		if m.currentCall == nil {
			return "", ""
		}
		args := formatArgsCompact(m.currentCall.Arguments)
		l1 := "  ▶ " + toolNameStyle.Render(m.currentCall.Name)
		if args != "" {
			l1 += "  " + args
		}
		l2 := approvalPromptStyle.Render("Allow? ") +
			keyHintStyle.Render("[y]") + "es  " +
			keyHintStyle.Render("[n]") + "o  " +
			keyHintStyle.Render("[a]") + "lways"
		return l1, l2
	default:
		mouseHint := "  " + keyHintStyle.Render("alt+m") + " select text"
		if !m.mouseEnabled {
			mouseHint = "  " + keyHintStyle.Render("alt+m") + " restore scroll"
		}
		pickerHint := ""
		if m.modelManager != nil {
			pickerHint = "  " + keyHintStyle.Render("ctrl+p") + " switch model"
		}
		return mouseHint + pickerHint, ""
	}
}

func (m Model) inputView() string {
	prefix := inputPrefixStyle.Render("You> ")
	return prefix + m.input.View()
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
	case itemUser:
		return userPrefixStyle.Render("You") + "  " + item.content

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
		name := toolNameStyle.Render(item.toolName)
		line := "  " + badge + " " + name
		if item.toolArgs != "" {
			line += "  " + item.toolArgs
		}
		if item.toolStatus == toolStatusDenied {
			line += "  " + toolDeniedStyle.Render("(denied)")
		}
		return line

	case itemError:
		return errorStyle.Render("Error: ") + item.content

	case itemInfo:
		return infoStyle.Render(item.content)

	case itemProcessOutput:
		prefix := processHandleStyle.Render("[process:" + item.handle + "]")
		// Raw output may contain ANSI codes, display as-is.
		return prefix + "\n" + item.content

	default:
		return item.content
	}
}

// --- Agent loop helpers ---

// approvePathRequest calls the appropriate session approval method based on the request's Write flag.
func (m *Model) approvePathRequest(req chat.PathAccessRequest) {
	if req.Write {
		m.session.ApprovePathWriteForSession(req.Path)
	} else {
		m.session.ApprovePathReadForSession(req.Path)
	}
}

// processNextPath advances the path approval queue.
// When all paths are approved, dispatch the pending tool call.
// Must only be called from Update.
func (m *Model) processNextPath() tea.Cmd {
	if len(m.pendingPathApprovals) > 0 {
		m.currentPathRequest = m.pendingPathApprovals[0]
		m.pendingPathApprovals = m.pendingPathApprovals[1:]
		m.state = stateAwaitingPathApproval
		return nil
	}
	// All paths approved — proceed with the pending tool call.
	m.state = stateIdle
	if m.pendingCallAfterPaths != nil {
		call := *m.pendingCallAfterPaths
		m.pendingCallAfterPaths = nil
		if idx, ok := m.toolCallIdx[call.ID]; ok {
			m.items[idx].toolStatus = toolStatusRunning
		}
		m.callingToolName = call.Name
		m.executingCall = &call
		m.state = stateCallingTool
		return doCallTool(m.ctx, m.session, call)
	}
	return m.input.Focus()
}

// processQueueOrIdle is called when an AI turn finishes successfully.
// If queued prompts exist, the next one is dequeued and sent immediately,
// keeping the agent busy. Otherwise the TUI returns to idle.
func (m *Model) processQueueOrIdle() tea.Cmd {
	if len(m.queuedPrompts) > 0 {
		text := m.queuedPrompts[0]
		m.queuedPrompts = m.queuedPrompts[1:]
		m.messages = append(m.messages, ai.Message{Role: "user", Content: text})
		m.items = append(m.items, displayItem{kind: itemUser, content: text})
		m.state = stateThinking
		return doStartStream(m.ctx, m.client, m.messages, m.tools)
	}
	m.state = stateIdle
	return m.input.Focus()
}

// processNextCall advances the tool call queue, updating state and returning the
// next command to execute. Must only be called from Update.
func (m *Model) processNextCall() tea.Cmd {
	if len(m.pendingCalls) == 0 {
		debugLog("processNextCall: no more pending calls, starting new stream (messages=%d)", len(m.messages))
		m.state = stateThinking
		return doStartStream(m.ctx, m.client, m.messages, m.tools)
	}

	call := m.pendingCalls[0]
	m.pendingCalls = m.pendingCalls[1:]
	callCopy := call
	m.currentCall = &callCopy

	debugLog("processNextCall: dispatching tool=%q id=%q approved=%v", call.Name, call.ID, m.session.IsApproved(call.Name))
	if m.session.IsApproved(call.Name) {
		if idx, ok := m.toolCallIdx[call.ID]; ok {
			m.items[idx].toolStatus = toolStatusRunning
		}
		m.callingToolName = call.Name
		m.currentCall = nil
		m.executingCall = &callCopy
		m.state = stateCallingTool
		return doCallTool(m.ctx, m.session, call)
	}

	m.state = stateAwaitingApproval
	return nil
}

// --- Tea commands ---

// doStartStream kicks off a streaming completion in a goroutine and returns
// the first chunk as a streamChunkMsg (carrying the channel for further reads).
func doStartStream(ctx context.Context, client ai.StreamCompleter, messages []ai.Message, tools []ai.ToolDefinition) tea.Cmd {
	snapshot := make([]ai.Message, len(messages))
	copy(snapshot, messages)
	return func() tea.Msg {
		ch := client.CompleteStream(ctx, snapshot, tools)
		return readNextChunk(ch)()
	}
}

// readNextChunk returns a Cmd that reads one chunk from ch and wraps it in a
// streamChunkMsg (which carries ch so Update can dispatch the next read).
func readNextChunk(ch <-chan ai.StreamChunk) tea.Cmd {
	return func() tea.Msg {
		chunk, ok := <-ch
		if !ok {
			// Channel closed without a Done chunk — treat as done with empty message.
			return streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Done: true}}
		}
		return streamChunkMsg{ch: ch, chunk: chunk}
	}
}

func doCallTool(ctx context.Context, session *chat.Session, tc ai.ToolCall) tea.Cmd {
	return func() tea.Msg {
		result, err := session.CallTool(ctx, tc)
		return toolResultMsg{tc.ID, tc.Name, result, err}
	}
}

func watchContext(ctx context.Context) tea.Cmd {
	return func() tea.Msg {
		<-ctx.Done()
		return contextDoneMsg{}
	}
}

// doConnectOAuth runs pending OAuth server connections in a goroutine.
// The send function is used to notify the TUI when an auth URL is ready to display.
func doConnectOAuth(ctx context.Context, session *chat.Session, send func(tea.Msg)) tea.Cmd {
	return func() tea.Msg {
		display := func(serverName, authURL string) {
			send(oauthNeedAuthMsg{serverName: serverName, authURL: authURL})
		}
		if err := session.ConnectPendingOAuth(ctx, display); err != nil {
			return oauthConnectFailedMsg{err: err}
		}
		return oauthConnectedMsg{}
	}
}

// doListModels fetches available chat models from the ModelManager.
func doListModels(ctx context.Context, mm ai.ModelManager) tea.Cmd {
	return func() tea.Msg {
		models, err := mm.ListModels(ctx)
		if err != nil {
			return modelsErrMsg{err: err}
		}
		return modelsLoadedMsg{models: models}
	}
}

// --- Formatting helpers ---

func formatArgsCompact(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}
	b, err := json.Marshal(args)
	if err != nil {
		return fmt.Sprintf("%v", args)
	}
	s := string(b)
	if len(s) > 120 {
		s = s[:120] + "…"
	}
	return s
}

// --- Entry point ---

// RunTUI starts the full-screen interactive TUI. It blocks until the user exits
// or ctx is cancelled.
func RunTUI(
	ctx context.Context,
	client ai.StreamCompleter,
	session *chat.Session,
	toolMgr *tools.Manager,
	modelName string,
	serverNames []string,
) error {
	tools, err := session.Tools(ctx)
	if err != nil {
		return fmt.Errorf("fetching tools: %w", err)
	}

	// Resolve glamour style before starting bubbletea. WithAutoStyle() sends an
	// OSC 11 query to stdout; the terminal's response arrives on stdin and would
	// be swallowed by bubbletea as garbage key input if done after p.Run().
	glamourStyle := resolveGlamourStyle()

	// Enable debug logging to a file if WERKLER_DEBUG_LOG is set.
	if logPath := os.Getenv("WERKLER_DEBUG_LOG"); logPath != "" {
		f, ferr := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
		if ferr == nil {
			debugLogger = log.New(f, "", log.Ltime|log.Lmicroseconds)
			defer func() { _ = f.Close() }()
		}
	}

	// sendFn is a closure that forwards messages to the bubbletea program.
	// It is set after tea.NewProgram so that the model owns the same function pointer.
	var sendFn func(tea.Msg)
	sendFn = func(msg tea.Msg) {
		// no-op until prog is assigned below
	}

	m := initialModel(ctx, client, session, tools, modelName, serverNames, glamourStyle, sendFn)

	// Wire p.Send so goroutines (OAuth, etc.) can post messages back to the TUI.
	// tea.NewProgram copies m, so the closure captures prog by pointer and prog is
	// non-nil by the time any cmd goroutine fires (after p.Run starts the event loop).
	var prog *tea.Program
	sendFn = func(msg tea.Msg) {
		if prog != nil {
			prog.Send(msg)
		}
	}
	// Update the send field on the already-constructed model so it references the
	// real function.
	m.send = sendFn

	// Wire live process output from the tools manager into the TUI.
	if toolMgr != nil {
		toolMgr.SetOutputNotify(func(handle, raw, clean string) {
			sendFn(processOutputMsg{handle: handle, raw: raw, clean: clean})
		})
	}

	prog = tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	_, err = prog.Run()
	return err
}

// resolveGlamourStyle returns the glamour style name to use for markdown rendering.
// It honours the GLAMOUR_STYLE environment variable, falling back to auto-detection
// via lipgloss (safe to call before bubbletea starts since it queries the terminal
// directly on real stdin/stdout before bubbletea intercepts them).
func resolveGlamourStyle() string {
	if s := os.Getenv("GLAMOUR_STYLE"); s != "" {
		return s
	}
	if lipgloss.HasDarkBackground() {
		return "dark"
	}
	return "light"
}

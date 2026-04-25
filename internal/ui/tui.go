package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
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
)

// inputPlaceholder returns the appropriate placeholder text for the input
// based on the current TUI state.
func inputPlaceholder(state tuiState) string {
	switch state {
	case stateAwaitingApproval:
		return "Approve the tool call above (y / n / a)…"
	case stateIdle:
		return "Type a message, press Enter to send…"
	default:
		return "Queue a follow-up, press Enter…"
	}
}

// --- Display item kinds and statuses ---

const (
	itemUser      = "user"
	itemAssistant = "assistant"
	itemToolCall  = "tool_call"
	itemError     = "error"
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

// --- Model ---

// Model is the bubbletea model for the interactive TUI.
type Model struct {
	ctx     context.Context
	client  ai.StreamCompleter
	session *chat.Session
	tools   []ai.ToolDefinition

	// Conversation history (managed only in Update, never in tea.Cmd goroutines).
	messages []ai.Message

	// Agent state machine.
	state            tuiState
	pendingCalls     []ai.ToolCall
	currentCall      *ai.ToolCall
	callingToolName  string // name of tool currently executing (stateCallingTool)
	streamingItemIdx int    // index into items of the in-progress assistant item; -1 if none

	// Queue of user prompts entered while the AI is busy.
	// Processed FIFO after the current agent turn completes successfully.
	queuedPrompts []string

	// Display items for the viewport.
	items       []displayItem
	toolCallIdx map[string]int // callID → index in items

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
}

func initialModel(
	ctx context.Context,
	client ai.StreamCompleter,
	session *chat.Session,
	tools []ai.ToolDefinition,
	modelName string,
	serverNames []string,
	glamourStyle string,
) Model {
	sp := spinner.New()
	sp.Spinner = thinkingSpinner

	ti := textinput.New()
	ti.Placeholder = inputPlaceholder(stateIdle)
	ti.Prompt = ""
	ti.CharLimit = 0
	ti.Focus() // must be called here; Init() runs on a value copy so mutations there are lost

	return Model{
		ctx:              ctx,
		client:           client,
		session:          session,
		tools:            tools,
		messages:         chat.NewConversation(),
		state:            stateIdle,
		streamingItemIdx: -1,
		toolCallIdx:      make(map[string]int),
		viewport:         viewport.New(0, 0),
		input:            ti,
		spinner:          sp,
		modelName:        modelName,
		serverNames:      serverNames,
		glamourStyle:     glamourStyle,
	}
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

		switch m.state {
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
			if msg.Type == tea.KeyEnter {
				text := strings.TrimSpace(m.input.Value())
				if text != "" {
					m.input.Reset()
					m.messages = append(m.messages, ai.Message{Role: "user", Content: text})
					m.items = append(m.items, displayItem{kind: itemUser, content: text})
					m.state = stateThinking
					needRebuild = true
					cmds = append(cmds, doStartStream(m.ctx, m.client, m.messages, m.tools))
				}
			} else {
				var inputCmd tea.Cmd
				m.input, inputCmd = m.input.Update(msg)
				cmds = append(cmds, inputCmd)
			}

		case stateThinking, stateStreaming, stateCallingTool:
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

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		vph := m.height - fixedLines
		if vph < 1 {
			vph = 1
		}
		// Forward to viewport so it can update its internal state, then
		// override the dimensions we want.
		var vpCmd tea.Cmd
		m.viewport, vpCmd = m.viewport.Update(msg)
		cmds = append(cmds, vpCmd)
		m.viewport.Width = m.width
		m.viewport.Height = vph
		m.input.Width = m.width - 5 // 5 = len("You> ")
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
			// Tool execution error: go idle but keep queued prompts intact.
			if idx, ok := m.toolCallIdx[msg.callID]; ok {
				m.items[idx].toolStatus = toolStatusFailed
			}
			m.items = append(m.items, displayItem{kind: itemError, content: msg.err.Error()})
			m.callingToolName = ""
			m.pendingCalls = nil
			m.currentCall = nil
			m.state = stateIdle
			needRebuild = true
			cmds = append(cmds, m.input.Focus())
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
			nextCmd := m.processNextCall()
			needRebuild = true
			cmds = append(cmds, nextCmd)
		}

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
	case stateStreaming:
		return statusStyle.Render(m.spinner.View()+" Streaming…") + queueHint, ""
	case stateCallingTool:
		name := toolNameStyle.Render(m.callingToolName)
		return statusStyle.Render(m.spinner.View()+" Calling tool: ") + name + queueHint, ""
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
		return "", ""
	}
}

func (m Model) inputView() string {
	prefix := inputPrefixStyle.Render("You> ")
	return prefix + m.input.View()
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

	default:
		return item.content
	}
}

// --- Agent loop helpers ---

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

	m := initialModel(ctx, client, session, tools, modelName, serverNames, glamourStyle)
	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	_, err = p.Run()
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

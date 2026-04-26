package ui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/config"
	mcppkg "github.com/icedream/werkler/internal/mcp"
	"github.com/icedream/werkler/internal/sessionstore"
	"github.com/icedream/werkler/internal/skills"
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
	stateConnectingMCP        // MCP servers connecting in background
	stateConnectingOAuth      // running deferred OAuth server connections
	statePickingModel         // model selection list is open
	statePickingSession       // session picker list is open
	statePickingTools         // tool enable/disable picker is open
	stateAwaitingUserQuestion // AI asked a question; waiting for the user's reply
)

// inputPlaceholder returns the appropriate placeholder text for the input
// based on the current TUI state.
func inputPlaceholder(state tuiState) string {
	switch state {
	case stateAwaitingApproval:
		return "Approve the tool call above (y / n / a)…"
	case stateAwaitingPathApproval:
		return "Approve path access (y / n / a)…"
	case stateAwaitingUserQuestion:
		return "Use ↑/↓ to select a choice, Enter to confirm…"
	case stateConnectingMCP:
		return "Connecting to MCP servers… (queue a message with Enter)"
	case stateConnectingOAuth:
		return "Waiting for authorization in browser…"
	case statePickingModel, statePickingSession, statePickingTools:
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
	itemAskUser       = "ask_user"    // ask_user tool waiting for a response
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

type modelsLoadedMsg struct{ models []ai.ModelItem }
type modelsErrMsg struct{ err error }

// --- Tool picker messages ---

type allToolsMsg struct{ tools []ai.ToolDefinition }
type allToolsErrMsg struct{ err error }

// processOutputMsg carries new output from a running process for live display.
type processOutputMsg struct {
	handle string
	raw    string // ANSI-preserved for display
	clean  string // ANSI-stripped (used for AI, not displayed)
}

// sessionHintMsg delivers the latest CWD-matching session found on startup.
type sessionHintMsg struct{ sess *sessionstore.Session }

// sessionSavedMsg signals that a background save completed (errors are silent).
type sessionSavedMsg struct{}

// sessionsListMsg delivers the list of sessions for the picker.
type sessionsListMsg struct {
	sessions []sessionstore.Session
	err      error
}

// askUserResult carries the user's reply (or context error) back to the blocked
// tool goroutine.
type askUserResult struct {
	answer string
	err    error
}

// askUserMsg is sent by the ask_user tool goroutine to the TUI so it can present
// the question and collect a reply. resultCh is buffered (capacity 1).
type askUserMsg struct {
	callID        string
	question      string
	choices       []string
	recommended   string
	allowFreeform bool
	resultCh      chan<- askUserResult
}

// mcpServerStatusMsg is sent by a background MCP connect tea.Cmd when a single
// server finishes connecting (or fails, or is deferred for OAuth).
type mcpServerStatusMsg struct {
	name     string
	deferred bool // true = OAuth deferred, not an error
	err      error
}

// mcpToolsRefreshedMsg is sent after all MCP servers have finished connecting,
// once the tool list has been re-fetched from the session.
type mcpToolsRefreshedMsg struct {
	tools []ai.ToolDefinition
	err   error
}

// modelItem implements list.Item for a model picker entry.
type modelItem struct{ ai.ModelItem }

func (m modelItem) FilterValue() string { return m.Display() }
func (m modelItem) Title() string       { return m.Model }
func (m modelItem) Description() string { return m.Provider }

// sessionItem implements list.Item for the session picker.
type sessionItem struct{ sess sessionstore.Session }

func (s sessionItem) FilterValue() string { return s.sess.Title }
func (s sessionItem) Title() string       { return s.sess.Title }
func (s sessionItem) Description() string {
	return formatAge(s.sess.UpdatedAt) + "  " + shortenHomePath(s.sess.CWD)
}

// toolItem implements list.Item for the tool picker.
// The name field holds the full AI-facing tool name (e.g. "github__get_file_contents").
type toolItem struct {
	name        string
	description string
	enabled     bool
}

func toolServerName(name string) string {
	if i := strings.Index(name, "__"); i >= 0 {
		return name[:i]
	}
	return "(built-in)"
}

func toolBaseName(name string) string {
	if i := strings.Index(name, "__"); i >= 0 {
		return name[i+2:]
	}
	return name
}

func (t toolItem) FilterValue() string { return t.name }
func (t toolItem) Title() string {
	check := "[ ]"
	if t.enabled {
		check = "[✓]"
	}
	return check + " " + toolBaseName(t.name)
}
func (t toolItem) Description() string {
	server := toolServerName(t.name)
	desc := t.description
	const maxDesc = 80
	if len(desc) > maxDesc {
		desc = desc[:maxDesc] + "…"
	}
	if desc == "" {
		return "[" + server + "]"
	}
	return "[" + server + "] " + desc
}

// SessionOptions configures optional session persistence behaviour for RunTUI.
type SessionOptions struct {
	Store      *sessionstore.Store   // nil = session persistence disabled
	Initial    *sessionstore.Session // non-nil = load this session on startup
	OpenPicker bool                  // open session picker immediately on launch
	// PersistToolApproval, if non-nil, is called when the user permanently
	// approves a tool (adds it to auto_approve_tools in the config file).
	PersistToolApproval func(toolName string) error
	// PersistPathApproval, if non-nil, is called when the user permanently
	// approves a path (adds it to auto_approve_paths in the config file).
	// write indicates whether write access was granted (false = read-only).
	PersistPathApproval func(path string, write bool) error
	// Skills is the list of loaded skills to mention in the system prompt.
	Skills []skills.Skill
	// MCPManager and MCPServers, when both non-nil/non-empty, enable background
	// MCP server connection. The TUI starts in stateConnectingMCP and transitions
	// to stateIdle once all servers have been connected (or failed).
	MCPManager *mcppkg.Manager
	MCPServers []config.MCPServerConfig
}

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
			name:        "clear",
			description: "Clear the conversation history",
			action: func(m *Model) []tea.Cmd {
				m.messages = m.newConversation()
				m.items = nil
				m.toolCallIdx = make(map[string]int)
				m.streamingItemIdx = -1
				m.oauthInfoIdx = -1
				m.rebuildContent()
				return nil
			},
		},
		{
			name:        "allow-all",
			description: "Toggle allow-all mode — auto-approve all tool calls and path access without prompting",
			action: func(m *Model) []tea.Cmd {
				m.session.SetAllowAll(!m.session.AllowAll())
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

	// Session picker (only valid during statePickingSession).
	sessionPicker list.Model

	// Tool picker (only valid during statePickingTools).
	toolPicker  list.Model
	allToolDefs []ai.ToolDefinition // full unfiltered list, set on allToolsMsg

	// Session persistence.
	sessionStore     *sessionstore.Store
	sessionID        string    // current session's ID; empty until first save
	sessionCreatedAt time.Time // set on first save
	sessionCWD       string    // cwd at TUI startup (used for resume hint + new session)
	// resumeHint is the latest CWD-matching session found on startup.
	// Shown in the status bar until the first message is sent.
	resumeHint *sessionstore.Session

	// persistToolApproval, if non-nil, saves a tool name to auto_approve_tools.
	persistToolApproval func(toolName string) error
	// persistPathApproval, if non-nil, saves a path to auto_approve_paths.
	persistPathApproval func(path string, write bool) error

	// pendingApprovalChoice is the staged choice in an approval dialog ("y", "a",
	// "p", or "n"). Empty means nothing is staged yet. Confirmed with Enter.
	pendingApprovalChoice string

	// ask_user state (stateAwaitingUserQuestion).
	askUserCallID        string
	askUserQuestion      string
	askUserChoices       []string
	askUserRecommended   string
	askUserAllowFreeform bool
	askUserSelectedIdx   int // index of highlighted choice; -1 = freeform input active
	askUserItemIdx       int // index of the question display item in items; -1 if none
	askUserResultCh      chan<- askUserResult
	askUserSavedDraft    string // input text saved on entry, restored on exit

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

	// skills holds loaded skills, used to build the system prompt hint.
	skills []skills.Skill

	// MCP background connection state.
	// mcpManager is the manager used for ConnectOne calls during startup.
	// mcpPending is decremented by each mcpServerStatusMsg; when it hits 0,
	// doRefreshMCPTools is dispatched and state transitions to stateIdle.
	// mcpConnectingInfoIdx is the index of the status display item.
	// mcpInitCmds holds the per-server tea.Cmds returned from Init().
	mcpManager           *mcppkg.Manager
	mcpPending           int
	mcpConnected         int
	mcpFailed            int
	mcpConnectingInfoIdx int
	mcpInitCmds          []tea.Cmd

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

	// Session picker: same style as model picker.
	sessDel := list.NewDefaultDelegate()
	sessPicker := list.New(nil, sessDel, 0, 0)
	sessPicker.Title = "Resume session"
	sessPicker.SetShowStatusBar(false)
	sessPicker.SetFilteringEnabled(true)
	sessPicker.DisableQuitKeybindings()

	// Tool picker: space to toggle, filtering enabled.
	toolDel := list.NewDefaultDelegate()
	toolPickerM := list.New(nil, toolDel, 0, 0)
	toolPickerM.Title = "Toggle tools  [space] enable/disable"
	toolPickerM.SetShowStatusBar(false)
	toolPickerM.SetFilteringEnabled(true)
	toolPickerM.DisableQuitKeybindings()

	// Use modelManager when the client also implements ModelManager.
	var mm ai.ModelManager
	if m, ok := client.(ai.ModelManager); ok {
		mm = m
	}

	cwd, _ := os.Getwd()

	return Model{
		ctx:                ctx,
		client:             client,
		session:            session,
		tools:              tools,
		send:               send,
		modelManager:       mm,
		messages:           chat.NewConversation(),
		state:              stateIdle,
		streamingItemIdx:   -1,
		oauthInfoIdx:       -1,
		askUserSelectedIdx: -1,
		askUserItemIdx:     -1,
		toolCallIdx:        make(map[string]int),
		viewport:           viewport.New(0, 0),
		modelPicker:        picker,
		sessionPicker:      sessPicker,
		toolPicker:         toolPickerM,
		input:              ti,
		spinner:            sp,
		modelName:          modelName,
		serverNames:        serverNames,
		glamourStyle:       glamourStyle,
		mouseEnabled:       true,
		sessionCWD:         cwd,
	}
}

// --- Slash-command helpers ---

// newConversation builds an initial messages list for a fresh conversation,
// injecting a skills hint into the system prompt when skills are loaded.
func (m *Model) newConversation() []ai.Message {
	if len(m.skills) == 0 {
		return chat.NewConversation()
	}
	parts := make([]string, len(m.skills))
	for i, s := range m.skills {
		parts[i] = s.Name + ": " + s.Description
	}
	hint := "Available skills (call use_skill to load instructions):\n"
	for _, p := range parts {
		hint += "- " + p + "\n"
	}
	return chat.NewConversation(strings.TrimRight(hint, "\n"))
}

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
	cmds := []tea.Cmd{
		textinput.Blink,
		m.spinner.Tick,
		watchContext(m.ctx),
	}
	if m.sessionStore != nil {
		cmds = append(cmds, doLoadSessionHint(m.sessionStore, m.sessionCWD))
	}
	if len(m.mcpInitCmds) > 0 {
		cmds = append(cmds, m.mcpInitCmds...)
	}
	return tea.Batch(cmds...)
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
				case "y", "Y", "a", "A", "n", "N":
					key := strings.ToLower(msg.String())
					if m.pendingApprovalChoice == key {
						m.pendingApprovalChoice = "" // toggle off
					} else {
						m.pendingApprovalChoice = key
					}
					needRebuild = true
				case "d", "D":
					if m.pendingApprovalChoice == "d" {
						m.pendingApprovalChoice = ""
					} else {
						m.pendingApprovalChoice = "d"
					}
					needRebuild = true
				case "p", "P":
					if m.persistPathApproval != nil {
						if m.pendingApprovalChoice == "p" {
							m.pendingApprovalChoice = ""
						} else {
							m.pendingApprovalChoice = "p"
						}
						needRebuild = true
					}
				case "enter":
					switch m.pendingApprovalChoice {
					case "y":
						m.approvePathRequest(m.currentPathRequest)
						m.currentPathRequest = chat.PathAccessRequest{}
						cmds = append(cmds, m.processNextPath())
						needRebuild = true
					case "d":
						// Approve the whole containing directory.
						dirReq := chat.PathAccessRequest{
							Path:    filepath.Dir(m.currentPathRequest.Path),
							Write:   m.currentPathRequest.Write,
							Execute: m.currentPathRequest.Execute,
						}
						m.approvePathRequest(dirReq)
						m.currentPathRequest = chat.PathAccessRequest{}
						cmds = append(cmds, m.processNextPath())
						needRebuild = true
					case "a":
						m.approvePathRequest(m.currentPathRequest)
						for _, req := range m.pendingPathApprovals {
							m.approvePathRequest(req)
						}
						m.pendingPathApprovals = nil
						m.currentPathRequest = chat.PathAccessRequest{}
						cmds = append(cmds, m.processNextPath())
						needRebuild = true
					case "p":
						req := m.currentPathRequest
						m.approvePathRequest(req)
						if err := m.persistPathApproval(req.Path, req.Write || req.Execute); err != nil {
							m.items = append(m.items, displayItem{
								kind:    itemError,
								content: "Failed to save approval: " + err.Error(),
							})
						}
						m.currentPathRequest = chat.PathAccessRequest{}
						cmds = append(cmds, m.processNextPath())
						needRebuild = true
					case "n":
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
					}
				case "esc":
					m.pendingApprovalChoice = ""
					needRebuild = true
				default:
					var vpCmd tea.Cmd
					m.viewport, vpCmd = m.viewport.Update(msg)
					cmds = append(cmds, vpCmd)
				}
			}

		case stateAwaitingApproval:
			if m.currentCall != nil {
				switch msg.String() {
				case "y", "Y", "a", "A", "n", "N":
					key := strings.ToLower(msg.String())
					if m.pendingApprovalChoice == key {
						m.pendingApprovalChoice = ""
					} else {
						m.pendingApprovalChoice = key
					}
					needRebuild = true
				case "p", "P":
					if m.persistToolApproval != nil {
						if m.pendingApprovalChoice == "p" {
							m.pendingApprovalChoice = ""
						} else {
							m.pendingApprovalChoice = "p"
						}
						needRebuild = true
					}
				case "enter":
					switch m.pendingApprovalChoice {
					case "y":
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
					case "a":
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
					case "p":
						call := *m.currentCall
						m.session.ApproveForSession(call.Name)
						if err := m.persistToolApproval(call.Name); err != nil {
							m.items = append(m.items, displayItem{
								kind:    itemError,
								content: "Failed to save approval: " + err.Error(),
							})
						}
						if idx, ok := m.toolCallIdx[call.ID]; ok {
							m.items[idx].toolStatus = toolStatusRunning
						}
						m.callingToolName = call.Name
						m.currentCall = nil
						m.executingCall = &call
						m.state = stateCallingTool
						needRebuild = true
						cmds = append(cmds, doCallTool(m.ctx, m.session, call))
					case "n":
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
					}
				case "esc":
					m.pendingApprovalChoice = ""
					needRebuild = true
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
							m.resumeHint = nil // dismiss hint once user starts chatting
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

			case tea.KeyCtrlR:
				if m.sessionStore != nil {
					m.showCompletion = false
					m.state = statePickingSession
					m.sessionPicker.SetItems(nil)
					m.sessionPicker.SetSize(m.width, m.height-fixedLines)
					cmds = append(cmds, doLoadSessions(m.sessionStore))
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
					item := sel.(modelItem)
					m.modelManager.SetModel(item.ModelItem)
					m.modelName = item.Display()
				}
				m.state = stateIdle
				m.updateCompletion()
			default:
				var pickerCmd tea.Cmd
				m.modelPicker, pickerCmd = m.modelPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			}

		case statePickingSession:
			switch msg.Type {
			case tea.KeyEsc, tea.KeyCtrlC:
				m.state = stateIdle
				m.updateCompletion()
			case tea.KeyEnter:
				if sel := m.sessionPicker.SelectedItem(); sel != nil {
					sess := sel.(sessionItem).sess
					m.resumeHint = nil
					m.applySession(&sess)
					needRebuild = true
				}
				m.state = stateIdle
				m.updateCompletion()
			default:
				var pickerCmd tea.Cmd
				m.sessionPicker, pickerCmd = m.sessionPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			}

		case statePickingTools:
			switch msg.Type {
			case tea.KeyEsc, tea.KeyCtrlC:
				// Refresh m.tools from the (possibly updated) disabled set and go back.
				m.tools = m.filteredFromAllDefs()
				m.state = stateIdle
				m.updateCompletion()
			case tea.KeyRunes:
				if msg.String() == " " {
					// Toggle the currently selected tool.
					if sel := m.toolPicker.SelectedItem(); sel != nil {
						item := sel.(toolItem)
						item.enabled = !item.enabled
						m.session.SetToolEnabled(item.name, item.enabled)
						m.toolPicker.SetItem(m.toolPicker.Index(), item)
					}
					break
				}
				var pickerCmd tea.Cmd
				m.toolPicker, pickerCmd = m.toolPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			default:
				var pickerCmd tea.Cmd
				m.toolPicker, pickerCmd = m.toolPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			}

		case stateConnectingMCP:
			// Allow viewport scrolling and prompt queuing; block submission until connected.
			switch msg.Type {
			case tea.KeyEnter:
				text := strings.TrimSpace(m.input.Value())
				if text != "" {
					m.input.Reset()
					m.items = append(m.items, displayItem{kind: itemUser, content: text})
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
				var inputCmd tea.Cmd
				m.input, inputCmd = m.input.Update(msg)
				cmds = append(cmds, inputCmd)
				var vpCmd tea.Cmd
				m.viewport, vpCmd = m.viewport.Update(msg)
				cmds = append(cmds, vpCmd)
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

		case stateAwaitingUserQuestion:
			switch msg.Type {
			case tea.KeyUp:
				if len(m.askUserChoices) > 0 {
					switch {
					case m.askUserSelectedIdx > 0:
						m.askUserSelectedIdx--
					case m.askUserAllowFreeform && m.askUserSelectedIdx == 0:
						m.askUserSelectedIdx = -1 // back to freeform
					default:
						m.askUserSelectedIdx = len(m.askUserChoices) - 1 // wrap (no-freeform)
					}
					needRebuild = true
				}
			case tea.KeyDown:
				if len(m.askUserChoices) > 0 {
					switch {
					case m.askUserSelectedIdx < len(m.askUserChoices)-1:
						m.askUserSelectedIdx++
					case m.askUserAllowFreeform:
						m.askUserSelectedIdx = -1 // past last = freeform
					default:
						m.askUserSelectedIdx = 0 // wrap (no-freeform)
					}
					needRebuild = true
				}
			case tea.KeyEnter:
				var answer string
				var readyToSubmit bool
				switch {
				case m.askUserSelectedIdx >= 0:
					answer = m.askUserChoices[m.askUserSelectedIdx]
					readyToSubmit = true
				case m.askUserAllowFreeform:
					answer = strings.TrimSpace(m.input.Value())
					readyToSubmit = answer != ""
				}
				if !readyToSubmit {
					break
				}
				// Finalise the display item to show the chosen answer.
				if m.askUserItemIdx >= 0 && m.askUserItemIdx < len(m.items) {
					m.items[m.askUserItemIdx] = displayItem{
						kind:    itemInfo,
						content: "❓ " + m.askUserQuestion + "\n\n→ " + answer,
					}
				}
				ch := m.askUserResultCh
				m.teardownAskUser()
				m.state = stateCallingTool
				needRebuild = true
				if ch != nil {
					ch <- askUserResult{answer: answer}
				}
			default:
				if m.askUserAllowFreeform || len(m.askUserChoices) == 0 {
					// Typing any rune deselects any highlighted choice.
					if msg.Type == tea.KeyRunes && m.askUserSelectedIdx >= 0 {
						m.askUserSelectedIdx = -1
						needRebuild = true
					}
					var inputCmd tea.Cmd
					m.input, inputCmd = m.input.Update(msg)
					cmds = append(cmds, inputCmd)
				}
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

	case mcpServerStatusMsg:
		// One MCP server finished connecting (or failed, or was deferred for OAuth).
		m.mcpPending--
		if msg.err != nil {
			m.mcpFailed++
		} else {
			m.mcpConnected++
		}
		// Update the connecting status item.
		if m.mcpConnectingInfoIdx >= 0 && m.mcpConnectingInfoIdx < len(m.items) {
			var content string
			switch {
			case m.mcpPending > 0:
				content = fmt.Sprintf("⟳ Connecting to MCP servers… (%d/%d done", m.mcpConnected+m.mcpFailed, m.mcpConnected+m.mcpFailed+m.mcpPending)
				if m.mcpFailed > 0 {
					content += fmt.Sprintf(", %d failed", m.mcpFailed)
				}
				content += ")"
			case m.mcpFailed == 0:
				content = fmt.Sprintf("✓ Connected to %d MCP server(s)", m.mcpConnected)
			default:
				content = fmt.Sprintf("⚠ MCP servers: %d connected, %d failed", m.mcpConnected, m.mcpFailed)
			}
			m.items[m.mcpConnectingInfoIdx].content = content
		}
		needRebuild = true
		if m.mcpPending == 0 {
			// All servers done — refresh the tool list and unblock input.
			cmds = append(cmds, doRefreshMCPTools(m.ctx, m.session))
		}

	case mcpToolsRefreshedMsg:
		if msg.err != nil {
			m.items = append(m.items, displayItem{kind: itemError, content: "loading MCP tools: " + msg.err.Error()})
		} else {
			m.tools = msg.tools
			m.allToolDefs = msg.tools
		}
		m.state = stateIdle
		m.input.Placeholder = inputPlaceholder(stateIdle)
		m.updateCompletion()
		needRebuild = true

	case modelsLoadedMsg:
		// Only process if we're still in model-picking state (guard against stale results).
		if m.state == statePickingModel {
			items := make([]list.Item, len(msg.models))
			selectedIdx := 0
			for i, it := range msg.models {
				items[i] = modelItem{it}
				if it.Display() == m.modelName {
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

	case allToolsMsg:
		if m.state == statePickingTools {
			m.allToolDefs = msg.tools
			items := make([]list.Item, len(msg.tools))
			for i, t := range msg.tools {
				items[i] = toolItem{
					name:        t.Name,
					description: t.Description,
					enabled:     m.session.IsToolEnabled(t.Name),
				}
			}
			m.toolPicker.SetItems(items)
			m.toolPicker.SetSize(m.width, m.height-fixedLines)
		}

	case allToolsErrMsg:
		if m.state == statePickingTools {
			m.items = append(m.items, displayItem{kind: itemError, content: "listing tools: " + msg.err.Error()})
			m.state = stateIdle
			m.updateCompletion()
			needRebuild = true
		}

	case sessionHintMsg:
		m.resumeHint = msg.sess
		needRebuild = true

	case sessionSavedMsg:
		// Auto-save completed; nothing to do (errors silently dropped).

	case sessionsListMsg:
		if m.state == statePickingSession {
			if msg.err != nil {
				m.items = append(m.items, displayItem{kind: itemError, content: "listing sessions: " + msg.err.Error()})
				m.state = stateIdle
				m.updateCompletion()
				needRebuild = true
			} else {
				items := make([]list.Item, len(msg.sessions))
				for i, s := range msg.sessions {
					items[i] = sessionItem{sess: s}
				}
				m.sessionPicker.SetItems(items)
				m.sessionPicker.SetSize(m.width, m.height-fixedLines)
			}
		}

	case askUserMsg:
		// Guard against stale messages that arrive after the call already finished
		// (e.g. context was cancelled between send and receive).
		if m.executingCall == nil || m.executingCall.ID != msg.callID {
			break
		}
		m.askUserCallID = msg.callID
		m.askUserQuestion = msg.question
		m.askUserChoices = msg.choices
		m.askUserRecommended = msg.recommended
		m.askUserAllowFreeform = msg.allowFreeform

		// Pre-select only when freeform is disabled; otherwise just mark recommended.
		m.askUserSelectedIdx = -1
		if !msg.allowFreeform && len(msg.choices) > 0 {
			m.askUserSelectedIdx = 0
			for i, c := range msg.choices {
				if c == msg.recommended {
					m.askUserSelectedIdx = i
					break
				}
			}
		}

		// Save and reset the input so the user's draft doesn't bleed into the answer.
		m.askUserSavedDraft = m.input.Value()
		m.input.Reset()
		if msg.allowFreeform {
			m.input.Placeholder = "Type a custom answer, or ↑/↓ to select a choice…"
		} else {
			m.input.Placeholder = "Use ↑/↓ to select a choice and press Enter…"
		}

		m.askUserResultCh = msg.resultCh
		m.state = stateAwaitingUserQuestion
		m.askUserItemIdx = len(m.items)
		m.items = append(m.items, displayItem{kind: itemAskUser})
		needRebuild = true

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
		m.toolPicker.SetSize(m.width, m.height-fixedLines)
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
			if m.streamingItemIdx < 0 && chunk.Msg.Content != "" {
				m.items = append(m.items, displayItem{kind: itemAssistant, content: chunk.Msg.Content})
			}
			m.streamingItemIdx = -1
			// Skip empty assistant messages (no content, no tool calls) — they
			// provide no value and some providers reject them with a 400 error.
			if chunk.Msg.Content != "" || len(chunk.Msg.ToolCalls) > 0 {
				m.messages = append(m.messages, chunk.Msg)
			}
			if len(chunk.Msg.ToolCalls) == 0 {
				// Turn complete: drain queued prompts or go idle.
				needRebuild = true
				if m.sessionStore != nil {
					snap := m.currentSessionSnapshot()
					m.sessionID = snap.ID
					m.sessionCreatedAt = snap.CreatedAt
					cmds = append(cmds, doSaveSession(m.sessionStore, snap))
				}
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
		// If a toolResultMsg arrives while we're still showing the ask_user prompt
		// (e.g. ctx was cancelled), tear down the question display gracefully.
		if m.state == stateAwaitingUserQuestion {
			if m.askUserItemIdx >= 0 && m.askUserItemIdx < len(m.items) {
				m.items[m.askUserItemIdx] = displayItem{
					kind:    itemInfo,
					content: "❓ " + m.askUserQuestion + " (cancelled)",
				}
			}
			m.teardownAskUser()
			// needRebuild is set by the toolResultMsg error/success branches below.
		}
		if msg.err != nil {
			debugLog("toolResult: error tool=%q err=%v", msg.toolName, msg.err)
			// Check if this is a path approval request from the tools manager.
			var pathErr chat.PathApprovalError
			switch {
			case errors.As(msg.err, &pathErr) && m.executingCall != nil:
				reqs := pathErr.AccessRequests()
				// Queue path approvals; once all paths are approved the call re-runs.
				m.pendingCallAfterPaths = m.executingCall
				m.executingCall = nil
				m.currentPathRequest = reqs[0]
				m.pendingPathApprovals = reqs[1:]
				m.state = stateAwaitingPathApproval
				m.pendingApprovalChoice = ""
				needRebuild = true
			default:
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

	sep := separator(m.width)
	if m.session.AllowAll() {
		sep = separatorAllowAllStyle.Render(strings.Repeat("─", m.width))
	}

	var b strings.Builder

	b.WriteString(m.headerView())
	b.WriteString("\n")
	b.WriteString(sep)
	b.WriteString("\n")

	b.WriteString(m.viewport.View())
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

func (m Model) headerView() string {
	servers := "no servers"
	if len(m.serverNames) > 0 {
		servers = strings.Join(m.serverNames, ", ")
	}
	text := fmt.Sprintf("werkler  model: %s  servers: %s", m.modelName, servers)
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

	switch m.state {
	case stateThinking:
		return statusStyle.Render(m.spinner.View()+" Thinking…") + queueHint + allowAllIndicator, ""
	case stateConnectingMCP:
		return statusStyle.Render(m.spinner.View()+" Connecting to MCP servers…") + queueHint + allowAllIndicator, ""
	case stateConnectingOAuth:
		return statusStyle.Render(m.spinner.View()+" Waiting for OAuth authorization…") + queueHint + allowAllIndicator, ""
	case stateStreaming:
		return statusStyle.Render(m.spinner.View()+" Streaming…") + queueHint + allowAllIndicator, ""
	case stateCallingTool:
		name := toolNameStyle.Render(m.callingToolName)
		return statusStyle.Render(m.spinner.View()+" Calling tool: ") + name + queueHint + allowAllIndicator, ""
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
		args := formatArgsCompact(m.currentCall.Arguments)
		l1 := "  ▶ " + toolNameStyle.Render(m.currentCall.Name)
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
		line1 := mouseHint + pickerHint + sessionHint + allowAllIndicator
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

	case itemAskUser:
		var sb strings.Builder
		sb.WriteString(approvalPromptStyle.Render("❓") + "  " + m.askUserQuestion)
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

// --- Agent loop helpers ---

// approvePathRequest calls the appropriate session approval method.
// Execute and Write both grant write-level trust (execute implies arbitrary capability).
func (m *Model) approvePathRequest(req chat.PathAccessRequest) {
	if req.Write || req.Execute {
		m.session.ApprovePathWriteForSession(req.Path)
	} else {
		m.session.ApprovePathReadForSession(req.Path)
	}
}

// teardownAskUser clears all ask_user state and restores the saved input draft.
// Callers are responsible for updating the display item and setting m.state.
func (m *Model) teardownAskUser() {
	m.askUserCallID = ""
	m.askUserQuestion = ""
	m.askUserChoices = nil
	m.askUserRecommended = ""
	m.askUserResultCh = nil
	m.askUserItemIdx = -1
	m.askUserSelectedIdx = -1
	m.input.SetValue(m.askUserSavedDraft)
	m.askUserSavedDraft = ""
	m.syncViewportHeight()
}

// processNextPath advances the path approval queue.
// When all paths are approved, dispatch the pending tool call.
// Must only be called from Update.
func (m *Model) processNextPath() tea.Cmd {
	if len(m.pendingPathApprovals) > 0 {
		m.currentPathRequest = m.pendingPathApprovals[0]
		m.pendingPathApprovals = m.pendingPathApprovals[1:]
		m.state = stateAwaitingPathApproval
		m.pendingApprovalChoice = ""
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
	// ask_user, rubber_duck_review, and use_skill are always dispatched immediately
	// without an approval dialog: ask_user suspends the goroutine waiting for the user,
	// rubber_duck_review sends data only to a user-configured reviewer, and use_skill
	// only returns pre-computed skill content (no side effects).
	if m.session.IsApproved(call.Name) || call.Name == "ask_user" || call.Name == "rubber_duck_review" || call.Name == "use_skill" {
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
	m.pendingApprovalChoice = ""
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

// doListAllTools fetches the full (unfiltered) tool list for the tool picker.
func doListAllTools(ctx context.Context, session *chat.Session) tea.Cmd {
	return func() tea.Msg {
		tools, err := session.AllTools(ctx)
		if err != nil {
			return allToolsErrMsg{err: err}
		}
		return allToolsMsg{tools: tools}
	}
}

// doConnectMCPServer connects a single MCP server in the background and returns
// a mcpServerStatusMsg when done.
func doConnectMCPServer(ctx context.Context, mgr *mcppkg.Manager, srv config.MCPServerConfig) tea.Cmd {
	return func() tea.Msg {
		deferred, err := mgr.ConnectOne(ctx, srv)
		return mcpServerStatusMsg{name: srv.Name, deferred: deferred, err: err}
	}
}

// doRefreshMCPTools re-fetches the full tool list from the session (after MCP
// servers have finished connecting) and returns a mcpToolsRefreshedMsg.
func doRefreshMCPTools(ctx context.Context, session *chat.Session) tea.Cmd {
	return func() tea.Msg {
		tools, err := session.Tools(ctx)
		return mcpToolsRefreshedMsg{tools: tools, err: err}
	}
}

// filteredFromAllDefs returns the enabled subset of m.allToolDefs according to
// the current session disabled set. Used to refresh m.tools after picker changes.
func (m *Model) filteredFromAllDefs() []ai.ToolDefinition {
	if len(m.allToolDefs) == 0 {
		return m.tools // nothing was loaded; keep existing slice
	}
	out := make([]ai.ToolDefinition, 0, len(m.allToolDefs))
	for _, t := range m.allToolDefs {
		if m.session.IsToolEnabled(t.Name) {
			out = append(out, t)
		}
	}
	return out
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

// --- Session helpers ---

// formatAge returns a human-friendly relative age string.
func formatAge(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	case d < 48*time.Hour:
		return "yesterday"
	case d < 7*24*time.Hour:
		return fmt.Sprintf("%d days ago", int(d.Hours()/24))
	default:
		return t.Format("2006-01-02")
	}
}

// shortenHomePath replaces the home directory prefix with ~.
func shortenHomePath(p string) string {
	home, err := os.UserHomeDir()
	if err != nil {
		return p
	}
	if strings.HasPrefix(p, home+"/") || strings.HasPrefix(p, home+"\\") {
		return "~" + p[len(home):]
	}
	if p == home {
		return "~"
	}
	return p
}

// applySession restores a previously saved session into the model,
// rebuilding the display items from the stored message history.
func (m *Model) applySession(sess *sessionstore.Session) {
	m.sessionID = sess.ID
	m.sessionCreatedAt = sess.CreatedAt
	m.sessionCWD = sess.CWD
	m.messages = sess.Messages
	m.items = rebuildItemsFromMessages(sess.Messages)
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
// Errors are ignored (returns sessionSavedMsg regardless).
func doSaveSession(store *sessionstore.Store, sess sessionstore.Session) tea.Cmd {
	return func() tea.Msg {
		_ = store.Save(&sess)
		return sessionSavedMsg{}
	}
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
	// Generate a title from the first user message if possible.
	title := sessionstore.GenerateTitle(m.messages)
	return sessionstore.Session{
		ID:        id,
		Title:     title,
		CWD:       m.sessionCWD,
		Messages:  m.messages,
		CreatedAt: createdAt,
		UpdatedAt: now,
	}
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
	opts SessionOptions,
) error {
	// Fetch initially-available tools (builtin only when background MCP is used).
	sessionTools, err := session.Tools(ctx)
	if err != nil {
		return fmt.Errorf("fetching tools: %w", err)
	}

	// Resolve glamour style before starting bubbletea.
	glamourStyle := resolveGlamourStyle()

	// Enable debug logging to a file if WERKLER_DEBUG_LOG is set.
	if logPath := os.Getenv("WERKLER_DEBUG_LOG"); logPath != "" {
		f, ferr := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
		if ferr == nil {
			debugLogger = log.New(f, "", log.Ltime|log.Lmicroseconds)
			defer func() { _ = f.Close() }()
		}
	}

	var sendFn func(tea.Msg)
	sendFn = func(msg tea.Msg) {}

	m := initialModel(ctx, client, session, sessionTools, modelName, serverNames, glamourStyle, sendFn)

	// Apply skills so the system prompt hint is correct from the first message.
	m.skills = opts.Skills
	if len(opts.Skills) > 0 {
		m.messages = m.newConversation()
	}

	// Set up background MCP connection when servers are provided.
	if opts.MCPManager != nil && len(opts.MCPServers) > 0 {
		m.mcpManager = opts.MCPManager
		m.mcpPending = len(opts.MCPServers)
		m.mcpConnectingInfoIdx = len(m.items)
		m.items = append(m.items, displayItem{
			kind:    itemInfo,
			content: fmt.Sprintf("⟳ Connecting to %d MCP server(s)…", len(opts.MCPServers)),
		})
		m.state = stateConnectingMCP
		m.input.Placeholder = inputPlaceholder(stateConnectingMCP)
		for _, srv := range opts.MCPServers {
			m.mcpInitCmds = append(m.mcpInitCmds, doConnectMCPServer(ctx, opts.MCPManager, srv))
		}
	}

	// Apply session persistence options.
	m.sessionStore = opts.Store
	m.persistToolApproval = opts.PersistToolApproval
	m.persistPathApproval = opts.PersistPathApproval
	if opts.Initial != nil {
		m.applySession(opts.Initial)
	} else if opts.OpenPicker && opts.Store != nil && m.state == stateIdle {
		// Pre-schedule opening the session picker as the first action.
		// Skip when MCP background connection is running (stateConnectingMCP takes priority).
		m.state = statePickingSession
	}

	var prog *tea.Program
	sendFn = func(msg tea.Msg) {
		if prog != nil {
			prog.Send(msg)
		}
	}
	m.send = sendFn

	if toolMgr != nil {
		toolMgr.SetOutputNotify(func(handle, raw, clean string) {
			sendFn(processOutputMsg{handle: handle, raw: raw, clean: clean})
		})
		toolMgr.SetUserAsker(func(ctx context.Context, question string, choices []string, recommended string, allowFreeform bool) (string, error) {
			resultCh := make(chan askUserResult, 1)
			if m.executingCall != nil {
				sendFn(askUserMsg{
					callID:        m.executingCall.ID,
					question:      question,
					choices:       choices,
					recommended:   recommended,
					allowFreeform: allowFreeform,
					resultCh:      resultCh,
				})
			}
			select {
			case r := <-resultCh:
				return r.answer, r.err
			case <-ctx.Done():
				return "", ctx.Err()
			}
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

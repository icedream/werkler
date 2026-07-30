package ui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"charm.land/bubbles/v2/list"
	"charm.land/bubbles/v2/spinner"
	"charm.land/bubbles/v2/textarea"
	"charm.land/bubbles/v2/textinput"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/glamour/v2"
	"charm.land/lipgloss/v2"

	"github.com/icedream/werkler/internal/agents"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/config"
	mcppkg "github.com/icedream/werkler/internal/mcp"
	"github.com/icedream/werkler/internal/memorystore"
	"github.com/icedream/werkler/internal/sessionstore"
	"github.com/icedream/werkler/internal/skills"
	"github.com/icedream/werkler/internal/todostore"
	"github.com/icedream/werkler/internal/tools"
)

// --- Model ---

// Model is the bubbletea model for the interactive TUI.
type Model struct {
	// --- Token speed display fields ---
	streamingStart time.Time // when current streaming response began
	streamedTokens int       // token count for live stream
	showTokenSpeed bool      // toggle: show live tokens/sec
	lastTokenSpeed float64   // summary: last turn's tokens/sec

	ctx             context.Context
	client          ai.StreamCompleter
	session         *chat.Session
	tools           []ai.ToolDefinition
	toolTokensCache int // schema-only token overhead for m.tools; updated by setTools()

	// modelManager is optionally set when the client also implements ModelManager.
	// When nil, the model picker feature is disabled.
	modelManager ai.ModelManager

	// send dispatches messages to this bubbletea program from goroutines.
	// Set in RunTUI before the program starts.
	send func(tea.Msg)

	// Conversation history (managed only in Update, never in tea.Cmd goroutines).
	messages []ai.Message

	// Incremental mode for context reuse across tool calls.
	// When true, doStream uses CompleteStreamIncremental instead of full-context
	// doStartStream, sending only new messages (tool results / user turns) via
	// previous_response_id after the first response.
	incrementalModeEnabled bool
	incrementalClient      *ai.IncrementalClient

	// turnSystemMsg caches the fully-built system message content (timestamp,
	// memory, MCP server list, autopilot note) for the duration of the current
	// turn.  It is cleared whenever a new user message starts a fresh turn so
	// the first API call of the turn always gets a fresh timestamp/memory, but
	// every *continuation* call within the same turn (e.g. sending a tool result
	// back to the model) reuses the identical string.  This keeps the KV-cache
	// prefix byte-for-byte stable so local servers (e.g. llama.cpp) can skip
	// re-encoding the already-cached tokens and only process the new ones.
	turnSystemMsg string

	// Agent state machine.
	state            tuiState
	pendingCalls     []ai.ToolCall
	currentCall      *ai.ToolCall
	callingToolName  string // name of tool currently executing (stateCallingTool)
	callingToolTitle string // AI-provided title for the current tool call (may be empty)
	streamingItemIdx int    // index into items of the in-progress assistant item; -1 if none
	reasoningItemIdx int    // index into items of the in-progress reasoning item; -1 if none

	// Compaction summary streaming state.
	compactSummaryItemIdx int // index of the live compact_summary item; -1 when idle
	compactSummaryAccum   strings.Builder
	compactSummaryFinal   string        // finalized summary text; consumed by applyCompaction
	compactSummaryPrevTok int           // contextUsage.Total before compaction started
	compactSavedItems     []displayItem // compact-summary display items snapshotted in Done tick

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
	// displayed=true means the itemUser bubble has already been added to m.items.
	queuedPrompts []queuedPrompt

	// pendingImages holds images staged via /image that will be attached to
	// the next outbound user message.
	pendingImages []ai.ImagePart

	// inputHistory holds all user prompts sent in this session (oldest first).
	// historyIdx is the current position when navigating (-1 = not navigating).
	// historyDraft is the unsent text saved when navigation begins, restored on exit.
	inputHistory []string
	historyIdx   int
	historyDraft string

	// Display items for the viewport.
	items        []displayItem
	toolCallIdx  map[string]int // callID → index in items
	oauthInfoIdx int            // index of the OAuth status item; -1 if none

	// Model picker (only valid during statePickingModel).
	modelPicker       list.Model
	pickerReturnState tuiState // state to restore when model picker is dismissed

	// Session picker (only valid during statePickingSession).
	sessionPicker list.Model

	// Tool picker (only valid during statePickingTools).
	toolPicker     list.Model
	allToolDefs    []ai.ToolDefinition // full unfiltered list, set on allToolsMsg
	toolDetailVP   viewport.Model      // detail view for a single tool
	toolDetailItem toolItem            // tool being inspected in stateViewingToolDetail

	// Session persistence.
	sessionStore     *sessionstore.Store
	sessionID        string    // current session's ID; empty until first save
	sessionCreatedAt time.Time // set on first save
	sessionCWD       string    // cwd at TUI startup (used for resume hint + new session)
	// resumeHint is the latest CWD-matching session found on startup.
	// Shown in the status bar until the first message is sent.
	resumeHint *sessionstore.Session

	// saveSeq is incremented each time a session save is requested.
	// Each doSaveSession call captures the seq at dispatch time; the
	// sessionSavedMsg handler discards arrivals with a seq older than the
	// highest confirmed seq, preventing stale goroutines from rolling back state.
	saveSeq              int
	lastConfirmedSaveSeq int

	// recountGen is incremented whenever contextUsage is set synchronously
	// (e.g. in processNextCall or applyCompaction).  Each recountContext()
	// goroutine captures the generation at dispatch time and the tokenCountMsg
	// handler discards results that are older than the current generation,
	// preventing a stale pre-compaction count from overwriting the correct
	// post-compaction value.
	recountGen int

	// persistToolApproval, if non-nil, saves a tool name to auto_approve_tools.
	persistToolApproval func(toolName string) error
	// persistPathApproval, if non-nil, saves a path to auto_approve_paths.
	persistPathApproval func(path string, write bool) error

	// pendingApprovalChoice is the staged choice in an approval dialog ("y", "a",
	// "p", or "n"). Empty means nothing is staged yet. Confirmed with Enter.
	pendingApprovalChoice string

	// ask_user state (stateAwaitingUserQuestion).
	askUserCallID             string
	askUserQuestion           string
	askUserChoices            []string
	askUserRecommended        string
	askUserAllowFreeform      bool
	askUserSelectedIdx        int // index of highlighted choice; -1 = freeform input active
	askUserItemIdx            int // index of the question display item in items; -1 if none
	askUserResultCh           chan<- askUserResult
	askUserSavedDraft         string // input text saved on entry, restored on exit
	askUserIsPlanConfirmation bool   // when true, answer submission applies mode/autopilot switch

	// Per-operation cancellation for in-flight streams and tool calls.
	// cancelOp is nil when idle. cancelPending is true after the first Esc
	// press, waiting for a second Esc to actually cancel.
	cancelOp      context.CancelFunc
	cancelPending bool

	// contextWindowOverride is set from SessionOptions when the config provides
	// an explicit context window size to use when the provider doesn't report one.
	contextWindowOverride int
	// disableReasoning mirrors SessionOptions.DisableReasoning; when true,
	// reasoning tools are hidden and no effort is ever sent.
	disableReasoning bool

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

	// modelInfo holds the most recently fetched model metadata (e.g. context
	// window size). It is refetched on model switches and used when building
	// the system prompt for new conversations.
	modelInfo ai.ModelInfo

	// lastRateLimits holds the most recent rate limit headers from the provider.
	// Zero when the provider has not returned rate limit data (e.g. Ollama).
	lastRateLimits ai.RateLimits

	// contextUsage holds the most recently computed token count for the current
	// message history. Updated asynchronously after each message change.
	contextUsage ai.TokenCount

	// turnRoundtrips counts how many AI completion calls have been made within
	// the current user turn. Resets when the user sends a new message. Used to
	// warn when the agent is looping excessively.
	turnRoundtrips int

	// emptyResponseRetries counts how many times a retry-with-nudge has been
	// attempted after an empty response within the current user turn. Resets
	// alongside turnRoundtrips at every new-turn boundary.
	emptyResponseRetries int

	// itemsBeforeRetry records len(m.items) just before an empty-response retry
	// nudge is dispatched. Used to roll back display items if the retry response
	// triggers programmatic auto-connect (so the recovery explanation is removed).
	itemsBeforeRetry int

	// autoCompactPending is set when context compaction was triggered
	// automatically (not by the user). After compaction completes, the
	// interrupted AI turn is restarted.
	autoCompactPending bool

	// mouseEnabled tracks whether the terminal mouse reporting mode is active.
	// When false, the terminal handles mouse events natively (text selection works).
	mouseEnabled bool

	// skills holds loaded skills, used to build the system prompt hint.
	skills         []skills.Skill
	disabledSkills map[string]bool // per-session skill toggles
	skillPicker    list.Model      // skill enable/disable picker
	toolMgr        *tools.Manager  // needed for mid-session skill updates

	// agents holds loaded custom agents.
	agents      []agents.Agent
	activeAgent *agents.Agent // nil if no agent is active

	// Agent wizard state.
	agentWizardDescribeTA textarea.Model     // textarea for AI-assisted description
	agentWizardManualTAs  [4]textinput.Model // name, description, when + instructions textarea
	agentWizardManualTA   textarea.Model     // instructions textarea (4th field)
	agentWizardManualIdx  int                // which field is focused (0-3)
	agentWizardGenerated  agents.Agent       // agent built from generation or form
	agentWizardGenErr     string             // non-empty if generation failed
	agentWizardReviewText string             // rendered TOML for review viewport
	agentWizardToolPicker list.Model         // tool picker for wizard
	agentWizardAllTools   []string           // all tool names (populated before wizard tools step)
	agentWizardExcluded   map[string]bool    // tools the user deselected
	agentWizardOverwrite  bool               // pending overwrite confirmation
	agentWizardSavedPath  string             // path of last saved agent file

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

	// Registry browser state (only valid during statePickingRegistry).
	registryPicker        list.Model
	registryNextCursor    string // non-empty if another page can be loaded
	registryCancelCtx     context.CancelFunc
	registryTab           int // 0 = browse, 1 = installed
	registrySearchInput   textinput.Model
	registrySearchSeq     uint64 // for stale-result detection
	registryInstalledList list.Model
	// configuredMCPServers is the mutable live list of all configured MCP servers.
	// Starts from opts.MCPServers and updated on add/remove in the registry browser.
	configuredMCPServers   []config.MCPServerConfig
	registryInstalledNames map[string]bool // set from configuredMCPServers; rebuilt on change
	// persistMCPServer, if non-nil, saves a server to the config file.
	persistMCPServer func(cfg config.MCPServerConfig) error
	// removeRegistryServer, if non-nil, removes a server from the config file.
	removeRegistryServer func(serverName string) error

	// Todo sidebar.
	todoStore    *todostore.Store
	sidebarOpen  bool
	sidebarWidth int // column width including 1-char separator; defaults to defaultSidebarWidth

	// collapsedHandles tracks which process output handles are collapsed.
	// By default all process outputs are collapsed (show first 2 lines).
	// Handle is un-collapsed via /expand <handle> or /expand (expands all).
	collapsedHandles map[string]bool

	// Cross-session project memory.
	memoryStore *memorystore.MemoryStore

	// autopilot fields
	autopilot       bool // autonomous loop currently active
	autopilotPaused bool // cap reached — waiting for user to resume
	autopilotCycle  int  // cycles completed since autopilot started
	autopilotMax    int  // configured cap (0 → autopilotDefaultMax)

	// currentTaskTitle is set by the AI via task_start and shown in the status
	// bar in place of the generic "Thinking…"/"Streaming…" text.
	// Cleared when the AI returns to idle or calls task_complete.
	currentTaskTitle string

	// sessionTitle is the human-readable title for the current session.
	// Initially set from GenerateTitle on the first user message; refined
	// asynchronously by a short AI call after the first text exchange.
	sessionTitle string
	// sessionTitleRefined is true once the async title refinement has been
	// dispatched (or suppressed, e.g. on resume with an existing title).
	// Set synchronously at dispatch time to prevent double-firing.
	sessionTitleRefined bool

	// thinkingStart records when we entered stateThinking for elapsed-time display.
	// Zero value means we are not in a thinking state.
	thinkingStart time.Time

	// showReasoning controls whether reasoning/thinking items are displayed.
	// Toggled by /reasoning. Defaults to true.
	showReasoning bool

	// activeMode is the currently active mode preset.
	activeMode chat.ResolvedMode
	// allModes is the full list of available modes for the /mode picker.
	allModes []chat.ResolvedMode
	// configuredModes holds the raw config entries for re-resolving modes on session restore.
	configuredModes []config.ModeConfig
	// implementationMode is the name of the mode preset to apply when the AI calls
	// confirm_plan and the user approves. Empty string means use the default mode.
	implementationMode string
	// modePicker is the list used for /mode selection.
	modePicker list.Model
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
	toolPickerM.Title = "Tools  [space] toggle  [enter] details"
	toolPickerM.SetShowStatusBar(false)
	toolPickerM.SetFilteringEnabled(true)
	toolPickerM.DisableQuitKeybindings()

	// Skill picker: same style as tool picker.
	skillDel := list.NewDefaultDelegate()
	skillPickerM := list.New(nil, skillDel, 0, 0)
	skillPickerM.Title = "Toggle skills  [space] enable/disable"
	skillPickerM.SetShowStatusBar(false)
	skillPickerM.SetFilteringEnabled(true)
	skillPickerM.DisableQuitKeybindings()

	// Mode picker: single-select, like the model picker.
	modeDel := list.NewDefaultDelegate()
	modePickerM := list.New(nil, modeDel, 0, 0)
	modePickerM.Title = "Select mode"
	modePickerM.SetShowStatusBar(false)
	modePickerM.SetFilteringEnabled(true)
	modePickerM.DisableQuitKeybindings()

	// Agent wizard tool picker.
	agentToolDel := list.NewDefaultDelegate()
	agentToolPickerM := list.New(nil, agentToolDel, 0, 0)
	agentToolPickerM.Title = "Restrict agent tools  [space] toggle  [enter] confirm"
	agentToolPickerM.SetShowStatusBar(false)
	agentToolPickerM.SetFilteringEnabled(true)
	agentToolPickerM.DisableQuitKeybindings()

	// Use modelManager when the client also implements ModelManager.
	var mm ai.ModelManager
	if m, ok := client.(ai.ModelManager); ok {
		mm = m
	}

	cwd, _ := os.Getwd()

	return Model{
		ctx:                    ctx,
		client:                 client,
		session:                session,
		tools:                  tools,
		send:                   send,
		modelManager:           mm,
		messages:               chat.NewConversation(),
		incrementalModeEnabled: true,
		incrementalClient:      ai.NewIncrementalClient(client),
		state:                  stateIdle,
		streamingItemIdx:       -1,
		reasoningItemIdx:       -1,
		compactSummaryItemIdx:  -1,
		oauthInfoIdx:           -1,
		askUserSelectedIdx:     -1,
		askUserItemIdx:         -1,
		historyIdx:             -1,
		toolCallIdx:            make(map[string]int),
		viewport:               viewport.New(viewport.WithWidth(0), viewport.WithHeight(0)),
		modelPicker:            picker,
		sessionPicker:          sessPicker,
		toolPicker:             toolPickerM,
		skillPicker:            skillPickerM,
		modePicker:             modePickerM,
		agentWizardToolPicker:  agentToolPickerM,
		agentWizardExcluded:    make(map[string]bool),
		disabledSkills:         make(map[string]bool),
		collapsedHandles:       make(map[string]bool),
		sidebarWidth:           defaultSidebarWidth,
		input:                  ti,
		spinner:                sp,
		modelName:              modelName,
		serverNames:            serverNames,
		glamourStyle:           glamourStyle,
		mouseEnabled:           true,
		sessionCWD:             cwd,
		showReasoning:          true,
		showTokenSpeed:         false,
	}
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
	if getter, ok := m.client.(ai.ModelInfoGetter); ok {
		cmds = append(cmds, doGetModelInfo(m.ctx, getter))
	}
	return tea.Batch(cmds...)
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	needRebuild := false

	switch msg := msg.(type) {
	case tea.PasteMsg:
		// Bracketed paste: insert pasted text into the input, collapsing newlines to spaces.
		text := strings.Map(func(r rune) rune {
			if r == '\n' || r == '\r' || r == '\t' {
				return ' '
			}
			return r
		}, string(msg.Content))
		m.input.SetValue(m.input.Value() + text)
		m.input.CursorEnd()
		needRebuild = true
	case tea.KeyPressMsg:
		if msg.String() == "ctrl+c" {
			return m, tea.Quit
		}

		// Alt+M toggles terminal mouse reporting so the user can select and copy text.
		if msg.String() == "alt+m" {
			m.mouseEnabled = !m.mouseEnabled
			needRebuild = true
		}

		// Ctrl+T toggles live token speed indicator.
		if msg.String() == "ctrl+t" {
			m.showTokenSpeed = !m.showTokenSpeed
			needRebuild = true
		}
		// ctrl+p opens the model picker from any quiescent state.
		if msg.String() == "ctrl+p" && m.modelManager != nil &&
			m.state != statePickingModel {
			switch m.state {
			case stateIdle, stateAwaitingUserQuestion:
				m.showCompletion = false
				cmds = append(cmds, m.openModelPicker(m.state))
			}
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
							if idx, ok := m.toolCallIdx[call.ID]; ok && idx >= 0 {
								m.items[idx].toolStatus = toolStatusDenied
								if h := m.items[idx].handle; h != "" {
									m.collapsedHandles[h] = true
								}
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
							m.setIdle()
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
						if idx, ok := m.toolCallIdx[call.ID]; ok && idx >= 0 {
							m.items[idx].toolStatus = toolStatusRunning
						}
						m.setCallingTool(call)
						m.currentCall = nil
						m.executingCall = &call
						m.state = stateCallingTool
						needRebuild = true
						cmds = append(cmds, doCallTool(m.newOpCtx(), m.toolMgr, m.session, call))
					case "a":
						call := *m.currentCall
						m.session.ApproveForSession(call.Name)
						if idx, ok := m.toolCallIdx[call.ID]; ok && idx >= 0 {
							m.items[idx].toolStatus = toolStatusRunning
						}
						m.setCallingTool(call)
						m.currentCall = nil
						m.executingCall = &call
						m.state = stateCallingTool
						needRebuild = true
						cmds = append(cmds, doCallTool(m.newOpCtx(), m.toolMgr, m.session, call))
					case "p":
						call := *m.currentCall
						m.session.ApproveForSession(call.Name)
						if err := m.persistToolApproval(call.Name); err != nil {
							m.items = append(m.items, displayItem{
								kind:    itemError,
								content: "Failed to save approval: " + err.Error(),
							})
						}
						if idx, ok := m.toolCallIdx[call.ID]; ok && idx >= 0 {
							m.items[idx].toolStatus = toolStatusRunning
						}
						m.setCallingTool(call)
						m.currentCall = nil
						m.executingCall = &call
						m.state = stateCallingTool
						needRebuild = true
						cmds = append(cmds, doCallTool(m.newOpCtx(), m.toolMgr, m.session, call))
					case "n":
						call := *m.currentCall
						if idx, ok := m.toolCallIdx[call.ID]; ok && idx >= 0 {
							m.items[idx].toolStatus = toolStatusDenied
							if h := m.items[idx].handle; h != "" {
								m.collapsedHandles[h] = true
							}
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
			switch msg.String() {
			case "enter":
				// When the completion popup is open: exact command name runs the
				// action, a command followed by args closes the popup and falls
				// through to text dispatch, and a partial prefix fills the input.
				handled := false
				if m.showCompletion {
					filtered := m.filteredCmds()
					if len(filtered) > 0 {
						selected := filtered[m.completionIdx]
						inputVal := m.input.Value()
						cmdFull := "/" + selected.name
						switch {
						case inputVal == cmdFull:
							// Exact match — run the command.
							cmds = append(cmds, m.runCompletion(selected)...)
							m.updateCompletion()
							handled = true
						case !strings.HasPrefix(inputVal, cmdFull+" "):
							// Partial prefix — fill the input without executing.
							m.input.SetValue(cmdFull)
							m.input.CursorEnd()
							m.updateCompletion()
							handled = true
						default:
							// Input is "/cmd args" — close popup WITHOUT calling
							// m.updateCompletion(), which would immediately re-open the
							// popup (filteredCmds still matches "/expand" in "/expand all")
							// causing the user to need a second Enter press.
							m.showCompletion = false
							m.syncViewportHeight()
						}
					}
				}
				// Check external commands before the built-in switch so they can
				// also be dispatched with arguments without being caught by text != "".
				if !handled {
					if inputText := strings.TrimSpace(m.input.Value()); inputText != "" {
						if ext, args, ok := matchExternal(inputText); ok {
							m.input.Reset()
							m.showCompletion = false
							m.updateCompletion()
							msg, extCmds := ext.Action(args)
							if msg != "" {
								m.items = append(m.items, displayItem{kind: itemInfo, content: msg})
								needRebuild = true
							}
							cmds = append(cmds, extCmds...)
							handled = true
						}
					}
				}
				if !handled {
					text := strings.TrimSpace(m.input.Value())
					switch {
					case text == "" && m.autopilotPaused:
						// Resume paused autopilot on bare Enter (empty input).
						m.autopilotPaused = false
						m.autopilotCycle = 0
						m.items = append(m.items, displayItem{kind: itemInfo, content: "⚡ Autopilot resumed."})
						needRebuild = true
						m.turnRoundtrips = 0
						m.turnSystemMsg = ""
						m.emptyResponseRetries = 0
						m.setThinking()
						cmds = append(cmds, m.doStream(m.newOpCtx(), m.buildStreamMessages(m.autopilotMessagesForStream()), m.tools))
					case strings.HasPrefix(text, "/image "):
						// /image <path-or-url> — load image and stage it as pending.
						arg := strings.TrimSpace(strings.TrimPrefix(text, "/image "))
						m.input.Reset()
						m.updateCompletion()
						if arg != "" {
							cmds = append(cmds, doLoadImage(arg))
						}
					case text == "/agent" || text == "/agent off":
						// /agent off (or bare /agent) -- deactivate current agent.
						m.input.Reset()
						m.updateCompletion()
						if m.activeAgent != nil {
							m.deactivateAgent()
							needRebuild = true
						} else if text == "/agent" {
							m.items = append(m.items, displayItem{kind: itemInfo, content: "No agent is currently active."})
							needRebuild = true
						}
					case strings.HasPrefix(text, "/agent "):
						// /agent <name> -- activate a named agent, or open wizard for "new".
						arg := strings.TrimSpace(strings.TrimPrefix(text, "/agent "))
						m.input.Reset()
						m.updateCompletion()
						switch arg {
						case "new":
							cmds = append(cmds, m.initAgentWizard()...)
						case "off":
							if m.activeAgent != nil {
								m.deactivateAgent()
								needRebuild = true
							}
						default:
							found := false
							for _, a := range m.agents {
								if a.Name == arg {
									m.activateAgent(a)
									needRebuild = true
									found = true
									break
								}
							}
							if !found {
								m.items = append(m.items, displayItem{
									kind:    itemError,
									content: fmt.Sprintf("Agent %q not found. Use /agent new to create one.", arg),
								})
								needRebuild = true
							}
						}
					case strings.HasPrefix(text, "/expand "):
						arg := strings.TrimSpace(strings.TrimPrefix(text, "/expand "))
						m.input.Reset()
						m.updateCompletion()
						if arg == "all" {
							for h := range m.collapsedHandles {
								m.collapsedHandles[h] = false
							}
						} else if arg != "" {
							m.collapsedHandles[arg] = false
						}
						needRebuild = true
					case strings.HasPrefix(text, "/collapse "):
						arg := strings.TrimSpace(strings.TrimPrefix(text, "/collapse "))
						m.input.Reset()
						m.updateCompletion()
						if arg == "all" {
							for h := range m.collapsedHandles {
								m.collapsedHandles[h] = true
							}
						} else if arg != "" {
							m.collapsedHandles[arg] = true
						}
						needRebuild = true
					case strings.HasPrefix(text, "/sidebar "):
						arg := strings.TrimSpace(strings.TrimPrefix(text, "/sidebar "))
						m.input.Reset()
						m.updateCompletion()
						const sidebarMin, sidebarMax = 20, 60
						switch arg {
						case "wider":
							if m.sidebarWidth < sidebarMax {
								m.sidebarWidth = min(m.sidebarWidth+4, sidebarMax)
								m.recalcLayout()
								needRebuild = true
							}
						case "narrower":
							if m.sidebarWidth > sidebarMin {
								m.sidebarWidth = max(m.sidebarWidth-4, sidebarMin)
								m.recalcLayout()
								needRebuild = true
							}
						case "reset":
							m.sidebarWidth = defaultSidebarWidth
							m.recalcLayout()
							needRebuild = true
						default:
							m.items = append(m.items, displayItem{kind: itemInfo, content: "Usage: /sidebar wider | /sidebar narrower | /sidebar reset"})
							needRebuild = true
						}
					case text != "":
						m.input.Reset()
						m.appendInputHistory(text)
						parts := m.pendingImages
						m.pendingImages = nil
						userContent := text
						if len(parts) > 0 {
							names := make([]string, len(parts))
							for i, p := range parts {
								names[i] = p.Name
							}
							userContent += " [📎 " + strings.Join(names, ", ") + "]"
						}
						m.items = append(m.items, displayItem{kind: itemUser, content: userContent})
						needRebuild = true

						if m.session.HasPendingOAuth() {
							// Defer the user prompt until OAuth servers are connected.
							// Item already shown above (displayed=true).
							m.queuedPrompts = append([]queuedPrompt{{text: text, parts: parts, displayed: true}}, m.queuedPrompts...)
							m.oauthInfoIdx = len(m.items)
							names := strings.Join(m.session.PendingOAuthNames(), ", ")
							m.items = append(m.items, displayItem{
								kind:    itemInfo,
								content: "Connecting to " + names + "…",
							})
							m.state = stateConnectingOAuth
							cmds = append(cmds, doConnectOAuth(m.ctx, m.session, m.send))
						} else {
							m.messages = append(m.messages, ai.Message{Role: "user", Content: text, Parts: parts})
							// Set initial session title from the first user message (synchronous).
							if m.sessionTitle == "" {
								m.sessionTitle = sessionstore.GenerateTitle(m.messages)
								if m.sessionStore != nil {
									cmds = append(cmds, m.saveSession())
								}
							}
							m.resumeHint = nil // dismiss hint once user starts chatting
							m.turnRoundtrips = 0
							m.turnSystemMsg = ""
							m.emptyResponseRetries = 0
							// Synchronously count the outbound payload before deciding
							// whether to compact, so the decision is current rather
							// than one turn stale.
							outbound := m.buildStreamMessages(m.messages)
							if count, cerr := ai.CountTokensWithTools(m.modelName, outbound, m.tools); cerr == nil {
								m.recountGen++
								m.contextUsage = count
							}
							if cmd := m.recountContext(); cmd != nil {
								cmds = append(cmds, cmd)
							}
							if m.shouldAutoCompact() {
								m.autoCompactPending = true
								m.state = stateCompacting
								cmds = append(cmds, doCompact(m.newOpCtx(), m.client, m.messages, m.modelName, m.toolTokensCache, m.modelInfo.Context.MaxTokens))
							} else {
								m.setThinking()
								cmds = append(cmds, m.doStream(m.newOpCtx(), m.buildStreamMessages(m.messages), m.tools))
							}
						}
					}
				}

			case "tab":
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

			case "shift+tab":
				if len(m.allModes) > 0 {
					cmds = append(cmds, m.cycleMode()...)
				}
			case "up":
				switch {
				case m.showCompletion:
					filtered := m.filteredCmds()
					if len(filtered) > 0 {
						m.completionIdx = (m.completionIdx - 1 + len(filtered)) % len(filtered)
					}
				case len(m.inputHistory) > 0:
					// History navigation takes priority over viewport scroll.
					m.historyUp()
					m.updateCompletion()
				default:
					var vpCmd tea.Cmd
					m.viewport, vpCmd = m.viewport.Update(msg)
					cmds = append(cmds, vpCmd)
				}

			case "down":
				switch {
				case m.showCompletion:
					filtered := m.filteredCmds()
					if len(filtered) > 0 {
						m.completionIdx = (m.completionIdx + 1) % len(filtered)
					}
				case m.historyIdx != -1:
					// Only intercept Down when actively navigating history.
					m.historyDown()
					m.updateCompletion()
				default:
					var vpCmd tea.Cmd
					m.viewport, vpCmd = m.viewport.Update(msg)
					cmds = append(cmds, vpCmd)
				}

			case "esc":
				if m.showCompletion {
					m.showCompletion = false
					m.syncViewportHeight()
				} else if m.input.Value() != "" {
					m.input.Reset()
					m.updateCompletion()
				}

			case "ctrl+r":
				if m.sessionStore != nil {
					m.showCompletion = false
					m.state = statePickingSession
					m.sessionPicker.SetItems(nil)
					m.sessionPicker.SetSize(m.width, m.height-fixedLines)
					cmds = append(cmds, doLoadSessions(m.sessionStore))
				}

			case "pgup", "pgdn":
				// Explicit page scroll — forward to viewport only (not to textinput).
				var vpCmd tea.Cmd
				m.viewport, vpCmd = m.viewport.Update(msg)
				cmds = append(cmds, vpCmd)

			default:
				// Forward to textinput only. Do NOT forward to viewport: the viewport's
				// default keymap binds printable runes (b, f, space, u, d, j, k, …) as
				// scroll keys; forwarding them here would scroll the view while typing.
				// Viewport scrolling is handled via arrow keys and PgUp/PgDn above.
				var inputCmd tea.Cmd
				m.input, inputCmd = m.input.Update(msg)
				cmds = append(cmds, inputCmd)
				m.updateCompletion()
			}

		case statePickingModel:
			switch msg.String() {
			case "esc", "ctrl+c":
				if m.modelPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.modelPicker, pickerCmd = m.modelPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					m.returnFromModelPicker()
				}
			case "enter":
				if m.modelPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.modelPicker, pickerCmd = m.modelPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					if sel := m.modelPicker.SelectedItem(); sel != nil {
						item := sel.(modelItem)
						m.modelManager.SetModel(item.ModelItem)
						m.modelName = item.Display()
						// Invalidate cached model info and re-fetch for the new model.
						m.modelInfo = ai.ModelInfo{}
						if getter, ok := m.client.(ai.ModelInfoGetter); ok {
							cmds = append(cmds, doGetModelInfo(m.ctx, getter))
						}
					}
					m.returnFromModelPicker()
				}
			default:
				var pickerCmd tea.Cmd
				m.modelPicker, pickerCmd = m.modelPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			}

		case statePickingSession:
			switch msg.String() {
			case "esc", "ctrl+c":
				if m.sessionPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.sessionPicker, pickerCmd = m.sessionPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					m.setIdle()
					m.updateCompletion()
				}
			case "enter":
				if m.sessionPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.sessionPicker, pickerCmd = m.sessionPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					if sel := m.sessionPicker.SelectedItem(); sel != nil {
						switch item := sel.(type) {
						case newSessionItem:
							m.messages = m.newConversation()
							m.items = nil
							m.toolCallIdx = make(map[string]int)
							m.streamingItemIdx = -1
							m.reasoningItemIdx = -1
							m.oauthInfoIdx = -1
							m.sessionID = ""
							m.sessionCreatedAt = time.Time{}
							needRebuild = true
						case sessionItem:
							m.resumeHint = nil
							m.applySession(&item.sess)
							needRebuild = true
							if cmd := m.recountContext(); cmd != nil {
								cmds = append(cmds, cmd)
							}
						}
					}
					m.setIdle()
					m.updateCompletion()
				}
			default:
				var pickerCmd tea.Cmd
				m.sessionPicker, pickerCmd = m.sessionPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			}

		case statePickingTools:
			switch msg.String() {
			case "esc", "ctrl+c":
				if m.toolPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.toolPicker, pickerCmd = m.toolPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					// Refresh m.tools from the (possibly updated) disabled set and go back.
					m.setTools(m.filteredFromAllDefs())
					m.setIdle()
					m.updateCompletion()
				}
			case "enter":
				if m.toolPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.toolPicker, pickerCmd = m.toolPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					// Open detail view for the selected tool.
					if sel := m.toolPicker.SelectedItem(); sel != nil {
						item := sel.(toolItem)
						m.toolDetailItem = item
						m.toolDetailVP = viewport.New(viewport.WithWidth(m.width), viewport.WithHeight(m.height-fixedLines))
						m.toolDetailVP.SetContent(buildToolDetail(item, m.width))
						m.state = stateViewingToolDetail
					}
				}
			case " ":
				if m.toolPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.toolPicker, pickerCmd = m.toolPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					// Toggle the currently selected tool.
					if sel := m.toolPicker.SelectedItem(); sel != nil {
						item := sel.(toolItem)
						item.enabled = !item.enabled
						m.session.SetToolEnabled(item.name, item.enabled)
						m.toolPicker.SetItem(m.toolPicker.Index(), item)
					}
				}
			case "":
				var pickerCmd tea.Cmd
				m.toolPicker, pickerCmd = m.toolPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			default:
				var pickerCmd tea.Cmd
				m.toolPicker, pickerCmd = m.toolPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			}

		case stateViewingToolDetail:
			switch msg.String() {
			case "esc", "ctrl+c", "enter":
				m.state = statePickingTools
			default:
				var vpCmd tea.Cmd
				m.toolDetailVP, vpCmd = m.toolDetailVP.Update(msg)
				cmds = append(cmds, vpCmd)
			}

		case statePickingSkills:
			switch msg.String() {
			case "esc", "ctrl+c":
				if m.skillPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.skillPicker, pickerCmd = m.skillPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					m.setIdle()
					m.updateCompletion()
				}
			case " ":
				if m.skillPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.skillPicker, pickerCmd = m.skillPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					if sel := m.skillPicker.SelectedItem(); sel != nil {
						item := sel.(skillItem)
						item.enabled = !item.enabled
						m.disabledSkills[item.name] = !item.enabled
						m.skillPicker.SetItem(m.skillPicker.Index(), item)
						m.applySkillToggle()
					}
				}
			case "":
				var pickerCmd tea.Cmd
				m.skillPicker, pickerCmd = m.skillPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			default:
				var pickerCmd tea.Cmd
				m.skillPicker, pickerCmd = m.skillPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			}

		case statePickingMode:
			switch msg.String() {
			case "esc", "ctrl+c":
				if m.modePicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.modePicker, pickerCmd = m.modePicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					m.setIdle()
					m.updateCompletion()
				}
			case "enter":
				if m.modePicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.modePicker, pickerCmd = m.modePicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					if sel := m.modePicker.SelectedItem(); sel != nil {
						item := sel.(modeItem)
						m.applyMode(item.mode)
					}
					m.setIdle()
					m.updateCompletion()
				}
			default:
				var pickerCmd tea.Cmd
				m.modePicker, pickerCmd = m.modePicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			}

		// --- Agent wizard states ---

		case stateAgentWizardMode:
			// Choose AI-assisted (a) or manual (m), or Esc to cancel.
			switch msg.String() {
			case "esc", "ctrl+c":
				m.setIdle()
				m.updateCompletion()
			case "":
				switch msg.String() {
				case "a", "A":
					m.state = stateAgentWizardDescribe
					cmds = append(cmds, m.agentWizardDescribeTA.Focus())
				case "m", "M":
					m.state = stateAgentWizardManual
					m.agentWizardManualTAs[0].Focus()
				}
			}
			needRebuild = true

		case stateAgentWizardDescribe:
			switch msg.String() {
			case "esc", "ctrl+c":
				m.setIdle()
				m.updateCompletion()
			case "ctrl+s":
				// Ctrl+S submits the description.
				desc := strings.TrimSpace(m.agentWizardDescribeTA.Value())
				if desc != "" {
					m.state = stateAgentWizardGenerate
					c, ok := m.client.(ai.Completer)
					if !ok {
						m.agentWizardGenErr = "The active AI provider does not support non-streaming completions required for profile generation. Please use the manual form (restart wizard and press 'm')."
						m.state = stateAgentWizardReview
					} else {
						cmds = append(cmds, doGenerateAgent(m.ctx, c, desc))
					}
				}
			default:
				var taCmd tea.Cmd
				m.agentWizardDescribeTA, taCmd = m.agentWizardDescribeTA.Update(msg)
				cmds = append(cmds, taCmd)
			}
			needRebuild = true

		case stateAgentWizardGenerate:
			// Spinner; awaiting agentGeneratedMsg.
			var spCmd tea.Cmd
			m.spinner, spCmd = m.spinner.Update(msg)
			cmds = append(cmds, spCmd)
			if msg.String() == "esc" || msg.String() == "ctrl+c" {
				m.setIdle()
				m.updateCompletion()
			}
			needRebuild = true

		case stateAgentWizardReview:
			switch msg.String() {
			case "esc", "ctrl+c":
				m.setIdle()
				m.updateCompletion()
			case "":
				switch msg.String() {
				case "c", "C":
					// Continue to tool picker.
					cmds = append(cmds, m.startAgentWizardToolPicker()...)
				case "e", "E":
					// Edit in manual form pre-filled with generated values.
					m.prefillManualFormFromGenerated()
					m.state = stateAgentWizardManual
				case "r", "R":
					// Retry generation.
					m.agentWizardGenErr = ""
					desc := strings.TrimSpace(m.agentWizardDescribeTA.Value())
					if desc != "" {
						m.state = stateAgentWizardGenerate
						c, ok := m.client.(ai.Completer)
						if !ok {
							m.agentWizardGenErr = "Provider does not support non-streaming completions."
							m.state = stateAgentWizardReview
						} else {
							cmds = append(cmds, doGenerateAgent(m.ctx, c, desc))
						}
					} else {
						m.state = stateAgentWizardDescribe
					}
				}
			}
			needRebuild = true

		case stateAgentWizardManual:
			switch msg.String() {
			case "esc", "ctrl+c":
				m.setIdle()
				m.updateCompletion()
			case "tab", "enter":
				switch {
				case m.agentWizardManualIdx < 2:
					// Advance through the single-line fields.
					m.agentWizardManualTAs[m.agentWizardManualIdx].Blur()
					m.agentWizardManualIdx++
					m.agentWizardManualTAs[m.agentWizardManualIdx].Focus()
				case m.agentWizardManualIdx == 2 && msg.String() == "tab":
					// Tab from 3rd single-line field -> instructions textarea.
					m.agentWizardManualTAs[2].Blur()
					m.agentWizardManualIdx = 3
					cmds = append(cmds, m.agentWizardManualTA.Focus())
				case m.agentWizardManualIdx == 3 && msg.String() == "tab":
					// Tab from instructions -> submit (validate & proceed).
					cmds = append(cmds, m.submitManualForm()...)
				}
			case "shift+tab":
				if m.agentWizardManualIdx == 3 {
					m.agentWizardManualTA.Blur()
					m.agentWizardManualIdx = 2
					m.agentWizardManualTAs[2].Focus()
				} else if m.agentWizardManualIdx > 0 {
					m.agentWizardManualTAs[m.agentWizardManualIdx].Blur()
					m.agentWizardManualIdx--
					m.agentWizardManualTAs[m.agentWizardManualIdx].Focus()
				}
			default:
				if m.agentWizardManualIdx < 3 {
					var tiCmd tea.Cmd
					m.agentWizardManualTAs[m.agentWizardManualIdx], tiCmd = m.agentWizardManualTAs[m.agentWizardManualIdx].Update(msg)
					cmds = append(cmds, tiCmd)
				} else {
					var taCmd tea.Cmd
					m.agentWizardManualTA, taCmd = m.agentWizardManualTA.Update(msg)
					cmds = append(cmds, taCmd)
				}
			}
			needRebuild = true

		case stateAgentWizardTools:
			switch msg.String() {
			case "esc", "ctrl+c":
				if m.agentWizardToolPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.agentWizardToolPicker, pickerCmd = m.agentWizardToolPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					m.setIdle()
					m.updateCompletion()
				}
			case "enter":
				if m.agentWizardToolPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.agentWizardToolPicker, pickerCmd = m.agentWizardToolPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					// Confirm tool selection and save.
					cmds = append(cmds, m.saveAgentFromWizard()...)
				}
			case " ":
				if m.agentWizardToolPicker.SettingFilter() {
					var pickerCmd tea.Cmd
					m.agentWizardToolPicker, pickerCmd = m.agentWizardToolPicker.Update(msg)
					cmds = append(cmds, pickerCmd)
				} else {
					if sel := m.agentWizardToolPicker.SelectedItem(); sel != nil {
						item := sel.(agentToolItem)
						if !item.alwaysOn {
							item.enabled = !item.enabled
							m.agentWizardExcluded[item.name] = !item.enabled
							m.agentWizardToolPicker.SetItem(m.agentWizardToolPicker.Index(), item)
						}
					}
				}
			case "":
				var pickerCmd tea.Cmd
				m.agentWizardToolPicker, pickerCmd = m.agentWizardToolPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			default:
				var pickerCmd tea.Cmd
				m.agentWizardToolPicker, pickerCmd = m.agentWizardToolPicker.Update(msg)
				cmds = append(cmds, pickerCmd)
			}

		case stateAgentWizardDone:
			// Any key returns to idle.
			m.setIdle()
			m.updateCompletion()
			needRebuild = true

		case statePickingRegistry:
			switch msg.String() {
			case "esc", "ctrl+c":
				// On browse tab: Esc clears search first, then closes on second press.
				if m.registryTab == 0 && m.registrySearchInput.Value() != "" {
					m.registrySearchInput.Reset()
					if m.registryCancelCtx != nil {
						m.registryCancelCtx()
					}
					ctx, cancel := context.WithCancel(m.ctx)
					m.registryCancelCtx = cancel
					m.registrySearchSeq++
					cmds = append(cmds, doFetchRegistry(ctx, m.registrySearchSeq, "", ""))
				} else {
					if m.registryCancelCtx != nil {
						m.registryCancelCtx()
						m.registryCancelCtx = nil
					}
					m.setIdle()
					m.updateCompletion()
					cmds = append(cmds, m.input.Focus())
				}

			case "tab":
				m.registryTab = 1 - m.registryTab

			case "enter":
				if m.registryTab == 0 {
					sel := m.registryPicker.SelectedItem()
					if sel == nil {
						break
					}
					ri := sel.(registryItem)
					if ri.installed {
						m.items = append(m.items, displayItem{
							kind:    itemInfo,
							content: `"` + ri.srv.Title + `" is already installed.`,
						})
						needRebuild = true
						break
					}
					if ri.srv.HasPackage {
						m.items = append(m.items, displayItem{
							kind:    itemInfo,
							content: `Server "` + ri.srv.Title + `" requires a local package install and cannot be added automatically. See: https://registry.modelcontextprotocol.io`,
						})
						needRebuild = true
						break
					}
					if m.persistMCPServer == nil {
						m.items = append(m.items, displayItem{
							kind:    itemError,
							content: "Config persistence is not available in this session.",
						})
						needRebuild = true
						break
					}
					if ri.srv.FirstRemoteURL() == "" {
						m.items = append(m.items, displayItem{
							kind:    itemInfo,
							content: `Server "` + ri.srv.Title + `" has no streamable HTTP endpoint and cannot be added automatically. See: https://registry.modelcontextprotocol.io`,
						})
						needRebuild = true
						break
					}
					cmds = append(cmds, doSaveMCPServer(m.ctx, ri.srv, m.persistMCPServer))
				}

			case "ctrl+d":
				if m.registryTab == 1 {
					sel := m.registryInstalledList.SelectedItem()
					if sel == nil {
						break
					}
					srv := sel.(registryInstalledItem).srv
					if m.removeRegistryServer == nil {
						m.items = append(m.items, displayItem{
							kind:    itemError,
							content: "Config persistence is not available in this session.",
						})
						needRebuild = true
						break
					}
					cmds = append(cmds, doRemoveMCPServer(srv.Name, m.removeRegistryServer))
				}

			default:
				if m.registryTab == 0 {
					// Text keys go to search input; nav keys go to list.
					prev := m.registrySearchInput.Value()
					var searchCmd tea.Cmd
					m.registrySearchInput, searchCmd = m.registrySearchInput.Update(msg)
					cmds = append(cmds, searchCmd)
					if m.registrySearchInput.Value() != prev {
						if m.registryCancelCtx != nil {
							m.registryCancelCtx()
						}
						ctx, cancel := context.WithCancel(m.ctx)
						m.registryCancelCtx = cancel
						m.registrySearchSeq++
						cmds = append(cmds, doFetchRegistry(ctx, m.registrySearchSeq, m.registrySearchInput.Value(), ""))
					}
					switch msg.String() {
					case "up", "down", "pgup", "pgdn":
						var listCmd tea.Cmd
						m.registryPicker, listCmd = m.registryPicker.Update(msg)
						cmds = append(cmds, listCmd)
					}
				} else {
					var listCmd tea.Cmd
					m.registryInstalledList, listCmd = m.registryInstalledList.Update(msg)
					cmds = append(cmds, listCmd)
				}
			}

		case stateConnectingMCP:
			// Allow viewport scrolling and prompt queuing; block submission until connected.
			switch msg.String() {
			case "enter":
				text := strings.TrimSpace(m.input.Value())
				if text != "" {
					m.input.Reset()
					m.appendInputHistory(text)
					m.items = append(m.items, displayItem{kind: itemUser, content: text})
					m.queuedPrompts = append(m.queuedPrompts, queuedPrompt{text: text, displayed: true})
					needRebuild = true
				}
			case "esc":
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

		case stateThinking, stateStreaming, stateCallingTool, stateConnectingOAuth, stateCompacting:
			switch msg.String() {
			case "enter":
				text := strings.TrimSpace(m.input.Value())
				// Handle slash commands with arguments immediately — before the
				// completion-popup path (which only runs the no-arg action) and
				// before the queueing path (which would send them to the AI).
				switch {
				case strings.HasPrefix(text, "/expand "):
					arg := strings.TrimSpace(strings.TrimPrefix(text, "/expand "))
					m.input.Reset()
					m.showCompletion = false
					m.updateCompletion()
					if arg == "all" {
						for h := range m.collapsedHandles {
							m.collapsedHandles[h] = false
						}
					} else if arg != "" {
						m.collapsedHandles[arg] = false
					}
					needRebuild = true
				case strings.HasPrefix(text, "/collapse "):
					arg := strings.TrimSpace(strings.TrimPrefix(text, "/collapse "))
					m.input.Reset()
					m.showCompletion = false
					m.updateCompletion()
					if arg == "all" {
						for h := range m.collapsedHandles {
							m.collapsedHandles[h] = true
						}
					} else if arg != "" {
						m.collapsedHandles[arg] = true
					}
					needRebuild = true
				case strings.HasPrefix(text, "/image "):
					arg := strings.TrimSpace(strings.TrimPrefix(text, "/image "))
					m.input.Reset()
					m.showCompletion = false
					m.updateCompletion()
					if arg != "" {
						cmds = append(cmds, doLoadImage(arg))
					}
					needRebuild = true
				default:
					// External commands that are safeWhileBusy can run immediately.
					if ext, args, ok := matchExternal(text); ok && ext.SafeWhileBusy {
						m.input.Reset()
						m.showCompletion = false
						m.updateCompletion()
						msg, extCmds := ext.Action(args)
						if msg != "" {
							m.items = append(m.items, displayItem{kind: itemInfo, content: msg})
							needRebuild = true
						}
						cmds = append(cmds, extCmds...)
						break
					}
					// If the completion popup is showing and a safe-while-busy command
					// is selected (exact name, no args), run it immediately.
					if m.showCompletion {
						filtered := m.filteredCmds()
						if m.completionIdx < len(filtered) {
							cmds = append(cmds, m.runCompletion(filtered[m.completionIdx])...)
							break
						}
					}
					if text != "" {
						m.input.Reset()
						m.showCompletion = false
						m.appendInputHistory(text)
						parts := m.pendingImages
						m.pendingImages = nil
						// Do NOT add to m.items here — the message should appear in
						// the chat log at the moment it is actually sent to the AI.
						m.queuedPrompts = append(m.queuedPrompts, queuedPrompt{text: text, parts: parts, displayed: false})
						needRebuild = true
					}
				}
			case "tab":
				if m.showCompletion {
					filtered := m.filteredCmds()
					if len(filtered) > 0 {
						m.input.SetValue("/" + filtered[m.completionIdx].name)
						m.input.CursorEnd()
						m.updateCompletion()
					}
				}
			case "shift+tab":
				if len(m.allModes) > 0 {
					cmds = append(cmds, m.cycleMode()...)
				}
			case "up":
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
			case "down":
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
			case "esc":
				switch {
				case m.showCompletion:
					m.showCompletion = false
					m.syncViewportHeight()
				case m.input.Value() != "":
					m.input.Reset()
					m.cancelPending = false
				case len(m.queuedPrompts) > 0:
					m.queuedPrompts = m.queuedPrompts[:len(m.queuedPrompts)-1]
					m.cancelPending = false
					needRebuild = true
				case m.state != stateConnectingOAuth && m.state != stateCompacting && m.cancelOp != nil:
					if m.cancelPending {
						// Second Esc — cancel the operation.
						m.cancelOp()
						m.cancelOp = nil
						m.cancelPending = false
					} else {
						// First Esc — arm cancellation.
						m.cancelPending = true
						needRebuild = true
					}
				}
			default:
				m.cancelPending = false
				// Forward to both: textinput handles text-entry keys (runes, backspace,
				// cursor movement); viewport handles navigation keys (Up/Down/PgUp/PgDn).
				// Single-line textinput ignores directional keys, so double-routing is safe.
				var inputCmd tea.Cmd
				m.input, inputCmd = m.input.Update(msg)
				cmds = append(cmds, inputCmd)
				m.updateCompletion()
				var vpCmd tea.Cmd
				m.viewport, vpCmd = m.viewport.Update(msg)
				cmds = append(cmds, vpCmd)
			}

		case stateAwaitingUserQuestion:
			switch msg.String() {
			case "up":
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
			case "down":
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
			case "enter":
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
				isPlanConfirm := m.askUserIsPlanConfirmation
				m.teardownAskUser()
				m.state = stateCallingTool
				needRebuild = true
				if isPlanConfirm {
					// TUI owns mode/autopilot — apply before returning to the AI.
					switch answer {
					case "Implement it":
						targetMode, err := chat.ResolveMode(m.implementationMode, m.configuredModes)
						if err == nil {
							m.applyMode(targetMode)
						}
						if m.autopilot || m.autopilotPaused {
							m.autopilotDisable()
						}
						answer = "approved: proceed with implementation"
					case "Implement it with autopilot":
						targetMode, err := chat.ResolveMode(m.implementationMode, m.configuredModes)
						if err == nil {
							m.applyMode(targetMode)
						}
						m.autopilotEnable()
						answer = "approved_with_autopilot: autopilot will continue — stop your current response"
					default:
						if !strings.HasPrefix(answer, "rejected:") {
							answer = "rejected: " + answer
						}
					}
				}
				if ch != nil {
					ch <- askUserResult{answer: answer}
				}
			default:
				if m.askUserAllowFreeform || len(m.askUserChoices) == 0 {
					// Typing any rune deselects any highlighted choice.
					if msg.Text != "" && m.askUserSelectedIdx >= 0 {
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
			cmds = append(cmds, doRefreshMCPTools(m.ctx, m.session, false))
		}

	case autoConnectResultMsg:
		for _, name := range msg.connected {
			// Replace "Auto-connecting to X…" with "✓ Connected to X".
			for i := len(m.items) - 1; i >= 0; i-- {
				if m.items[i].kind == itemInfo && m.items[i].content == "Auto-connecting to "+name+"…" {
					m.items[i].content = "✓ Auto-connected to " + name
					break
				}
			}
		}
		for name, err := range msg.failed {
			m.items = append(m.items, displayItem{
				kind:    itemError,
				content: "Auto-connect failed for " + name + ": " + err.Error(),
			})
		}
		needRebuild = true
		if len(msg.connected) > 0 {
			// Refresh tool list, then resume the stream via processNextCall.
			cmds = append(cmds, doRefreshMCPTools(m.ctx, m.session, true))
		} else {
			cmds = append(cmds, m.processQueueOrIdle())
		}

	case mcpToolsRefreshedMsg:
		if msg.err != nil {
			m.items = append(m.items, displayItem{kind: itemError, content: "loading MCP tools: " + msg.err.Error()})
		} else {
			m.setTools(msg.tools)
			m.allToolDefs = msg.tools
		}
		m.updateCompletion()
		needRebuild = true
		if msg.resumeCall {
			// Refresh was triggered by connect_server mid-turn — resume the queue.
			cmds = append(cmds, m.processNextCall())
		} else {
			// Drain any prompts queued while MCP servers were still connecting.
			cmds = append(cmds, m.processQueueOrIdle())
		}

	case modelInfoMsg:
		info := msg.info
		// Apply user override when the provider probe didn't return a context window,
		// or when the override is explicitly set (always takes precedence).
		if m.contextWindowOverride > 0 && (info.Context.MaxTokens == 0 || !info.HasContext()) {
			info.Context.MaxTokens = m.contextWindowOverride
		}
		if msg.err == nil {
			m.modelInfo = info
			// Update the system prompt in a fresh (non-resumed) conversation.
			if len(m.messages) == 1 && m.messages[0].Role == "system" {
				m.messages = m.newConversation()
			}
		}

	case tokenCountMsg:
		// Discard stale results — a synchronous update (processNextCall,
		// applyCompaction) has already superseded this goroutine's snapshot.
		if msg.gen == m.recountGen && msg.count.Total > 0 {
			m.contextUsage = msg.count
		}

	case todoUpdateMsg:
		if msg.autoOpen {
			m.sidebarOpen = true
			m.recalcLayout()
		}
		needRebuild = true

	case taskCompleteMsg:
		// Inject the tool result so the message history stays valid.
		m.messages = append(m.messages, ai.Message{
			Role:       "tool",
			ToolCallID: msg.callID,
			Content:    "Task complete: " + msg.summary,
		})
		m.autopilotDisable()
		m.currentTaskTitle = "" // clear task title on completion
		m.items = append(m.items, displayItem{
			kind:        itemMarkdown,
			content:     "✓ **Task complete**\n\n" + msg.summary,
			rawMarkdown: "✓ **Task complete**\n\n" + msg.summary,
		})
		needRebuild = true
		if m.sessionStore != nil {
			cmds = append(cmds, m.saveSession())
		}
		cmds = append(cmds, m.processQueueOrIdle())

	case taskTitleMsg:
		m.currentTaskTitle = msg.title
		needRebuild = true

	case sessionTitleRefinedMsg:
		if msg.err == nil && msg.title != "" {
			m.sessionTitle = msg.title
			if m.sessionStore != nil {
				cmds = append(cmds, m.saveSession())
			}
		}

	case agentActivateMsg:
		// Ignore if busy -- the use_agent tool call is completing but we
		// cannot safely mutate agent state mid-stream. The tool already
		// returned "Agent X activated." to the model; the actual activation
		// happens when the AI turn finishes via processQueueOrIdle.
		if !m.isBusy() {
			for _, a := range m.agents {
				if a.Name == msg.name {
					m.activateAgent(a)
					needRebuild = true
					break
				}
			}
		}

	case agentGeneratedMsg:
		if msg.err != nil {
			m.agentWizardGenErr = msg.err.Error()
		} else {
			m.agentWizardGenerated = msg.agent
			m.agentWizardGenErr = ""
			// Build review text as TOML-like preview.
			m.agentWizardReviewText = formatAgentPreview(msg.agent)
		}
		m.state = stateAgentWizardReview
		needRebuild = true

	case compactDoneMsg:
		if msg.err != nil {
			// Compaction failed: keep history intact, show error, go idle.
			m.items = append(m.items, displayItem{
				kind:    itemError,
				content: "Context compaction failed: " + msg.err.Error(),
			})
			m.autoCompactPending = false
			m.setIdle()
			needRebuild = true
			cmds = append(cmds, m.input.Focus())
		} else {
			cmds = append(cmds, m.applyCompaction(msg)...)
			// Show non-fatal advisory (e.g. context still very full after compaction).
			if msg.warning != "" {
				m.items = append(m.items, displayItem{
					kind:    itemInfo,
					content: "⚠ " + msg.warning,
				})
				needRebuild = true
			}
		}

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
			m.returnFromModelPicker()
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
					inputSchema: t.InputSchema,
				}
			}
			m.toolPicker.SetItems(items)
			m.toolPicker.SetSize(m.width, m.height-fixedLines)
		}

	case allToolsErrMsg:
		if m.state == statePickingTools {
			m.items = append(m.items, displayItem{kind: itemError, content: "listing tools: " + msg.err.Error()})
			m.setIdle()
			m.updateCompletion()
			needRebuild = true
		}

	case registryLoadedMsg:
		if m.state == statePickingRegistry && msg.seq == m.registrySearchSeq {
			if msg.err != nil {
				m.items = append(m.items, displayItem{
					kind:    itemError,
					content: "registry fetch failed: " + msg.err.Error(),
				})
				m.setIdle()
				m.updateCompletion()
				needRebuild = true
				break
			}
			items := make([]list.Item, len(msg.servers))
			for i, s := range msg.servers {
				items[i] = registryItem{srv: s, installed: m.registryInstalledNames[s.Name]}
			}
			m.registryNextCursor = msg.nextCursor
			m.registryPicker.SetItems(items)
			m.registryPicker.SetSize(m.width, m.height-fixedLines-2)
		}

	case registrySavedMsg:
		if msg.err == nil {
			// Keep configuredMCPServers in sync.
			alreadyKnown := false
			for _, s := range m.configuredMCPServers {
				if s.Name == msg.cfg.Name {
					alreadyKnown = true
					break
				}
			}
			if !alreadyKnown {
				m.configuredMCPServers = append(m.configuredMCPServers, msg.cfg)
				m.rebuildRegistryInstalledState()
			}
		}
		var content string
		switch {
		case msg.err != nil:
			content = `Failed to save MCP server "` + msg.cfg.Name + `": ` + msg.err.Error()
		case msg.noDCR:
			content = `Added "` + msg.cfg.Name + `" with OAuth enabled — restart werkler to connect. ` +
				`⚠ This server does not support automatic client registration. ` +
				`If authentication fails, add oauth_client_id to the server config.`
		case msg.oauthDetected:
			content = `Added "` + msg.cfg.Name + `" with OAuth auto-detected — restart werkler to connect.`
		case msg.probeErr:
			content = `Added "` + msg.cfg.Name + `" — restart werkler to connect. ` +
				`(Could not reach server to check OAuth requirements; set oauth = true in the config if needed.)`
		default:
			content = `Added "` + msg.cfg.Name + `" — restart werkler to connect.`
		}
		if m.state == statePickingRegistry {
			m.setIdle()
			m.updateCompletion()
		}
		m.items = append(m.items, displayItem{kind: itemInfo, content: content})
		needRebuild = true
		cmds = append(cmds, m.input.Focus())
		// Fire hint generation if the server has a description.
		if msg.err == nil && msg.srv.Description != "" && m.client != nil {
			if c, ok := m.client.(ai.Completer); ok {
				cmds = append(cmds, doGenerateServerHint(m.ctx, msg.srv, c))
			}
		}

	case serverHintMsg:
		if msg.hint != "" {
			m.items = append(m.items, displayItem{
				kind:    itemInfo,
				content: `Try: "` + msg.hint + `"`,
			})
			needRebuild = true
		}

	case registryRemovedMsg:
		if msg.err == nil {
			filtered := m.configuredMCPServers[:0]
			for _, s := range m.configuredMCPServers {
				if s.Name != msg.name {
					filtered = append(filtered, s)
				}
			}
			m.configuredMCPServers = filtered
			if m.state == statePickingRegistry {
				m.rebuildRegistryInstalledState()
			}
		}
		content := `Removed MCP server "` + msg.name + `" — restart werkler to apply.`
		if msg.err != nil {
			content = "Failed to remove MCP server: " + msg.err.Error()
		}
		m.items = append(m.items, displayItem{kind: itemInfo, content: content})
		needRebuild = true

	case sessionHintMsg:
		m.resumeHint = msg.sess
		needRebuild = true

	case sessionSavedMsg:
		// Update the high-water mark; discard if a newer save has already completed.
		if msg.seq > m.lastConfirmedSaveSeq {
			m.lastConfirmedSaveSeq = msg.seq
		}

	case sessionsListMsg:
		if m.state == statePickingSession {
			if msg.err != nil {
				m.items = append(m.items, displayItem{kind: itemError, content: "listing sessions: " + msg.err.Error()})
				m.setIdle()
				m.updateCompletion()
				needRebuild = true
			} else {
				items := make([]list.Item, 0, len(msg.sessions)+1)
				items = append(items, newSessionItem{})
				for _, s := range msg.sessions {
					items = append(items, sessionItem{sess: s})
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

		if msg.isPlanConfirmation {
			// Plan confirmation: the TUI owns all mode/autopilot decisions.
			// If autopilot is already running, auto-confirm without showing a dialog.
			if m.autopilot {
				targetMode, err := chat.ResolveMode(m.implementationMode, m.configuredModes)
				if err == nil {
					wasAutopilot := m.autopilot
					m.applyMode(targetMode)
					if !m.autopilot && wasAutopilot {
						m.autopilotEnable() // restore if applyMode disabled it
					}
				}
				go func() {
					msg.resultCh <- askUserResult{answer: "approved_with_autopilot: autopilot will continue"}
				}()
				break
			}
			// Override to fixed plan-confirmation choices; freeform allows a rejection reason.
			msg.choices = []string{"Implement it", "Implement it with autopilot", "Reject"}
			msg.recommended = "Implement it"
			msg.allowFreeform = true
		}

		m.askUserCallID = msg.callID
		m.askUserQuestion = msg.question
		m.askUserChoices = msg.choices
		m.askUserRecommended = msg.recommended
		m.askUserAllowFreeform = msg.allowFreeform
		m.askUserIsPlanConfirmation = msg.isPlanConfirmation

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
				last.truncatedLines += trimProcessOutputLines(&last.content, processOutputLineCap)
				found = true
			}
		}
		if !found {
			item := displayItem{
				kind:    itemProcessOutput,
				handle:  msg.handle,
				content: msg.raw,
			}
			item.truncatedLines += trimProcessOutputLines(&item.content, processOutputLineCap)
			m.items = append(m.items, item)
			// New process output starts collapsed.
			m.collapsedHandles[msg.handle] = true
		}
		needRebuild = true

	case imageLoadedMsg:
		m.pendingImages = append(m.pendingImages, msg.part)
		m.items = append(m.items, displayItem{
			kind:    itemInfo,
			content: fmt.Sprintf("📎 Attached: %s (send your next message to include it)", msg.part.Name),
		})
		needRebuild = true

	case imageLoadErrMsg:
		m.items = append(m.items, displayItem{
			kind:    itemError,
			content: "Failed to load image: " + msg.err.Error(),
		})
		needRebuild = true

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		// Forward to viewport so it can update its internal state, then
		// override the dimensions we want.
		var vpCmd tea.Cmd
		m.viewport, vpCmd = m.viewport.Update(msg)
		cmds = append(cmds, vpCmd)
		m.modelPicker.SetSize(m.width, m.height-fixedLines)
		m.toolPicker.SetSize(m.width, m.height-fixedLines)
		m.skillPicker.SetSize(m.width, m.height-fixedLines)
		m.modePicker.SetSize(m.width, m.height-fixedLines)
		m.agentWizardToolPicker.SetSize(m.width, m.height-fixedLines)
		// Only resize wizard textareas once they've been initialised by initAgentWizard.
		if isAgentWizardState(m.state) {
			m.agentWizardDescribeTA.SetWidth(m.width - 4)
			m.agentWizardManualTA.SetWidth(m.width - 4)
		}
		for i := range m.agentWizardManualTAs {
			m.agentWizardManualTAs[i].SetWidth(m.width - 6)
		}
		if m.state == stateViewingToolDetail {
			m.toolDetailVP.SetWidth(m.width)
			m.toolDetailVP.SetHeight(m.height - fixedLines)
		}
		if m.state == statePickingRegistry {
			m.registryPicker.SetSize(m.width, m.height-fixedLines-2)
			m.registryInstalledList.SetSize(m.width, m.height-fixedLines)
			m.registrySearchInput.SetWidth(m.width - 12)
		}
		m.recalcLayout() // sets viewport.Width, input.Width, viewport.Height
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
			switch {
			case errors.Is(chunk.Err, context.Canceled):
				// User-initiated cancellation: go idle cleanly.
				// Roll back the last user message since it got no response.
				if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "user" {
					m.messages = m.messages[:len(m.messages)-1]
				}
				if m.streamingItemIdx >= 0 {
					m.items[m.streamingItemIdx].content += " ✗"
					m.streamingItemIdx = -1
					m.reasoningItemIdx = -1
				}
				m.cancelOp = nil
				m.cancelPending = false
				m.setIdle()
				needRebuild = true
				cmds = append(cmds, m.input.Focus())
			case isContextOverflowError(chunk.Err):
				// The context has grown beyond the model's limit — the last user
				// message pushed us over.  Remove it so compaction can run, then
				// compact and restart the AI turn.
				debugLog("streamChunk: context overflow — compacting")
				if m.streamingItemIdx >= 0 {
					m.items[m.streamingItemIdx].content += "\n[context full — compacting]"
					m.streamingItemIdx = -1
					m.reasoningItemIdx = -1
				} else {
					m.items = append(m.items, displayItem{kind: itemInfo, content: "⚠ Context full — compacting…"})
				}
				// Drop the last user message (the one that caused the overflow).
				if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "user" {
					m.messages = m.messages[:len(m.messages)-1]
				}
				// Clear the incremental client's cached response ID so the next
				// stream call sends full context (the old ID is tied to the
				// pre‑compaction history).
				if m.incrementalClient != nil {
					m.incrementalClient.SetLastResponseID("")
				}
				m.autoCompactPending = true
				m.state = stateCompacting
				cmds = append(cmds, doCompact(m.newOpCtx(), m.client, m.messages, m.modelName, m.toolTokensCache, m.modelInfo.Context.MaxTokens))
			default:
				// Stream error: go idle but keep queued prompts intact.
				// The user can retry from idle state.
				if m.streamingItemIdx >= 0 {
					m.items[m.streamingItemIdx].content += "\n[stream error: " + chunk.Err.Error() + "]"
					m.streamingItemIdx = -1
					m.reasoningItemIdx = -1
				} else {
					m.items = append(m.items, displayItem{kind: itemError, content: chunk.Err.Error()})
				}
				m.setIdle()
				needRebuild = true
				cmds = append(cmds, m.input.Focus())
			}
		case chunk.Done:
			debugLog("streamChunk: done, toolCalls=%d, streamingItemIdx=%d, content=%q, finishReason=%q", len(chunk.Msg.ToolCalls), m.streamingItemIdx, chunk.Msg.Content, chunk.FinishReason)
			// Capture rate limit headers reported by the provider.
			if chunk.RateLimits.IsKnown() {
				m.lastRateLimits = chunk.RateLimits
			}
			// Stream finished — append the full message to history and handle tool calls.
			// Non-streaming path: no delta chunks were sent, so build items from the
			// final message directly. Reasoning is always placed before the response.
			if m.streamingItemIdx < 0 {
				if chunk.Msg.Reasoning != "" {
					m.items = append(m.items, displayItem{kind: itemReasoning, content: chunk.Msg.Reasoning})
				}
				if chunk.Msg.Content != "" {
					m.items = append(m.items, displayItem{kind: itemAssistant, content: chunk.Msg.Content, rawMarkdown: chunk.Msg.Content})
				}
			}
			m.lastTokenSpeed = 0
			elapsed := time.Since(m.streamingStart).Seconds()
			if elapsed > 0 && m.streamedTokens > 0 {
				m.lastTokenSpeed = float64(m.streamedTokens) / elapsed
			}
			// Reset for next stream.
			m.streamingStart = time.Time{}
			m.streamedTokens = 0
			m.streamingItemIdx = -1
			m.reasoningItemIdx = -1

			// Skip empty assistant messages (no content, no tool calls, no reasoning) — they
			// provide no value and some providers reject them with a 400 error.
			isEmpty := chunk.Msg.Content == "" && len(chunk.Msg.ToolCalls) == 0 && chunk.Msg.Reasoning == ""
			if !isEmpty {
				m.messages = append(m.messages, chunk.Msg)
				// Auto-connect recovery: when we're in an empty-response retry and the
				// model responds with text (no tool calls) mentioning configured server
				// names, connect those servers programmatically — bypassing the tool call
				// mechanism entirely (which devstral/Ollama can silently drop).
				if m.emptyResponseRetries > 0 &&
					len(chunk.Msg.ToolCalls) == 0 &&
					chunk.Msg.Content != "" &&
					m.mcpManager != nil {
					if serverNames := m.detectMentionedServers(chunk.Msg.Content); len(serverNames) > 0 {
						// Pop the recovery explanation from history and display — it was
						// scaffolding, not real conversation content.
						m.messages = m.messages[:len(m.messages)-1]
						if m.itemsBeforeRetry > 0 && m.itemsBeforeRetry <= len(m.items) {
							m.items = m.items[:m.itemsBeforeRetry]
						}
						m.emptyResponseRetries = 0
						for _, name := range serverNames {
							m.items = append(m.items, displayItem{
								kind:    itemInfo,
								content: "Auto-connecting to " + name + "…",
							})
						}
						debugLog("streamChunk: auto-connect triggered for %v", serverNames)
						m.setThinking()
						cmds = append(cmds, doAutoConnectServers(m.newOpCtx(), m.mcpManager, serverNames))
						needRebuild = true
						break
					}
				}
			} else if chunk.Usage.CompletionTokens > 0 {
				const maxEmptyRetries = 2
				if m.emptyResponseRetries < maxEmptyRetries {
					m.emptyResponseRetries++
					m.turnRoundtrips++
					base := m.buildStreamMessages(m.messages)
					nudgeMsgs := make([]ai.Message, len(base)+1)
					copy(nudgeMsgs, base)
					if m.emptyResponseRetries == 1 {
						// First retry: give a clean second shot. Do NOT say "plain text" —
						// the model must be free to call tools. Allowing a brief thought
						// before the tool call helps parsers (e.g. Ollama/Ministral) that
						// require some text content before a [TOOL_CALLS] token.
						nudgeMsgs[len(base)] = ai.Message{
							Role:    "user",
							Content: "Your last response was empty. Please try again — write a brief thought if needed, then call any tools required to complete the task.",
						}
						m.items = append(m.items, displayItem{
							kind:    itemInfo,
							content: "Empty response — retrying…",
						})
						debugLog("streamChunk: empty response (tokens=%d), clean retry (%d/%d)",
							chunk.Usage.CompletionTokens, m.emptyResponseRetries, maxEmptyRetries)
					} else {
						// Second retry: ask for intent. Tool calls are still allowed so the
						// model can act on its explanation immediately. We record the item
						// count here so we can roll back the display if auto-connect fires.
						nudgeMsgs[len(base)] = ai.Message{
							Role: "user",
							Content: "Your response was empty again. Explain what you were trying to do " +
								"and what you need to proceed, then attempt it.",
						}
						m.items = append(m.items, displayItem{
							kind:    itemInfo,
							content: "Empty response again — asking model to explain its intent…",
						})
						debugLog("streamChunk: empty response (tokens=%d), intent retry (%d/%d)",
							chunk.Usage.CompletionTokens, m.emptyResponseRetries, maxEmptyRetries)
					}
					// Record items watermark so auto-connect can roll back the display.
					m.itemsBeforeRetry = len(m.items)
					m.setThinking()
					cmds = append(cmds, m.doStream(m.newOpCtx(), nudgeMsgs, m.tools))
					needRebuild = true
					break
				}
				// All retries exhausted — surface the error.
				m.items = append(m.items, displayItem{
					kind:    itemError,
					content: fmt.Sprintf("Provider returned %d completion tokens but no content or tool calls — the model output may have been silently dropped by the backend. Try rephrasing your prompt or switching models.", chunk.Usage.CompletionTokens),
				})
			}
			if len(chunk.Msg.ToolCalls) == 0 {
				// If the provider cut off the response due to output token limits,
				// automatically continue rather than going idle mid-task.
				if chunk.FinishReason == "length" {
					debugLog("streamChunk: finish_reason=length, auto-continuing")
					m.messages = append(m.messages, ai.Message{Role: "user", Content: "Continue."})
					m.turnRoundtrips++
					m.setThinking()
					cmds = append(cmds, m.doStream(m.newOpCtx(), m.buildStreamMessages(m.messages), m.tools))
					needRebuild = true
					break
				}
				// Turn complete: drain queued prompts or go idle.
				if m.showTokenSpeed && m.lastTokenSpeed > 0 {
					m.items = append(m.items, displayItem{
						kind:    itemInfo,
						content: formatUsage(m.lastTokenSpeed, chunk.Usage),
					})
				}
				needRebuild = true
				// Fire async title refinement on the first text-producing AI turn
				// of a new session (not yet refined, not a resumed session).
				if !m.sessionTitleRefined && chunk.Msg.Content != "" {
					m.sessionTitleRefined = true // set synchronously before dispatch
					var firstUser, firstAI string
					for _, msg := range m.messages {
						if firstUser == "" && msg.Role == "user" && msg.Content != "" {
							firstUser = msg.Content
						}
						if firstAI == "" && msg.Role == "assistant" && msg.Content != "" {
							firstAI = msg.Content
						}
						if firstUser != "" && firstAI != "" {
							break
						}
					}
					if firstUser != "" && firstAI != "" {
						cmds = append(cmds, doRefineSessionTitle(m.newOpCtx(), m.client, firstUser, firstAI))
					}
				}
				if m.sessionStore != nil {
					cmds = append(cmds, m.saveSession())
				}
				if cmd := m.recountContext(); cmd != nil {
					cmds = append(cmds, cmd)
				}
				// Autopilot: if active and no queued user prompts, inject an
				// ephemeral continuation instead of going idle.
				if m.autopilot && len(m.queuedPrompts) == 0 {
					m.autopilotCycle++
					if m.autopilotCycle >= m.effectiveAutopilotMax() {
						m.autopilotPaused = true
						m.items = append(m.items, displayItem{
							kind: itemInfo,
							content: fmt.Sprintf(
								"⚡ Autopilot paused after %d cycles — press Enter to resume, or /autopilot to disable.",
								m.autopilotCycle,
							),
						})
						cmds = append(cmds, m.processQueueOrIdle())
					} else {
						m.turnRoundtrips = 0
						m.turnSystemMsg = ""
						m.emptyResponseRetries = 0
						m.setThinking()
						cmds = append(cmds, m.doStream(m.newOpCtx(), m.buildStreamMessages(m.autopilotMessagesForStream()), m.tools))
					}
					break
				}
				cmds = append(cmds, m.processQueueOrIdle())
			} else {
				for _, tc := range chunk.Msg.ToolCalls {
					debugLog("streamChunk: tool call id=%q name=%q", tc.ID, tc.Name)
					rawArgs, _ := json.Marshal(tc.Arguments)
					if tc.Name == "task_start" {
						m.toolCallIdx[tc.ID] = -1
					} else if liveIdx, existed := m.toolCallIdx[tc.ID]; existed && liveIdx >= 0 {
						// Live-streamed item already created; patch in the final display args.
						m.items[liveIdx].toolArgs = toolCallDisplayArgs(tc.Name, tc.Arguments)
						m.items[liveIdx].toolRawArgs = string(rawArgs)
						updateToolDisplayArgs(&m.items[liveIdx], tc.Name)
					} else {
						// No live-streamed item (server sent no argument deltas): create now.
						m.toolCallIdx[tc.ID] = len(m.items)
						rawArgBytes, _ := json.Marshal(tc.Arguments)
						m.items = append(m.items, displayItem{
							kind:        itemToolCall,
							toolName:    tc.Name,
							toolArgs:    toolCallDisplayArgs(tc.Name, tc.Arguments),
							toolRawArgs: string(rawArgBytes),
							toolNote:    toolCallIntent(tc.Name, tc.Arguments),
							toolStatus:  toolStatusPending,
							handle:      tc.ID, // enables /expand <callID>
						})
						// Start expanded so the user sees full args while the tool runs.
						// collapsedHandles[key]==false is the default (absent = expanded);
						// we explicitly set false so /collapse all does not fight us.
						m.collapsedHandles[tc.ID] = false
					}
				}
				m.pendingCalls = append(m.pendingCalls, chunk.Msg.ToolCalls...)
				nextCmd := m.processNextCall()
				needRebuild = true
				cmds = append(cmds, nextCmd)
			}
		default:
			// Live tool call argument streaming: create/update tool call bubbles
			// before any text content so the user sees the call immediately.
			if len(chunk.ToolCallChunks) > 0 {
				if m.state != stateStreaming {
					m.state = stateStreaming
					m.streamingStart = time.Now()
					m.streamedTokens = 0
				}
				for _, tcc := range chunk.ToolCallChunks {
					m.streamedTokens += countTokens(tcc.ArgumentsDelta) + countTokens(tcc.Name)
					// Create item on the first chunk (has both ID and Name).
					// Use a separate if (not else-if) so that a first chunk which
					// also carries an ArgumentsDelta is handled correctly below.
					if tcc.ID != "" && tcc.Name != "" {
						if tcc.Name == "task_start" {
							m.toolCallIdx[tcc.ID] = -1
						} else if _, exists := m.toolCallIdx[tcc.ID]; !exists {
							m.toolCallIdx[tcc.ID] = len(m.items)
							m.items = append(m.items, displayItem{
								kind:        itemToolCall,
								toolName:    tcc.Name,
								toolArgs:    "",
								toolRawArgs: "",
								toolNote:    toolCallIntent(tcc.Name, nil),
								toolStatus:  toolStatusPending,
								handle:      tcc.ID,
							})
						}
					}
					// Append argument delta.  Every ToolCallChunk carries the call
					// ID (acc.id for Chat Completions, call_id for Responses API),
					// so we always look up by ID — never by index.  This is the only
					// correct approach when multiple tool calls run in parallel.
					if tcc.ArgumentsDelta != "" && tcc.ID != "" {
						if itemIdx, ok := m.toolCallIdx[tcc.ID]; ok && itemIdx >= 0 && itemIdx < len(m.items) {
							m.items[itemIdx].toolRawArgs += tcc.ArgumentsDelta
							// Update compact args and intent title from partial JSON.
							updateToolDisplayArgs(&m.items[itemIdx], tcc.Name)
						}
					}
				}
				needRebuild = true
			}

			// Delta — reasoning or content fragment. On the first delta of any kind,
			// reserve both a reasoning slot and an assistant slot in that order so
			// reasoning is always rendered above the response regardless of arrival order.
			// --- Token speed indicator updates ---
			if m.streamingItemIdx < 0 && (chunk.Delta != "" || chunk.ReasoningDelta != "") {
				m.reasoningItemIdx = len(m.items)
				m.items = append(m.items, displayItem{kind: itemReasoning, content: ""})
				m.streamingItemIdx = len(m.items)
				m.items = append(m.items, displayItem{kind: itemAssistant, content: "", rawMarkdown: ""})
				if m.state != stateStreaming {
					m.state = stateStreaming
					m.streamingStart = time.Now()
					m.streamedTokens = 0
				}
			}
			addTokens := countTokens(chunk.Delta) + countTokens(chunk.ReasoningDelta)
			m.streamedTokens += addTokens

			if chunk.ReasoningDelta != "" {
				if m.reasoningItemIdx >= 0 {
					m.items[m.reasoningItemIdx].content += chunk.ReasoningDelta
				}
			} else if chunk.Delta != "" {
				if m.streamingItemIdx >= 0 {
					m.items[m.streamingItemIdx].content += chunk.Delta
				}
			}
			if chunk.Delta != "" || chunk.ReasoningDelta != "" {
				needRebuild = true
			}
			// Sync rawMarkdown alongside content so the cache stays valid.
			if m.streamingItemIdx >= 0 {
				m.items[m.streamingItemIdx].rawMarkdown += chunk.Delta
			}
			cmds = append(cmds, readNextChunk(msg.ch))
		}

	case compactStartMsg:
		// The summarisation stream just opened.  Add the live compact-summary item
		// so the user can watch the summary being written in real time.
		const compactHandle = "compact"
		m.compactSummaryPrevTok = m.contextUsage.Total
		m.compactSummaryItemIdx = len(m.items)
		m.items = append(m.items, displayItem{
			kind:        itemCompactSummary,
			handle:      compactHandle,
			content:     "",
			rawMarkdown: "",
		})
		m.collapsedHandles[compactHandle] = true
		m.compactSummaryAccum.Reset()
		needRebuild = true
		cmds = append(cmds, readNextCompactChunk(msg.ch, msg.snap, msg.modelName, msg.toolTokens, msg.maxTokens))

	case compactChunkMsg:
		chunk := msg.chunk
		switch {
		case chunk.Err != nil && !errors.Is(chunk.Err, io.EOF):
			// Stream error — fall through to compactDoneMsg error path.
			m.compactSummaryItemIdx = -1
			m.compactSummaryAccum.Reset()
			err := chunk.Err
			cmds = append(cmds, func() tea.Msg { return compactDoneMsg{err: err} })
		case chunk.Done || errors.Is(chunk.Err, io.EOF):
			// Stream finished.  Finalise summary and hand off to compactDoneMsg.
			// Use the delta accumulator as the primary source (it already holds
			// all streamed text).  Fall back to chunk.Msg.Content only when no
			// deltas were received (e.g. non-streaming model that sends the full
			// response in the Done chunk).
			s := strings.TrimSpace(m.compactSummaryAccum.String())
			if s == "" && chunk.Done && chunk.Msg.Content != "" {
				s = strings.TrimSpace(chunk.Msg.Content)
			}
			// Update the live item to show the final text.
			if m.compactSummaryItemIdx >= 0 && m.compactSummaryItemIdx < len(m.items) {
				if s != "" {
					m.items[m.compactSummaryItemIdx].content = s
					m.items[m.compactSummaryItemIdx].rawMarkdown = s
				}
			}
			m.compactSummaryItemIdx = -1
			m.compactSummaryAccum.Reset()
			needRebuild = true
			if s == "" {
				cmds = append(cmds, func() tea.Msg {
					return compactDoneMsg{err: fmt.Errorf("summarization returned empty response")}
				})
			} else {
				// Snapshot the compact-summary display items NOW (same tick as the
				// Done handler, while we know for certain they are in m.items and
				// have their final content).  applyCompaction runs in the next tick
				// and consumes these from m.compactSavedItems.
				m.compactSummaryFinal = s
				m.compactSavedItems = nil
				for _, it := range m.items {
					if it.kind == itemCompactSummary {
						it.content = s // ensure final content
						m.compactSavedItems = append(m.compactSavedItems, it)
					}
				}
				keepTurns, totalTokens, warning := computeCompactionParams(
					s, msg.snap, msg.modelName, msg.toolTokens, msg.maxTokens,
				)
				doneMsg := compactDoneMsg{
					summary:     s,
					keepTurns:   keepTurns,
					totalTokens: totalTokens,
					warning:     warning,
				}
				cmds = append(cmds, func() tea.Msg { return doneMsg })
			}
		default:
			// Delta chunk — append to the live item.
			if chunk.Delta != "" {
				m.compactSummaryAccum.WriteString(chunk.Delta)
				if m.compactSummaryItemIdx >= 0 && m.compactSummaryItemIdx < len(m.items) {
					m.items[m.compactSummaryItemIdx].content = m.compactSummaryAccum.String()
					m.items[m.compactSummaryItemIdx].rawMarkdown = m.compactSummaryAccum.String()
				}
				needRebuild = true
			}
			cmds = append(cmds, readNextCompactChunk(msg.ch, msg.snap, msg.modelName, msg.toolTokens, msg.maxTokens))
		}

	case toolOutputChunkMsg:
		needRebuild = true
		if msg.event.Done {
			// Tool finished — clear live output and route through toolResultMsg.
			if idx, ok := m.toolCallIdx[msg.callID]; ok && idx >= 0 {
				m.items[idx].toolLiveOutput = ""
			}
			cmds = append(cmds, func() tea.Msg {
				return toolResultMsg{
					callID:   msg.callID,
					toolName: msg.toolName,
					result:   msg.event.Result,
					diff:     msg.event.Diff,
					parts:    msg.event.Parts,
					err:      msg.event.Err,
				}
			})
		} else {
			// Output line — append to item and schedule next read.
			if idx, ok := m.toolCallIdx[msg.callID]; ok && idx >= 0 {
				out := &m.items[idx].toolLiveOutput
				if *out != "" {
					*out += "\n"
				}
				*out += msg.event.Line
				// Keep only the last 200 lines in memory; the renderer shows
				// at most 30 anyway, so older lines are never needed.
				const keepLines = 200
				if strings.Count(*out, "\n") > keepLines {
					for strings.Count(*out, "\n") > keepLines {
						if k := strings.IndexByte(*out, '\n'); k >= 0 {
							*out = (*out)[k+1:]
						} else {
							break
						}
					}
				}
			}
			cmds = append(cmds, readNextToolOutputChunk(msg.ch, msg.callID, msg.toolName))
		}

	case toolResultMsg:
		// If compaction is in progress, the tool was already given a placeholder
		// result ("not executed") before doCompact was dispatched.  Accepting the
		// real result now would append it to m.messages AFTER the snapshot that
		// doCompact already captured, so computeCompactionParams would
		// under-estimate the kept-turn size and cause an immediate re-compact.
		// Discard the late result; the AI will re-issue the call if needed.
		if m.autoCompactPending {
			break
		}
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
			case errors.Is(msg.err, context.Canceled):
				// User-initiated cancellation: add synthetic results so the
				// assistant tool_calls message has matching tool results, then
				// go idle without triggering a new stream.
				if idx, ok := m.toolCallIdx[msg.callID]; ok && idx >= 0 {
					m.items[idx].toolStatus = toolStatusDenied
					if h := m.items[idx].handle; h != "" {
						m.collapsedHandles[h] = true
					}
				}
				m.messages = append(m.messages, ai.Message{
					Role:       "tool",
					ToolCallID: msg.callID,
					Content:    "(cancelled by user)",
				})
				for _, pc := range m.pendingCalls {
					if idx, ok := m.toolCallIdx[pc.ID]; ok && idx >= 0 {
						m.items[idx].toolStatus = toolStatusDenied
						if h := m.items[idx].handle; h != "" {
							m.collapsedHandles[h] = true
						}
					}
					m.messages = append(m.messages, ai.Message{
						Role:       "tool",
						ToolCallID: pc.ID,
						Content:    "(not executed — cancelled by user)",
					})
				}
				m.pendingCalls = nil
				m.executingCall = nil
				m.currentCall = nil
				m.callingToolName = ""
				m.callingToolTitle = ""
				m.cancelOp = nil
				m.cancelPending = false
				m.setIdle()
				needRebuild = true
				cmds = append(cmds, m.input.Focus())
			default:
				// Tool execution error: add the error as a tool result so the AI
				// can see it and decide how to recover. Mark remaining calls as
				// not-executed and let the AI respond via a new stream.
				if idx, ok := m.toolCallIdx[msg.callID]; ok && idx >= 0 {
					m.items[idx].toolStatus = toolStatusFailed
					if h := m.items[idx].handle; h != "" {
						m.collapsedHandles[h] = true
					}
					if fn := toolDescriptor(msg.toolName).ParseErrorNote; fn != nil {
						m.items[idx].toolNote = fn(msg.err)
					}
				}
				m.messages = append(m.messages, ai.Message{
					Role:       "tool",
					ToolCallID: msg.callID,
					Content:    "Error: " + msg.err.Error(),
				})
				for _, pc := range m.pendingCalls {
					m.messages = append(m.messages, ai.Message{
						Role:       "tool",
						ToolCallID: pc.ID,
						Content:    "(not executed — previous call failed)",
					})
					if idx, ok := m.toolCallIdx[pc.ID]; ok && idx >= 0 {
						m.items[idx].toolStatus = toolStatusFailed
						if h := m.items[idx].handle; h != "" {
							m.collapsedHandles[h] = true
						}
					}
				}
				m.pendingCalls = nil
				m.executingCall = nil
				m.currentCall = nil
				m.callingToolName = ""
				m.callingToolTitle = ""
				needRebuild = true
				cmds = append(cmds, m.processNextCall())
			}
		} else {
			debugLog("toolResult: ok tool=%q result=%q", msg.toolName, msg.result[:min(len(msg.result), 80)])
			if idx, ok := m.toolCallIdx[msg.callID]; ok && idx >= 0 {
				m.items[idx].toolStatus = toolStatusDone
				// Collapse the args now that the call is complete.
				if h := m.items[idx].handle; h != "" {
					m.collapsedHandles[h] = true
				}
				d := toolDescriptor(msg.toolName)
				if fn := d.ParseResultNote; fn != nil {
					if d.UsesIntentTitle {
						m.items[idx].toolResultNote = fn(msg.result)
					} else {
						m.items[idx].toolNote = fn(msg.result)
					}
				}
				if d.StoreResultOutput {
					var out struct {
						Combined string `json:"combined_output"`
					}
					if json.Unmarshal([]byte(msg.result), &out) == nil && out.Combined != "" {
						m.items[idx].toolOutput = strings.TrimRight(out.Combined, "\n")
					}
				}
				if msg.diff != "" {
					m.items[idx].toolDiff = msg.diff
					m.items[idx].handle = msg.callID
					m.collapsedHandles[msg.callID] = true
				}
			}
			m.messages = append(m.messages, ai.Message{
				Role:       "tool",
				ToolCallID: msg.callID,
				Content:    msg.result,
				Parts:      msg.parts,
			})
			m.callingToolName = ""
			m.callingToolTitle = ""
			m.executingCall = nil
			// Save after every successful tool result so that session state is
			// preserved even if the app is terminated mid-turn.
			if m.sessionStore != nil {
				cmds = append(cmds, m.saveSession())
			}
			// connect_server: refresh tool list before resuming so newly available
			// tools are visible to the AI on the next stream.
			// Also clear any OAuth display item that was shown during OAuth flow.
			var nextCmd tea.Cmd
			if msg.toolName == "connect_server" {
				if m.oauthInfoIdx >= 0 {
					m.items[m.oauthInfoIdx].content = "✓ Connected"
					m.oauthInfoIdx = -1
				}
				nextCmd = doRefreshMCPTools(m.ctx, m.session, true)
			} else {
				nextCmd = m.processNextCall()
			}
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
			m.setTools(newTools)
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
		m.setIdle()
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
	// Register MCP servers for lazy connection and wire up the connect_server
	// builtin BEFORE fetching the initial tool list so it appears on turn 1.
	if opts.MCPManager != nil && len(opts.MCPServers) > 0 {
		if err := opts.MCPManager.Register(opts.MCPServers); err != nil {
			return fmt.Errorf("registering MCP servers: %w", err)
		}
		toolMgr.SetMCPManager(opts.MCPManager)
	}

	// Fetch initially-available tools (includes connect_server when servers are registered).
	sessionTools, err := session.Tools(ctx)
	if err != nil {
		return fmt.Errorf("fetching tools: %w", err)
	}

	// Resolve glamour style before starting bubbletea.
	glamourStyle := resolveGlamourStyle()

	// Enable debug logging to a file if WERKLER_DEBUG_LOG is set.
	if logPath := os.Getenv("WERKLER_DEBUG_LOG"); logPath != "" {
		f, ferr := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
		if ferr == nil {
			debugLogger = log.New(f, "", log.Ltime|log.Lmicroseconds)
			defer func() { _ = f.Close() }()
		}
	}

	var sendFn func(tea.Msg)
	sendFn = func(msg tea.Msg) {}

	m := initialModel(ctx, client, session, sessionTools, modelName, serverNames, glamourStyle, sendFn)

	// Store toolMgr so the TUI can call SetSkills on skill toggles.
	m.toolMgr = toolMgr

	// Apply skills so the system prompt hint is correct from the first message.
	m.skills = opts.Skills
	if len(opts.Skills) > 0 {
		m.messages = m.newConversation()
	}

	// Apply custom agents.
	m.agents = opts.Agents
	if len(opts.Agents) > 0 && toolMgr != nil {
		toolMgr.SetAgents(opts.Agents)
	}
	if opts.InitialAgent != nil {
		m.activateAgent(*opts.InitialAgent)
	}

	// Apply mode presets.
	m.allModes = opts.AllModes
	m.configuredModes = opts.ConfiguredModes
	m.implementationMode = opts.ImplementationMode
	m.contextWindowOverride = opts.ContextWindowOverride
	m.disableReasoning = opts.DisableReasoning
	if opts.ActiveMode.Name != "" {
		m.activeMode = opts.ActiveMode
		// Apply autopilot/approve settings from the mode (not using applyMode
		// to avoid the info item before any user message is shown).
		if opts.ActiveMode.Autopilot != nil && *opts.ActiveMode.Autopilot {
			if opts.ActiveMode.AutopilotMaxCycles > 0 {
				m.autopilotMax = opts.ActiveMode.AutopilotMaxCycles
			}
			m.autopilot = true
		}
		for _, tool := range opts.ActiveMode.AutoApproveTools {
			m.session.ApproveForSession(tool)
		}
	}

	// Keep a reference to the MCP manager for the configured-servers system prompt injection.
	if opts.MCPManager != nil {
		m.mcpManager = opts.MCPManager
	}

	// Apply session persistence options.
	m.sessionStore = opts.Store
	m.persistToolApproval = opts.PersistToolApproval
	m.persistPathApproval = opts.PersistPathApproval
	m.persistMCPServer = opts.PersistMCPServer
	m.removeRegistryServer = opts.RemoveMCPServer
	// Seed modelInfo from the override immediately so shouldAutoCompact works
	// even before the async model-probe completes.
	if m.contextWindowOverride > 0 && m.modelInfo.Context.MaxTokens == 0 {
		m.modelInfo.Context.MaxTokens = m.contextWindowOverride
	}
	if opts.MCPServers != nil {
		m.configuredMCPServers = make([]config.MCPServerConfig, len(opts.MCPServers))
		copy(m.configuredMCPServers, opts.MCPServers)
	}
	if opts.TodoStore != nil {
		m.todoStore = opts.TodoStore
		m.todoStore.SetNotify(func() {
			todos := m.todoStore.List()
			// Auto-open sidebar when the first todo is added.
			// Send in a goroutine: prog.Send blocks on an unbuffered channel and
			// this callback can be called from within Update (e.g. applySession →
			// todoStore.Restore), which would deadlock the event loop.
			autoOpen := len(todos) == 1
			go sendFn(todoUpdateMsg{autoOpen: autoOpen})
		})
	}
	if opts.Autopilot {
		m.autopilot = true
		m.autopilotMax = opts.AutopilotMaxCycles
	}
	if opts.MemoryStore != nil {
		m.memoryStore = opts.MemoryStore
		if opts.MemoryStore.Exists() {
			m.items = append(m.items, displayItem{
				kind:    itemInfo,
				content: "📝 Project memory loaded.",
			})
		}
	}
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

	// Wire up the OAuth display callback so ConnectByName can show auth URLs
	// in the TUI when the AI calls connect_server for an OAuth server.
	if opts.MCPManager != nil {
		opts.MCPManager.SetOAuthDisplay(func(serverName, authURL string) {
			sendFn(oauthNeedAuthMsg{serverName: serverName, authURL: authURL})
		})
	}

	if toolMgr != nil {
		toolMgr.SetOutputNotify(func(handle, raw, clean string) {
			sendFn(processOutputMsg{handle: handle, raw: raw, clean: clean})
		})
		toolMgr.SetUserAsker(func(ctx context.Context, question string, choices []string, recommended string, allowFreeform bool) (string, error) {
			resultCh := make(chan askUserResult, 1)
			callID := toolMgr.ActiveCallID()
			if callID != "" {
				sendFn(askUserMsg{
					callID:        callID,
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
		toolMgr.SetTaskTitleNotify(func(title string) {
			sendFn(taskTitleMsg{title: title})
		})
		toolMgr.SetPlanConfirmer(func(ctx context.Context, summary string) (string, error) {
			resultCh := make(chan askUserResult, 1)
			callID := toolMgr.ActiveCallID()
			if callID != "" {
				sendFn(askUserMsg{
					callID:             callID,
					question:           summary,
					isPlanConfirmation: true,
					resultCh:           resultCh,
				})
			}
			select {
			case r := <-resultCh:
				return r.answer, r.err
			case <-ctx.Done():
				return "", ctx.Err()
			}
		})
		toolMgr.SetAgentActivateNotify(func(name string) {
			sendFn(agentActivateMsg{name: name})
		})
	}

	prog = tea.NewProgram(&m)
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
	if lipgloss.HasDarkBackground(os.Stdin, os.Stdout) {
		return "dark"
	}
	return "light"
}

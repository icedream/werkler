package ui

import (
	"log"
	"slices"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/icedream/werkler/internal/agents"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/config"
	mcppkg "github.com/icedream/werkler/internal/mcp"
	"github.com/icedream/werkler/internal/memorystore"
	"github.com/icedream/werkler/internal/registry"
	"github.com/icedream/werkler/internal/sessionstore"
	"github.com/icedream/werkler/internal/skills"
	"github.com/icedream/werkler/internal/todostore"
)

// countTokens returns an approximate token count for a string.
func countTokens(text string) int {
	// Very rough English delta: 4 chars ≈ 1 token. Refine later with per-model tokenizer.
	return (len(text) + 3) / 4
}

const fixedLines = 7

// defaultSidebarWidth is the default column width of the todo sidebar (including 1 separator column).
const defaultSidebarWidth = 33 // 32 content + 1 for "│"

// minMainWidth is the minimum width of the main pane; below this the sidebar is hidden.
const minMainWidth = 40

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
	stateViewingToolDetail    // detail view for a single tool (schema + description)
	stateAwaitingUserQuestion // AI asked a question; waiting for the user's reply
	stateCompacting           // compacting conversation history via AI summary
	statePickingRegistry      // MCP registry browser is open
	statePickingSkills        // skill enable/disable picker is open
	statePickingMode          // mode preset picker is open
	// Agent wizard states.
	stateAgentWizardMode     // choose AI-assisted or manual
	stateAgentWizardDescribe // describe the agent (AI-assisted path)
	stateAgentWizardGenerate // spinner while AI generates the profile
	stateAgentWizardReview   // review generated profile
	stateAgentWizardManual   // manual form (4 fields)
	stateAgentWizardTools    // tool restriction picker
	stateAgentWizardDone     // confirmation after save
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
	case stateCompacting:
		return "Compacting context… (queue a follow-up, press Enter)"
	case statePickingModel, statePickingSession, statePickingTools, statePickingRegistry, statePickingSkills, statePickingMode,
		stateAgentWizardMode, stateAgentWizardDescribe, stateAgentWizardGenerate, stateAgentWizardReview,
		stateAgentWizardManual, stateAgentWizardTools, stateAgentWizardDone:
		return ""
	case stateIdle:
		return "Type a message, press Enter to send…"
	default:
		return "Queue a follow-up, press Enter…"
	}
}

// --- Display item kinds and statuses ---

const (
	itemUser           = "user"
	itemAssistant      = "assistant"
	itemReasoning      = "reasoning" // thinking/reasoning content from reasoning models
	itemMarkdown       = "markdown"  // rendered markdown (e.g. /help output)
	itemToolCall       = "tool_call"
	itemError          = "error"
	itemInfo           = "info"            // neutral status/system messages
	itemProcessOutput  = "proc_output"     // live process output streamed into viewport
	itemCompactSummary = "compact_summary" // expandable compaction summary
	itemAskUser        = "ask_user"        // ask_user tool waiting for a response
)

const (
	toolStatusPending = iota
	toolStatusRunning
	toolStatusDone
	toolStatusFailed
	toolStatusDenied
)

// autopilotDefaultMax is the default cycle cap for autopilot mode.
const autopilotDefaultMax = 50

// autopilotSystemNote is appended to the system prompt when autopilot is active.
// It is injected at request-build time, not stored in message history.
const autopilotSystemNote = `## Autopilot mode
You are operating in autopilot mode. The user is not currently present.
Work autonomously toward the given goal. Use todo_add, todo_update, and todo_list to track progress.
Call task_complete(summary) when all work is done. Call ask_user only if you are completely blocked and cannot proceed without human input.`

// processOutputLineCap is the maximum number of lines kept in a process output
// display item.  Lines beyond this are counted but not displayed.
const processOutputLineCap = 50

type displayItem struct {
	kind           string
	content        string
	toolName       string
	toolArgs       string // compact one-liner args (shown collapsed)
	toolRawArgs    string // raw JSON args for pretty-printed expanded display
	toolStatus     int
	handle         string // expand/collapse handle: callID for tool calls, process handle for output
	toolNote       string // secondary annotation: intent title or diff summary
	truncatedLines int    // lines dropped from the top of process output to stay within cap
	toolDiff       string // unified diff for file_edit/write/delete; empty for other tools
	toolLiveOutput string // live stdout/stderr while run_command is executing; cleared on done
}

// --- Tea messages ---

// streamChunkMsg carries one StreamChunk from the AI stream, plus the channel
// so Update can dispatch the next read without storing the channel in the model.
type streamChunkMsg struct {
	ch    <-chan ai.StreamChunk
	chunk ai.StreamChunk
}

// compactStartMsg is returned by the doCompact goroutine once the summarisation
// stream has been opened.  The Update handler adds the live summary item to the
// chat log and starts reading chunks.
type compactStartMsg struct {
	ch         <-chan ai.StreamChunk
	snap       []ai.Message // snapshot of messages passed to doCompact
	modelName  string
	toolTokens int
	maxTokens  int
}

// compactChunkMsg carries one streaming delta from the summarisation LLM call.
type compactChunkMsg struct {
	ch         <-chan ai.StreamChunk
	chunk      ai.StreamChunk
	snap       []ai.Message
	modelName  string
	toolTokens int
	maxTokens  int
}

type toolResultMsg struct {
	callID   string
	toolName string
	result   string
	diff     string         // unified diff for file_edit/write/delete
	parts    []ai.ImagePart // non-nil when the tool returned image data (e.g. read_image)
	err      error
}

type contextDoneMsg struct{}

// queuedPrompt is a user prompt held in the queue while the AI or MCP connect is busy.
// displayed=true means the itemUser bubble was already added to m.items (e.g. typed
// during stateConnectingMCP or stateIdle+OAuth), so processQueueOrIdle must not add it again.
type queuedPrompt struct {
	text      string
	parts     []ai.ImagePart // optional image attachments
	displayed bool
}

// imageLoadedMsg carries a successfully loaded image part ready to attach.
type imageLoadedMsg struct {
	part ai.ImagePart
}

// imageLoadErrMsg carries an error from a failed image load attempt.
type imageLoadErrMsg struct {
	err error
}

// tokenCountMsg carries the result of an async token-count operation.
type tokenCountMsg struct {
	count ai.TokenCount
	gen   int // recountGen at dispatch time; stale if < m.recountGen
}

// compactDoneMsg is returned by doCompact when the AI summary is complete or
// an error has occurred.
type compactDoneMsg struct {
	summary     string
	keepTurns   int    // number of recent turns to retain verbatim; 0 = use compactionKeepTurns default
	warning     string // non-fatal advisory shown after compaction (e.g. context still very full)
	err         error
	totalTokens int // cached total token count from pre-check (for optimization)
}

// registryLoadedMsg carries a page of servers from the MCP registry.
type registryLoadedMsg struct {
	seq        uint64
	servers    []registry.Server
	nextCursor string
	err        error
}

// registrySavedMsg reports the result of saving a registry server to config.
type registrySavedMsg struct {
	cfg           config.MCPServerConfig
	srv           registry.Server // original registry entry (for hint generation)
	err           error
	oauthDetected bool // OAuth was auto-detected and enabled on the server
	noDCR         bool // server needs OAuth but doesn't support Dynamic Client Registration
	probeErr      bool // probe couldn't reach server; OAuth status unknown
}

// serverHintMsg carries an AI-generated one-liner test suggestion for a newly
// added MCP server.
type serverHintMsg struct {
	name string
	hint string
}

// registryRemovedMsg reports the result of removing a registry server from config.
type registryRemovedMsg struct {
	name string
	err  error
}

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

type (
	modelsLoadedMsg struct{ models []ai.ModelItem }
	modelsErrMsg    struct{ err error }
)

// --- Tool picker messages ---

type (
	allToolsMsg    struct{ tools []ai.ToolDefinition }
	allToolsErrMsg struct{ err error }
)

// processOutputMsg carries new output from a running process for live display.
type processOutputMsg struct {
	handle string
	raw    string // ANSI-preserved for display
	clean  string // ANSI-stripped (used for AI, not displayed)
}

// sessionHintMsg delivers the latest CWD-matching session found on startup.
type sessionHintMsg struct{ sess *sessionstore.Session }

// sessionSavedMsg signals that a background save completed (errors are silent).
type sessionSavedMsg struct{ seq int }

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
	callID             string
	question           string
	choices            []string
	recommended        string
	allowFreeform      bool
	isPlanConfirmation bool // when true, override choices with fixed plan-confirm options
	resultCh           chan<- askUserResult
}

// mcpServerStatusMsg is sent by a background MCP connect tea.Cmd when a single
// server finishes connecting (or fails, or is deferred for OAuth).
type mcpServerStatusMsg struct {
	err error
}

// mcpToolsRefreshedMsg is sent after MCP tool state changes (startup connect or
// lazy connect_server call) and the tool list has been re-fetched.
// resumeCall, when true, means the refresh was triggered mid-turn by connect_server;
// Update should call processNextCall instead of processQueueOrIdle.
type mcpToolsRefreshedMsg struct {
	tools      []ai.ToolDefinition
	err        error
	resumeCall bool
}

// autoConnectResultMsg is sent when a programmatic auto-connect attempt
// (triggered by detecting server names in a model's intent explanation) completes.
type autoConnectResultMsg struct {
	connected []string
	failed    map[string]error
}

// modelInfoMsg is sent when a GetModelInfo probe completes.
type modelInfoMsg struct {
	info ai.ModelInfo
	err  error
}

// todoUpdateMsg is sent by the todo store notify callback when todos change.
type todoUpdateMsg struct{ autoOpen bool }

// taskCompleteMsg is sent when the AI calls task_complete.
// callID is the tool call ID so we can synthesise the tool result message.
type taskCompleteMsg struct {
	callID  string
	summary string
}

// taskTitleMsg is sent when the AI calls task_start to name its current work.
type taskTitleMsg struct{ title string }

// sessionTitleRefinedMsg carries the async AI-generated session title back to Update.
type sessionTitleRefinedMsg struct {
	title string
	err   error
}

// agentActivateMsg is sent when the AI calls use_agent.
type agentActivateMsg struct{ name string }

// agentGeneratedMsg carries the result of an AI-assisted agent profile generation.
type agentGeneratedMsg struct {
	agent agents.Agent
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

// newSessionItem is the sentinel "↳ New session" entry at the top of the picker.
type newSessionItem struct{}

func (newSessionItem) FilterValue() string { return "new session" }
func (newSessionItem) Title() string       { return "↳ New session" }
func (newSessionItem) Description() string { return "Start a fresh conversation" }

// toolItem implements list.Item for the tool picker.
// The name field holds the full AI-facing tool name (e.g. "github__get_file_contents").
type toolItem struct {
	name        string
	description string
	enabled     bool
	inputSchema any
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
	return check + " " + toolFriendlyName(t.name)
}

func (t toolItem) Description() string {
	server := toolServerName(t.name)
	desc := t.description
	const maxDesc = 80
	if len(desc) > maxDesc {
		desc = desc[:maxDesc] + "…"
	}
	params := toolParamSummary(t.inputSchema)
	suffix := ""
	if params != "" {
		suffix = "  · " + params
	}
	if desc == "" {
		return "[" + server + "]" + suffix
	}
	return "[" + server + "] " + desc + suffix
}

// toolParamSummary returns a compact comma-separated list of parameter names
// extracted from a JSON Schema object (the InputSchema of a tool definition).
func toolParamSummary(schema any) string {
	m, ok := schema.(map[string]any)
	if !ok {
		return ""
	}
	props, ok := m["properties"].(map[string]any)
	if !ok || len(props) == 0 {
		return ""
	}
	required := map[string]bool{}
	if req, ok := m["required"].([]any); ok {
		for _, r := range req {
			if s, ok := r.(string); ok {
				required[s] = true
			}
		}
	}
	names := make([]string, 0, len(props))
	for k := range props {
		if required[k] {
			names = append(names, k)
		}
	}
	// Append optional params after required ones.
	for k := range props {
		if !required[k] {
			names = append(names, k+"?")
		}
	}
	slices.Sort(names)
	return strings.Join(names, ", ")
}

// skillItem implements list.Item for the skill enable/disable picker.
type skillItem struct {
	name        string
	description string
	enabled     bool
}

func (s skillItem) FilterValue() string { return s.name }
func (s skillItem) Title() string {
	check := "[ ]"
	if s.enabled {
		check = "[✓]"
	}
	return check + " " + s.name
}

func (s skillItem) Description() string {
	const maxDesc = 100
	desc := s.description
	if len(desc) > maxDesc {
		desc = desc[:maxDesc] + "…"
	}
	return desc
}

// modeItem implements list.Item for the mode picker.
type modeItem struct {
	mode   chat.ResolvedMode
	active bool
}

func (m modeItem) FilterValue() string { return m.mode.Name }
func (m modeItem) Title() string {
	indicator := "  "
	if m.active {
		indicator = "● "
	}
	return indicator + m.mode.Name
}

func (m modeItem) Description() string {
	if m.mode.IsDefault {
		return "Standard mode -- no modifications to system prompt or session settings"
	}
	if m.mode.SystemPromptExtra != "" {
		// Show first line of the extra as description.
		lines := strings.SplitN(m.mode.SystemPromptExtra, "\n", 3)
		for _, line := range lines {
			line = strings.TrimSpace(strings.TrimPrefix(line, "##"))
			line = strings.TrimSpace(line)
			if line != "" {
				return line
			}
		}
	}
	return ""
}

// agentToolItem implements list.Item for the wizard tool restriction picker.
type agentToolItem struct {
	name     string
	enabled  bool
	alwaysOn bool // infra tool: always enabled, cannot be toggled
}

func (a agentToolItem) FilterValue() string { return a.name }
func (a agentToolItem) Title() string {
	check := "[ ]"
	if a.enabled {
		check = "[✓]"
	}
	if a.alwaysOn {
		return check + " " + a.name + "  (always on)"
	}
	return check + " " + a.name
}

func (a agentToolItem) Description() string {
	if a.alwaysOn {
		return "Infrastructure tool -- always available regardless of agent tool restrictions"
	}
	return ""
}

// registryItem implements list.Item for the MCP registry picker.
type registryItem struct {
	srv       registry.Server
	installed bool
}

func (r registryItem) FilterValue() string { return r.srv.Title + " " + r.srv.Name }
func (r registryItem) Title() string {
	t := r.srv.Title
	if r.srv.HasPackage {
		t += "  (requires install)"
	}
	if r.installed {
		t += "  ✓"
	}
	return t
}

func (r registryItem) Description() string {
	desc := r.srv.Description
	const maxDesc = 100
	if len(desc) > maxDesc {
		desc = desc[:maxDesc] + "…"
	}
	url := r.srv.FirstRemoteURL()
	if url != "" {
		return desc + "  — " + url
	}
	return desc
}

// registryInstalledItem implements list.Item for the installed-servers tab.
type registryInstalledItem struct{ srv config.MCPServerConfig }

func (r registryInstalledItem) Title() string {
	return r.srv.Name
}

func (r registryInstalledItem) Description() string {
	switch r.srv.Transport {
	case config.MCPTransportStreamable, config.MCPTransportSSE:
		u := r.srv.URL
		if u == "" {
			u = "(no URL)"
		}
		return string(r.srv.Transport) + " — " + u
	case config.MCPTransportStdio:
		return "stdio — " + r.srv.Command
	}
	return string(r.srv.Transport)
}
func (r registryInstalledItem) FilterValue() string { return r.srv.Name }

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
	// PersistMCPServer, if non-nil, is called when the user adds an MCP server
	// from the registry browser (saves it to the config file).
	PersistMCPServer func(cfg config.MCPServerConfig) error
	// RemoveMCPServer, if non-nil, is called when the user removes an MCP server
	// in the registry browser (removes it from the config file).
	RemoveMCPServer func(serverName string) error
	// Skills is the list of loaded skills to mention in the system prompt.
	Skills []skills.Skill
	// TodoStore, if non-nil, enables the AI-managed todo sidebar.
	TodoStore *todostore.Store
	// MCPManager and MCPServers, when both non-nil/non-empty, enable lazy MCP
	// server connection. Servers are registered at startup and the AI can connect
	// them on demand via the connect_server tool.
	MCPManager *mcppkg.Manager
	MCPServers []config.MCPServerConfig
	// Autopilot, when true, enables autonomous loop mode from the first prompt.
	Autopilot bool
	// AutopilotMaxCycles overrides the cycle cap (0 = use config/default).
	AutopilotMaxCycles int
	// ActiveMode is the resolved mode to apply at session start.
	// The zero value (IsDefault=false, Name="") is treated as the default mode.
	ActiveMode chat.ResolvedMode
	// AllModes is the full list of modes available for the /mode picker.
	AllModes []chat.ResolvedMode
	// ConfiguredModes is the raw config-level list used to re-resolve modes
	// when restoring a session.
	ConfiguredModes []config.ModeConfig
	// ImplementationMode is the name of the mode preset to apply when the AI
	// calls confirm_plan and the user approves. Empty string uses the default mode.
	ImplementationMode string
	// ContextWindowOverride, if > 0, is used as the model's context window size
	// when the provider does not report it (e.g. GitHub Copilot).
	ContextWindowOverride int
	// ReasoningEffort is the effort level to use when reasoning is active
	// ("low", "medium", "high"). Empty means use the model's default.
	ReasoningEffort string
	// DisableReasoning suppresses reasoning tools and never sends reasoning_effort.
	DisableReasoning bool
	// MemoryStore, if non-nil, enables cross-session project memory tools and
	// injects the current memory into the system prompt at request time.
	MemoryStore *memorystore.MemoryStore
	// Agents is the list of loaded custom agents.
	Agents []agents.Agent
	// InitialAgent, if non-nil, is activated at session start.
	InitialAgent *agents.Agent
}

// --- Slash commands ---

// slashCommand describes a /command available in the input box.
type slashCommand struct {
	name        string // without leading slash, e.g. "model"
	description string
	available   func(*Model) bool      // nil = always available
	action      func(*Model) []tea.Cmd // executed on selection
	// safeWhileBusy, when true, allows this command to be executed while the AI
	// is actively thinking/streaming/calling tools. Commands that open picker
	// overlays must NOT set this, since overlay + async state mutation is unsafe.
	safeWhileBusy bool
}

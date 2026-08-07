package chat

import (
	"github.com/icedream/werkler/internal/ai"
)

// autopilotEphemeralMsg is the content of the injected continuation message;
// it is shared verbatim between prompt mode and the TUI so behaviour is
// consistent across both.
const autopilotEphemeralMsg = "Continue working."

// autopilotExceededMsg is the error message returned when the autopilot
// cycle cap is exceeded without calling task_complete.
const autopilotExceededMsg = "autopilot reached cycle cap of %d without calling task_complete"

// AlwaysApprovedTools is the set of tool names that are always allowed without
// requiring user approval in non-interactive modes. Control-flow, state, and
// utility tools that carry no user-visible side-effects live here.
var AlwaysApprovedTools = map[string]bool{
	"ask_user":       true,
	"confirm_plan":   true,
	"todo_add":       true,
	"todo_add_many":  true,
	"todo_update":    true,
	"todo_list":      true,
	"memory_list":    true,
	"memory_read":    true,
	"memory_write":   true,
	"memory_delete":  true,
	"memory_promote": true,
	"calculate":      true,
	"sleep":          true,
}

// toolDescriptorIsAutoApprove returns true if the given tool name is on the
// AlwaysApproved list. The TUI additionally consults its own
// toolDescriptor(name).AutoApprove flag; callers that need it should extend
// this check at the call site rather than pulling the UI package into chat.
func toolDescriptorIsAutoApprove(name string) bool {
	return AlwaysApprovedTools[name]
}

// taskCompleteSummary extracts the "summary" argument from a task_complete tool
// call, falling back to "" if missing or malformed.
func taskCompleteSummary(tc ai.ToolCall) string {
	s, _ := tc.Arguments["summary"].(string)
	return s
}

// denyToolMessage builds the tool-result message appended to history when a
// tool call is denied. `reason` is free-form text shown to the AI (e.g. "(tool
// call was not approved — run interactively to approve tools, or add to
// auto_approve_tools in config)").
func denyToolMessage(tc ai.ToolCall, reason string) ai.Message {
	return ai.Message{
		Role:       "tool",
		ToolCallID: tc.ID,
		Content:    reason,
	}
}

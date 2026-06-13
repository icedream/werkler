package ui

import (
	"fmt"
	"strconv"
	"strings"
)

// ToolDescriptor bundles every piece of per-tool UI and execution behaviour
// that was previously scattered across switch statements in multiple files.
// Register one entry per built-in tool in builtinTools; unknown tools (e.g.
// MCP tools) fall back to a minimal default derived from the tool name.
type ToolDescriptor struct {
	// FriendlyName is the human-readable label shown in the collapsed badge,
	// e.g. "Edit file", "Run command".
	FriendlyName string

	// UsesIntentTitle: when true, the AI-provided "title" argument is shown as
	// the primary badge name while the call is pending/running (run_command,
	// process_start).  ToolResultNote is then shown as the secondary annotation.
	UsesIntentTitle bool

	// AutoApprove: when true, the tool is dispatched without an approval dialog,
	// equivalent to the long OR-chain that was in processNextCall.
	AutoApprove bool

	// StreamOutput: when true, doCallToolStream is used so live stdout/stderr
	// lines are forwarded to the TUI while the command runs.
	StreamOutput bool

	// StoreResultOutput: when true, combined_output from the result JSON is
	// stored in item.toolOutput so the user can /expand to see full output.
	StoreResultOutput bool

	// BadgeArgs formats a compact one-liner from tool arguments for the badge.
	// Returns "" to fall through to formatArgsCompact.
	BadgeArgs func(args map[string]any) string

	// Intent extracts the AI-provided intent/title string from tool arguments.
	// Returns "" for tools that don't carry a human-readable title.
	Intent func(args map[string]any) string

	// StreamingRenderer renders a partially-received JSON args string while the
	// tool call is still streaming in.  nil → renderStreamingGeneric fallback.
	StreamingRenderer func(rawArgs string, width int) string

	// ParseResultNote derives a short success annotation from the result JSON,
	// e.g. "+3 -2 lines @ line 42" or "exit: 0".  nil → no annotation.
	ParseResultNote func(result string) string

	// ParseErrorNote derives a short error annotation from a tool error,
	// e.g. "old_str not found".  nil → no annotation.
	ParseErrorNote func(err error) string
}

// titleArg is a shared Intent extractor that reads the "title" argument field,
// used by run_command, process_start, and similar tools.
func titleArg(args map[string]any) string {
	t, _ := args["title"].(string)
	return t
}

// filePathBadgeArgs is a shared BadgeArgs formatter that shows the "path" arg.
func filePathBadgeArgs(args map[string]any) string {
	if path, ok := args["path"].(string); ok && path != "" {
		return shortenHomePath(path)
	}
	return ""
}

// runCommandBadgeArgs formats run_command arguments as "$ command [args…]".
func runCommandBadgeArgs(args map[string]any) string {
	cmd, _ := args["command"].(string)
	if cmd == "" {
		return ""
	}
	useShell, _ := args["shell"].(bool)
	if useShell {
		return "$ " + cmd
	}
	parts := []string{cmd}
	if rawArgs, ok := args["args"].([]any); ok {
		for _, a := range rawArgs {
			if s, ok := a.(string); ok {
				parts = append(parts, s)
			}
		}
	}
	return "$ " + strings.Join(parts, " ")
}

// processStartBadgeArgs formats process_start arguments.
func processStartBadgeArgs(args map[string]any) string {
	cmd, _ := args["command"].(string)
	if cmd == "" {
		return ""
	}
	parts := []string{cmd}
	if rawArgs, ok := args["args"].([]any); ok {
		for _, a := range rawArgs {
			if s, ok := a.(string); ok {
				parts = append(parts, strconv.Quote(s))
			}
		}
	}
	return "$ " + strings.Join(parts, " ")
}

// connectServerBadgeArgs formats connect_server arguments.
func connectServerBadgeArgs(args map[string]any) string {
	if name, ok := args["name"].(string); ok && name != "" {
		return "→ " + name
	}
	return ""
}

// parseRunCommandResultNote extracts exit code and last error line.
func parseRunCommandResultNote(result string) string {
	return parseRunCommandNote(result)
}

// parseFileReadMultiNote extracts file paths with line ranges from a
// file_read_multi result.  The result text has headers like:
//
//	=== path/to/file [L10-L42 of 100] ===
//
// It returns a compact summary like "2 files — path.md, config.toml"
// or "path.md:1-613" for a single file.
// Errors (e.g. [ERROR] markers) are included in the summary.
func parseFileReadMultiNote(result string) string {
	lines := strings.Split(result, "\n")
	entries := make([]string, 0, len(lines))
	for _, line := range lines {
		if !strings.HasPrefix(line, "=== ") {
			continue
		}
		inner := strings.TrimSuffix(strings.TrimPrefix(line, "=== "), " ===")
		// Parse "path [Lstart-end of total]" → "path:start-end"
		// or "path [ERROR]" → "path [ERROR]"
		if idx := strings.Index(inner, " [L"); idx >= 0 {
			closing := strings.Index(inner[idx+1:], "]")
			if closing > 0 {
				path := strings.TrimSpace(inner[:idx])
				rangeText := inner[idx+2 : idx+1+closing-1] // e.g. "L10-L42 of 100"
				// Extract start-end from "Lstart-end of N"
				if ofIdx := strings.Index(rangeText, " of "); ofIdx > 0 {
					ranges := rangeText[:ofIdx] // "L10-L42"
					// Strip leading "L" and the second "L" before the end number
					startStr := strings.ReplaceAll(ranges, "L", "")
					entries = append(entries, path+":"+startStr)
				} else {
					entries = append(entries, inner)
				}
			} else {
				entries = append(entries, inner)
			}
		} else {
			// Keep as-is (e.g. error entries)
			entries = append(entries, inner)
		}
	}
	if len(entries) == 0 {
		return ""
	}
	if len(entries) == 1 {
		return entries[0]
	}
	const maxShow = 2
	total := len(entries)
	show := entries[:min(maxShow, total)]
	if total > maxShow {
		return fmt.Sprintf("%d files — %s, and %d more", total,
			strings.Join(show, ", "), total-maxShow)
	}
	return fmt.Sprintf("%d files — %s", total, strings.Join(show, ", "))
}

// builtinTools maps tool name → ToolDescriptor.  All switch statements that
// previously scattered tool-specific logic across multiple files are replaced
// by a single toolDescriptor(name) lookup.
var builtinTools = map[string]ToolDescriptor{
	// --- File tools ---
	"file_read_multi": {
		FriendlyName:    "Read files",
		BadgeArgs:       filePathBadgeArgs,
		ParseResultNote: parseFileReadMultiNote,
	},
	"file_write": {
		FriendlyName:      "Write file",
		BadgeArgs:         filePathBadgeArgs,
		StreamingRenderer: renderStreamingFileWrite,
		ParseResultNote:   parseFileWriteNote,
		ParseErrorNote:    fileWriteErrorNote,
	},
	"file_edit": {
		FriendlyName:      "Edit file",
		BadgeArgs:         filePathBadgeArgs,
		StreamingRenderer: renderStreamingFileEdit,
		ParseResultNote:   parseFileEditNote,
		ParseErrorNote:    fileEditErrorNote,
	},
	"file_list": {
		FriendlyName: "List files",
	},
	"file_glob": {
		FriendlyName: "Find files",
	},
	"file_delete": {
		FriendlyName:      "Delete file",
		BadgeArgs:         filePathBadgeArgs,
		StreamingRenderer: renderStreamingFilePath,
		ParseResultNote:   parseFileDeleteNote,
	},
	"read_image": {
		FriendlyName: "Read image",
		BadgeArgs:    filePathBadgeArgs,
	},

	// --- Process / command tools ---
	"run_command": {
		FriendlyName:      "Run command",
		UsesIntentTitle:   true,
		StreamOutput:      true,
		StoreResultOutput: true,
		BadgeArgs:         runCommandBadgeArgs,
		Intent:            titleArg,
		StreamingRenderer: renderStreamingCommand,
		ParseResultNote:   parseRunCommandResultNote,
	},
	"process_start": {
		FriendlyName:      "Run command",
		UsesIntentTitle:   true,
		BadgeArgs:         processStartBadgeArgs,
		Intent:            titleArg,
		StreamingRenderer: renderStreamingCommand,
	},
	"process_send": {
		FriendlyName: "Send input",
	},
	"process_send_key": {
		FriendlyName: "Send key",
	},
	"process_read": {
		FriendlyName: "Read output",
	},
	"process_stop": {
		FriendlyName: "Stop process",
	},

	// --- Infrastructure tools (always auto-approved) ---
	"connect_server": {
		FriendlyName: "Connect server",
		AutoApprove:  true,
		BadgeArgs:    connectServerBadgeArgs,
	},

	// --- User interaction (always auto-approved) ---
	"ask_user": {
		FriendlyName: "Ask user",
		AutoApprove:  true,
	},
	"confirm_plan": {
		FriendlyName: "Confirm plan",
		AutoApprove:  true,
	},

	// --- Task management (always auto-approved) ---
	"task_start": {
		FriendlyName: "Start task",
		AutoApprove:  true,
	},
	"task_complete": {
		FriendlyName: "Complete task",
	},

	// --- Todo tools (always auto-approved) ---
	"todo_add": {
		FriendlyName: "Add todo",
		AutoApprove:  true,
	},
	"todo_add_many": {
		FriendlyName: "Add todos",
		AutoApprove:  true,
	},
	"todo_update": {
		FriendlyName: "Update todo",
		AutoApprove:  true,
	},
	"todo_list": {
		FriendlyName: "List todos",
		AutoApprove:  true,
	},

	// --- Memory tools (always auto-approved) ---
	"memory_list": {
		FriendlyName: "List memories",
		AutoApprove:  true,
	},
	"memory_read": {
		FriendlyName: "Read memory",
		AutoApprove:  true,
	},
	"memory_write": {
		FriendlyName: "Write memory",
		AutoApprove:  true,
	},
	"memory_delete": {
		FriendlyName: "Delete memory",
		AutoApprove:  true,
	},
	"memory_promote": {
		FriendlyName: "Promote memory",
		AutoApprove:  true,
	},

	// --- Utility tools (always auto-approved) ---
	"calculate": {
		FriendlyName: "Calculate",
		AutoApprove:  true,
	},
	"sleep": {
		FriendlyName: "Sleep",
		AutoApprove:  true,
	},

	// --- Skill / agent tools (always auto-approved) ---
	"use_skill": {
		FriendlyName: "Use skill",
		AutoApprove:  true,
	},
	"use_agent": {
		FriendlyName: "Use agent",
		AutoApprove:  true,
	},
}

// toolDescriptor returns the ToolDescriptor for a tool name.
// Unknown tools (e.g. MCP tools) get a minimal default with a friendly name
// derived from the tool name via snakeCaseToTitle.
func toolDescriptor(name string) ToolDescriptor {
	if d, ok := builtinTools[name]; ok {
		return d
	}
	return ToolDescriptor{FriendlyName: fmt.Sprintf("%s [%s]", snakeCaseToTitle(toolBaseName(name)), name)}
}

// toolBadgeArgs returns the compact badge args string for a tool call.
func toolBadgeArgs(name string, args map[string]any) string {
	d := toolDescriptor(name)
	if d.BadgeArgs != nil {
		if s := d.BadgeArgs(args); s != "" {
			return s
		}
	}
	return formatArgsCompact(args)
}

// toolIntent returns the AI-provided intent/title for a tool call, or "".
func toolIntent(name string, args map[string]any) string {
	d := toolDescriptor(name)
	if d.Intent != nil {
		return d.Intent(args)
	}
	return ""
}

// toolStreamingRenderer returns the streaming arg renderer for a tool, falling
// back to renderStreamingGeneric for tools without a custom renderer.
func toolStreamingRenderer(name string) func(string, int) string {
	if d, ok := builtinTools[name]; ok && d.StreamingRenderer != nil {
		return d.StreamingRenderer
	}
	return renderStreamingGeneric
}

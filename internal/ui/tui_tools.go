package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/tools"
)

// doLoadImage loads an image from a local path or URL and returns it as an
// imageLoadedMsg (or imageLoadErrMsg on failure).
func doLoadImage(src string) tea.Cmd {
	return func() tea.Msg {
		if strings.HasPrefix(src, "http://") || strings.HasPrefix(src, "https://") {
			return imageLoadedMsg{part: ai.ImagePart{URL: src, Name: filepath.Base(src)}}
		}
		data, err := os.ReadFile(src)
		if err != nil {
			return imageLoadErrMsg{err: fmt.Errorf("read image: %w", err)}
		}
		mime := http.DetectContentType(data)
		// DetectContentType may return "application/octet-stream" for image types
		// it can't detect; fall back to extension mapping.
		if mime == "application/octet-stream" {
			ext := strings.ToLower(filepath.Ext(src))
			switch ext {
			case ".png":
				mime = "image/png"
			case ".jpg", ".jpeg":
				mime = "image/jpeg"
			case ".gif":
				mime = "image/gif"
			case ".webp":
				mime = "image/webp"
			default:
				return imageLoadErrMsg{err: fmt.Errorf("unsupported image type: %s", ext)}
			}
		}
		if !strings.HasPrefix(mime, "image/") {
			return imageLoadErrMsg{err: fmt.Errorf("not an image file (detected %s)", mime)}
		}
		return imageLoadedMsg{part: ai.ImagePart{
			Data:     data,
			MIMEType: mime,
			Name:     filepath.Base(src),
		}}
	}
}

func doCallTool(ctx context.Context, toolMgr *tools.Manager, session *chat.Session, tc ai.ToolCall) tea.Cmd {
	return func() tea.Msg {
		if toolMgr != nil {
			toolMgr.SetActiveCallID(tc.ID)
			defer toolMgr.SetActiveCallID("")
		}
		result, parts, err := session.CallToolWithParts(ctx, tc)
		// Extract unified diff from tool result JSON if present.
		var diff string
		if err == nil {
			var resultJSON map[string]any
			if jerr := json.Unmarshal([]byte(result), &resultJSON); jerr == nil {
				diff, _ = resultJSON["diff"].(string)
			}
		}
		return toolResultMsg{tc.ID, tc.Name, result, diff, parts, err}
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

// doRefreshMCPTools re-fetches the full tool list from the session (after MCP
// servers have finished connecting) and returns a mcpToolsRefreshedMsg.
// resumeCall should be true when called mid-turn (e.g. after connect_server).
func doRefreshMCPTools(ctx context.Context, session *chat.Session, resumeCall bool) tea.Cmd {
	return func() tea.Msg {
		tools, err := session.Tools(ctx)
		return mcpToolsRefreshedMsg{tools: tools, err: err, resumeCall: resumeCall}
	}
}

func doGetModelInfo(ctx context.Context, getter ai.ModelInfoGetter) tea.Cmd {
	return func() tea.Msg {
		info, err := getter.GetModelInfo(ctx)
		return modelInfoMsg{info: info, err: err}
	}
}

// toolResultCap returns the maximum number of characters to include from a
// tool result in the compaction transcript. Applies a recency bias: the most
// recent third of messages gets more space than older entries.
func toolResultCap(msgIndex, total int) int {
	switch {
	case total == 0 || msgIndex >= total*2/3:
		return 1500 // most recent third — most relevant
	case msgIndex >= total/3:
		return 600 // middle third
	default:
		return 200 // oldest third — summarise aggressively
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

// toolFriendlyName returns a human-readable label for a tool name.
// Built-in tools have explicit labels; MCP tools are auto-formatted from
// their base name (the part after "__") by converting snake_case to Title Case.
func toolFriendlyName(name string) string {
	switch name {
	case "file_read_multi":
		return "Read files"
	case "file_write":
		return "Write file"
	case "file_edit":
		return "Edit file"
	case "file_list":
		return "List files"
	case "file_glob":
		return "Find files"
	case "file_delete":
		return "Delete file"
	case "process_start":
		return "Run command"
	case "process_send":
		return "Send input"
	case "process_send_key":
		return "Send key"
	case "process_read":
		return "Read output"
	case "process_stop":
		return "Stop process"
	case "connect_server":
		return "Connect server"
	case "task_complete":
		return "Complete task"
	case "ask_user":
		return "Ask user"
	case "todo_add":
		return "Add todo"
	case "todo_update":
		return "Update todo"
	case "todo_list":
		return "List todos"
	case "memory_list":
		return "List memories"
	case "memory_read":
		return "Read memory"
	case "memory_write":
		return "Write memory"
	case "memory_delete":
		return "Delete memory"
	case "calculate":
		return "Calculate"
	case "sleep":
		return "Sleep"
	}
	// MCP tool: auto-format the base name.
	return snakeCaseToTitle(toolBaseName(name))
}

// snakeCaseToTitle converts a snake_case string to Title Case words
// (e.g. "get_file_contents" → "Get File Contents").
func snakeCaseToTitle(s string) string {
	words := strings.Split(s, "_")
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}
	return strings.Join(words, " ")
}

// toolCallDisplayArgs returns a compact display string for tool call arguments.
func toolCallDisplayArgs(toolName string, args map[string]any) string {
	switch toolName {
	case "file_edit", "file_write", "file_delete":
		if path, ok := args["path"].(string); ok && path != "" {
			return shortenHomePath(path)
		}
	case "process_start":
		cmd, _ := args["command"].(string)
		if cmd != "" {
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
	case "run_command":
		cmd, _ := args["command"].(string)
		if cmd != "" {
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
	case "connect_server":
		if name, ok := args["name"].(string); ok && name != "" {
			return "→ " + name
		}
	}
	return formatArgsCompact(args)
}

// toolCallIntent returns a short intent annotation for tool calls that carry a
// human-readable title field. Returns "" for all others.
func toolCallIntent(toolName string, args map[string]any) string {
	switch toolName {
	case "process_start", "run_command":
		if title, ok := args["title"].(string); ok {
			return title
		}
	}
	return ""
}

// parseFileEditNote extracts a human-readable annotation from a successful
// file_edit result JSON: e.g. "+3 -2 lines @ line 42".
func parseFileEditNote(result string) string {
	var data map[string]any
	if err := json.Unmarshal([]byte(result), &data); err != nil {
		return ""
	}
	added, _ := data["added"].(float64)
	removed, _ := data["removed"].(float64)
	line, _ := data["line"].(float64)
	if added == 0 && removed == 0 {
		return ""
	}
	add := diffAddedStyle.Render(fmt.Sprintf("+%d", int(added)))
	rem := diffRemovedStyle.Render(fmt.Sprintf("-%d", int(removed)))
	return fmt.Sprintf("%s %s lines @ line %d", add, rem, int(line))
}

// parseFileWriteNote extracts a human-readable annotation from a successful
// file_write result JSON: e.g. "wrote 1.2 KB".
func parseFileWriteNote(result string) string {
	var data map[string]any
	if err := json.Unmarshal([]byte(result), &data); err != nil {
		return ""
	}
	bytes, _ := data["bytes"].(float64)
	return fmt.Sprintf("wrote %s", formatBytes(int64(bytes)))
}

// parseFileDeleteNote extracts a human-readable annotation from a successful
// file_delete result JSON.
func parseFileDeleteNote(result string) string {
	var data map[string]any
	if err := json.Unmarshal([]byte(result), &data); err != nil {
		return ""
	}
	if _, ok := data["deleted"]; ok {
		return "deleted"
	}
	return ""
}

// fileEditErrorNote returns a short, user-readable summary of a file_edit error.
func fileEditErrorNote(err error) string {
	if err == nil {
		return ""
	}
	msg := err.Error()
	// Strip "file_edit: " prefix.
	msg = strings.TrimPrefix(msg, "file_edit: ")
	// Truncate verbose recovery hints after " — ".
	if i := strings.Index(msg, " — "); i > 0 {
		msg = msg[:i]
	}
	if len(msg) > 100 {
		msg = msg[:100] + "…"
	}
	return msg
}

// fileWriteErrorNote returns a short, user-readable summary of a file_write error.
func fileWriteErrorNote(err error) string {
	if err == nil {
		return ""
	}
	msg := err.Error()
	// Strip "file_write: " prefix.
	msg = strings.TrimPrefix(msg, "file_write: ")
	// Truncate verbose recovery hints after " — ".
	if i := strings.Index(msg, " — "); i > 0 {
		msg = msg[:i]
	}
	if len(msg) > 100 {
		msg = msg[:100] + "…"
	}
	return msg
}

// renderDiff renders a unified diff string with coloured +/-/@@ lines and
// intra-line character-level highlighting for changed lines.
func renderDiff(raw string, width int) string {
	_ = width // reserved for future line-wrapping
	lines := strings.Split(raw, "\n")

	// Collect removed/added lines in adjacent pairs for intra-line diff.
	// Pass 1: group consecutive - then + lines into pairs.
	type renderedLine struct{ s string }
	result := make([]renderedLine, 0, len(lines))

	i := 0
	for i < len(lines) {
		line := lines[i]
		switch {
		case strings.HasPrefix(line, "@@"):
			result = append(result, renderedLine{diffHunkStyle.Render(line)})
			i++
		case strings.HasPrefix(line, "---"), strings.HasPrefix(line, "+++"):
			// file header lines — dim, no intraline
			result = append(result, renderedLine{toolDimStyle.Render(line)})
			i++
		case strings.HasPrefix(line, "-"):
			// Collect a run of removed lines, then look ahead for matching added lines.
			var removed, added []string
			for i < len(lines) && strings.HasPrefix(lines[i], "-") {
				removed = append(removed, strings.TrimPrefix(lines[i], "-"))
				i++
			}
			for i < len(lines) && strings.HasPrefix(lines[i], "+") {
				added = append(added, strings.TrimPrefix(lines[i], "+"))
				i++
			}
			// Render pairs with intra-line highlighting; lone lines get plain colour.
			pairs := len(removed)
			if len(added) < pairs {
				pairs = len(added)
			}
			for j, rem := range removed {
				if j < pairs {
					segs := tools.IntralineDiff(rem, added[j])
					var sb strings.Builder
					sb.WriteString(diffRemovedStyle.Render("-"))
					for _, seg := range segs {
						if seg.Removed {
							sb.WriteString(diffRemovedHighStyle.Render(seg.Text))
						} else if !seg.Added {
							sb.WriteString(diffRemovedStyle.Render(seg.Text))
						}
					}
					result = append(result, renderedLine{sb.String()})
				} else {
					result = append(result, renderedLine{diffRemovedStyle.Render("-" + rem)})
				}
			}
			for j, add := range added {
				if j < pairs {
					segs := tools.IntralineDiff(removed[j], add)
					var sb strings.Builder
					sb.WriteString(diffAddedStyle.Render("+"))
					for _, seg := range segs {
						if seg.Added {
							sb.WriteString(diffAddedHighStyle.Render(seg.Text))
						} else if !seg.Removed {
							sb.WriteString(diffAddedStyle.Render(seg.Text))
						}
					}
					result = append(result, renderedLine{sb.String()})
				} else {
					result = append(result, renderedLine{diffAddedStyle.Render("+" + add)})
				}
			}
		case strings.HasPrefix(line, "+"):
			result = append(result, renderedLine{diffAddedStyle.Render(line)})
			i++
		default:
			result = append(result, renderedLine{toolDimStyle.Render(line)})
			i++
		}
	}

	var sb strings.Builder
	for _, r := range result {
		sb.WriteString(r.s + "\n")
	}
	return strings.TrimRight(sb.String(), "\n")
}

// countDiffChangedLines counts the number of added and removed lines in a diff.
func countDiffChangedLines(diff string) int {
	n := 0
	for _, l := range strings.Split(diff, "\n") {
		if (strings.HasPrefix(l, "+") && !strings.HasPrefix(l, "+++")) ||
			(strings.HasPrefix(l, "-") && !strings.HasPrefix(l, "---")) {
			n++
		}
	}
	return n
}

// formatExpandedArgs returns a pretty-printed, indented rendering of a tool
// call's JSON arguments for the expanded tool-call bubble.
func formatExpandedArgs(rawArgs string, width int) string {
	if rawArgs == "" || rawArgs == "{}" || rawArgs == "null" {
		return ""
	}
	var v any
	if err := json.Unmarshal([]byte(rawArgs), &v); err != nil {
		return toolDimStyle.Render("  " + rawArgs)
	}
	pretty, err := json.MarshalIndent(v, "  ", "  ")
	if err != nil {
		return toolDimStyle.Render("  " + rawArgs)
	}
	lines := strings.Split(string(pretty), "\n")
	var sb strings.Builder
	for i, l := range lines {
		if width > 4 && len(l) > width-2 {
			l = l[:width-5] + "…"
		}
		if i > 0 {
			sb.WriteByte('\n')
		}
		sb.WriteString(toolDimStyle.Render(l))
	}
	return sb.String()
}

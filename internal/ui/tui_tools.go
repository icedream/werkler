package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/tools"
	"github.com/muesli/reflow/wordwrap"
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

// toolOutputEvent is one event from a live-streaming tool execution.
type toolOutputEvent struct {
	Line   string         // stdout/stderr line (Done=false)
	Done   bool           // true when the tool has finished
	Result string         // final result JSON (Done=true)
	Diff   string         // file diff if any (Done=true)
	Parts  []ai.ImagePart // content parts (Done=true)
	Err    error          // execution error (Done=true)
}

// toolOutputChunkMsg carries one toolOutputEvent from the live-output channel.
type toolOutputChunkMsg struct {
	callID   string
	toolName string
	ch       <-chan toolOutputEvent
	event    toolOutputEvent
}

// readNextToolOutputChunk returns a Cmd that reads one event from ch and
// wraps it in a toolOutputChunkMsg.
func readNextToolOutputChunk(ch <-chan toolOutputEvent, callID, toolName string) tea.Cmd {
	return func() tea.Msg {
		event, ok := <-ch
		if !ok {
			// Channel closed without a Done event — treat as clean completion.
			event = toolOutputEvent{Done: true}
		}
		return toolOutputChunkMsg{callID: callID, toolName: toolName, ch: ch, event: event}
	}
}

// doCallToolStream starts a tool with live stdout/stderr streaming for tools
// that support it (currently run_command).  All other tools fall back to the
// regular single-shot doCallTool path.
func doCallToolStream(ctx context.Context, toolMgr *tools.Manager, session *chat.Session, tc ai.ToolCall) tea.Cmd {
	if !toolDescriptor(tc.Name).StreamOutput {
		return doCallTool(ctx, toolMgr, session, tc)
	}

	// rawLineCh receives individual output lines from the process via liveLineWriter.
	// Large buffer so the process's Write calls never block even if the TUI is slow.
	rawLineCh := make(chan string, 256)
	ctx = tools.WithLiveOutput(ctx, rawLineCh)

	// eventCh carries toolOutputEvents to the TUI (one per line, then one Done).
	eventCh := make(chan toolOutputEvent, 256)

	type resultT struct {
		result string
		diff   string
		parts  []ai.ImagePart
		err    error
	}
	resultCh := make(chan resultT, 1)

	// Goroutine 1: run the tool.
	// Explicitly close rawLineCh inside an inner func (not deferred from outer)
	// so the forwarding goroutine can drain all lines before reading the result.
	go func() {
		if toolMgr != nil {
			toolMgr.SetActiveCallID(tc.ID)
			defer toolMgr.SetActiveCallID("")
		}
		var res resultT
		func() {
			defer close(rawLineCh)
			r, parts, err := session.CallToolWithParts(ctx, tc)
			var diff string
			if err == nil {
				var m map[string]any
				if json.Unmarshal([]byte(r), &m) == nil {
					diff, _ = m["diff"].(string)
				}
			}
			res = resultT{r, diff, parts, err}
		}()
		resultCh <- res
	}()

	// Goroutine 2: forward lines to eventCh, then send the Done event.
	go func() {
		defer close(eventCh)
		for line := range rawLineCh {
			eventCh <- toolOutputEvent{Line: line}
		}
		res := <-resultCh
		eventCh <- toolOutputEvent{
			Done:   true,
			Result: res.result,
			Diff:   res.diff,
			Parts:  res.parts,
			Err:    res.err,
		}
	}()

	return readNextToolOutputChunk(eventCh, tc.ID, tc.Name)
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
	return toolBadgeArgs(toolName, args)
}

// toolCallIntent returns a short intent annotation for tools that carry a
// human-readable title field. Returns "" for all others.
func toolCallIntent(toolName string, args map[string]any) string {
	return toolIntent(toolName, args)
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

// parseRunCommandNote returns a short result annotation for a run_command call.
// Shows exit code and, when non-zero, the last non-empty output line so the
// user can see what failed without expanding the bubble.
func parseRunCommandNote(result string) string {
	var data struct {
		ExitCode int    `json:"exit_code"`
		TimedOut bool   `json:"timed_out"`
		Combined string `json:"combined_output"`
	}
	if err := json.Unmarshal([]byte(result), &data); err != nil {
		return ""
	}
	if data.TimedOut {
		return "timed out"
	}
	if data.ExitCode == 0 {
		return "exit: 0"
	}
	// Non-zero exit: show the last non-empty output line as a hint.
	lastLine := ""
	for _, l := range strings.Split(strings.TrimRight(data.Combined, "\n"), "\n") {
		if t := strings.TrimSpace(l); t != "" {
			lastLine = t
		}
	}
	if len(lastLine) > 80 {
		lastLine = lastLine[:79] + "…"
	}
	if lastLine != "" {
		return fmt.Sprintf("exit: %d — %s", data.ExitCode, lastLine)
	}
	return fmt.Sprintf("exit: %d", data.ExitCode)
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

// renderToolOutput renders stored run_command output (combined stdout+stderr)
// for the expanded tool-call bubble.  Each line is wordwrapped to width.
func renderToolOutput(output string, width int) string {
	if output == "" {
		return ""
	}
	maxW := width - 4
	if maxW < 20 {
		maxW = 20
	}
	var sb strings.Builder
	for i, l := range strings.Split(output, "\n") {
		if i > 0 {
			sb.WriteByte('\n')
		}
		wrapped := wordwrap.String(l, maxW)
		for j, wl := range strings.Split(wrapped, "\n") {
			if j > 0 {
				sb.WriteByte('\n')
			}
			if j == 0 {
				sb.WriteString(toolDimStyle.Render("  " + wl))
			} else {
				sb.WriteString(toolDimStyle.Render("    " + wl))
			}
		}
	}
	return sb.String()
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

// closePartialJSON adds the minimum closing characters needed to make a
// partial JSON string syntactically valid.  It handles nested braces/brackets
// and tracks whether a string literal is still open.
func closePartialJSON(s string) string {
	var stack []byte
	inString := false
	escaped := false

	for i := 0; i < len(s); i++ {
		c := s[i]
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && inString {
			escaped = true
			continue
		}
		if c == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}
		switch c {
		case '{':
			stack = append(stack, '}')
		case '[':
			stack = append(stack, ']')
		case '}', ']':
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
		}
	}

	var buf strings.Builder
	buf.WriteString(s)
	if inString {
		buf.WriteByte('"')
	}
	for i := len(stack) - 1; i >= 0; i-- {
		buf.WriteByte(stack[i])
	}
	return buf.String()
}

// jsonKeysInOrder returns the top-level object keys from a JSON string in
// document order.  json.Unmarshal into map[string]any loses order so we
// scan the raw text instead.
func jsonKeysInOrder(s string) []string {
	var keys []string
	inString := false
	escaped := false
	depth := 0
	keyStart := -1

	for i := 0; i < len(s); i++ {
		c := s[i]
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && inString {
			escaped = true
			continue
		}
		if c == '"' {
			if !inString {
				inString = true
				if depth == 1 {
					keyStart = i + 1
				}
			} else {
				inString = false
				if depth == 1 && keyStart >= 0 {
					// Only a key if followed by ':'
					j := i + 1
					for j < len(s) && (s[j] == ' ' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r') {
						j++
					}
					if j < len(s) && s[j] == ':' {
						keys = append(keys, s[keyStart:i])
					}
					keyStart = -1
				}
			}
			continue
		}
		if !inString {
			switch c {
			case '{', '[':
				depth++
			case '}', ']':
				if depth > 0 {
					depth--
				}
			}
		}
	}
	return keys
}

// renderStreamingArgs renders partially-received tool call arguments.
func renderStreamingArgs(rawArgs, toolName string, width int) string {
	return toolStreamingRenderer(toolName)(rawArgs, width)
}

// renderStreamingGeneric is the fallback: field-per-line with block cursor.
// renderStreamingGeneric is the fallback for unknown tools.
// Shows a stable cursor \u2014 never raw JSON text \u2014 while tokens arrive;
// transitions to field-per-line once the JSON is parseable.
// call's JSON arguments for the expanded tool-call bubble.
// renderStreamingGeneric is the fallback for unknown tools.
// Shows cursor only until JSON is parseable; never raw text.
func renderStreamingGeneric(rawArgs string, width int) string {
	if rawArgs == "" {
		return toolDimStyle.Render("  ▋")
	}
	closed := closePartialJSON(rawArgs)
	var parsed map[string]any
	if err := json.Unmarshal([]byte(closed), &parsed); err != nil || len(parsed) == 0 {
		return toolDimStyle.Render("  ▋")
	}
	keys := jsonKeysInOrder(rawArgs)
	if len(keys) == 0 {
		return toolDimStyle.Render("  ▋")
	}
	var sb strings.Builder
	for i, key := range keys {
		val, ok := parsed[key]
		if !ok {
			continue
		}
		var valStr string
		switch v := val.(type) {
		case string:
			valStr = `"` + v + `"`
		case nil:
			valStr = "null"
		default:
			b, _ := json.Marshal(v)
			valStr = string(b)
		}
		cursor := ""
		if i == len(keys)-1 {
			cursor = "▋"
		}
		maxVal := width - 18
		if maxVal < 12 {
			maxVal = 12
		}
		if len(valStr) > maxVal {
			if len(valStr) > 0 && valStr[0] == '"' {
				valStr = valStr[:maxVal-2] + `…"`
			} else {
				valStr = valStr[:maxVal-1] + "…"
			}
		}
		sb.WriteString(toolDimStyle.Render(fmt.Sprintf("  %-14s %s%s", key, valStr, cursor)))
		sb.WriteString("\n")
	}
	return strings.TrimRight(sb.String(), "\n")
}

// renderStreamingFileEdit renders the emerging diff as new_str tokens arrive.
// While only new_str is available it shows +lines (same as file_write);
// once both old_str and new_str are present it shows the full unified diff
// matching the final completed tool bubble style.
// renderStreamingFileEdit renders a live diff as old_str/new_str tokens arrive.
// Handles both the single-hunk form (top-level old_str/new_str) and the
// multi-hunk form (edits array) since the model may use either.
func renderStreamingFileEdit(rawArgs string, width int) string {
	closed := closePartialJSON(rawArgs)
	var parsed struct {
		Path   string `json:"path"`
		OldStr string `json:"old_str"` // single-hunk form
		NewStr string `json:"new_str"` // single-hunk form
		Edits  []struct {
			OldStr string `json:"old_str"`
			NewStr string `json:"new_str"`
		} `json:"edits"` // multi-hunk form
	}
	if err := json.Unmarshal([]byte(closed), &parsed); err != nil ||
		(parsed.Path == "" && parsed.OldStr == "" && parsed.NewStr == "" && len(parsed.Edits) == 0) {
		return toolDimStyle.Render("  ▋")
	}

	// Normalise: treat single-hunk fields as a one-element edit slice so the
	// rendering loop handles both forms uniformly.
	type editPair struct{ OldStr, NewStr string }
	var edits []editPair
	if len(parsed.Edits) > 0 {
		for _, e := range parsed.Edits {
			edits = append(edits, editPair{e.OldStr, e.NewStr})
		}
	} else {
		edits = []editPair{{parsed.OldStr, parsed.NewStr}}
	}

	var sb strings.Builder
	if parsed.Path != "" {
		sb.WriteString(toolDimStyle.Render(fmt.Sprintf("  path  %q", parsed.Path)))
		sb.WriteString("\n")
	}
	for _, edit := range edits {
		switch {
		case edit.OldStr != "" && edit.NewStr != "":
			if diff := tools.ComputeUnifiedDiff(edit.OldStr, edit.NewStr, parsed.Path); diff != "" {
				sb.WriteString(renderDiff(diff, width))
				sb.WriteString("\n")
			}
		case edit.OldStr != "":
			for _, l := range strings.Split(edit.OldStr, "\n") {
				sb.WriteString(diffRemovedStyle.Render("-" + l))
				sb.WriteString("\n")
			}
		case edit.NewStr != "":
			for _, l := range strings.Split(edit.NewStr, "\n") {
				sb.WriteString(diffAddedStyle.Render("+" + l))
				sb.WriteString("\n")
			}
		}
	}
	result := strings.TrimRight(sb.String(), "\n")
	if result == "" {
		if parsed.Path != "" {
			return toolDimStyle.Render(fmt.Sprintf("  path  %q▋", parsed.Path))
		}
		return toolDimStyle.Render("  ▋")
	}
	return result + toolDimStyle.Render("▋")
}

// renderStreamingFileWrite streams file content as +lines.
func renderStreamingFileWrite(rawArgs string, width int) string {
	closed := closePartialJSON(rawArgs)
	var parsed struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(closed), &parsed); err != nil ||
		(parsed.Path == "" && parsed.Content == "") {
		return toolDimStyle.Render("  ▋")
	}
	var sb strings.Builder
	if parsed.Path != "" {
		sb.WriteString(toolDimStyle.Render(fmt.Sprintf("  path  %q", parsed.Path)))
		sb.WriteString("\n")
	}
	for _, l := range strings.Split(parsed.Content, "\n") {
		sb.WriteString(diffAddedStyle.Render("+" + l))
		sb.WriteString("\n")
	}
	result := strings.TrimRight(sb.String(), "\n")
	if result == "" {
		if parsed.Path != "" {
			return toolDimStyle.Render(fmt.Sprintf("  path  %q▋", parsed.Path))
		}
		return toolDimStyle.Render("  ▋")
	}
	return result + toolDimStyle.Render("▋")
}

// renderStreamingFilePath shows just the path (file_delete).
func renderStreamingFilePath(rawArgs string, width int) string {
	closed := closePartialJSON(rawArgs)
	var parsed struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal([]byte(closed), &parsed); err != nil || parsed.Path == "" {
		return toolDimStyle.Render("  ▋")
	}
	return toolDimStyle.Render(fmt.Sprintf("  path  %q▋", parsed.Path))
}

// renderStreamingCommand shows the command as it arrives.
func renderStreamingCommand(rawArgs string, width int) string {
	closed := closePartialJSON(rawArgs)
	var parsed struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal([]byte(closed), &parsed); err != nil || parsed.Command == "" {
		return toolDimStyle.Render("  $ ▋")
	}
	cmd := parsed.Command
	if width > 6 && len(cmd) > width-4 {
		cmd = "…" + cmd[len(cmd)-(width-6):]
	}
	return toolDimStyle.Render("  $ " + cmd + "▋")
}

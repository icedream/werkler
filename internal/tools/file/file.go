package file

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
	"github.com/icedream/werkler/internal/tools/toolutil"
)

// PathApprover is the minimal interface used for read/write approval checks.
type PathApprover interface {
	IsPathReadApproved(path string) bool
	IsPathWriteApproved(path string) bool
}

// Context is the narrow interface the file handler needs from the Manager.
type Context interface {
	// ActiveApprover returns the path approver for this request, or nil.
	ActiveApprover(ctx context.Context) PathApprover
}

// Handler holds the file tool handlers.
type Handler struct{ ctx Context }

// NewHandler creates a Handler.
func NewHandler(ctx Context) *Handler { return &Handler{ctx: ctx} }

// Tools returns Builtin definitions for all file tools.
func (h *Handler) Tools() []toolutil.Builtin {
	return []toolutil.Builtin{
		{Def: fileReadMultiDef, Handle: h.handleFileReadMulti},
		{Def: readImageDef, HandleWithParts: h.handleReadImage},
		{Def: fileListDef, Handle: h.handleFileList},
		{Def: fileWriteDef, Handle: h.handleFileWrite},
		{Def: fileEditDef, Handle: h.handleFileEdit},
		{Def: fileDeleteDef, Handle: h.handleFileDelete},
	}
}

// ─── Schemas ─────────────────────────────────────────────────────────────────

var fileReadMultiDef = ai.ToolDefinition{
	Name: "file_read_multi",
	Description: `Read text file regions, multiple in one call possible. Returns each region labeled with a header line.
Each region may specify start_line and end_line (1-indexed, inclusive); omit both to read the full file.
Partial failures are reported inline — other regions still return their content.
Total output is capped at 8 KiB across all regions.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"regions": map[string]any{
				"type":        "array",
				"description": "List of regions to read",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"path":       map[string]any{"type": "string", "description": "Absolute or ~ path to the file"},
						"start_line": map[string]any{"type": "number", "description": "First line to return (1-indexed, default 1)"},
						"end_line":   map[string]any{"type": "number", "description": "Last line to return (1-indexed, default: end of file)"},
					},
					"required": []string{"path"},
				},
			},
		},
		"required": []string{"regions"},
	},
}

var readImageDef = ai.ToolDefinition{
	Name:        "read_image",
	Description: "Read an image file and return it as a base64-encoded image for visual analysis. ONLY use for image files (PNG, JPEG, GIF, WebP). DO NOT use for text documents, PDFs, or any non-image files. For reading text files, use file_read_multi instead.",
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string", "description": "Absolute or ~ path to the image file"},
		},
		"required": []string{"path"},
	},
}

var fileListDef = ai.ToolDefinition{
	Name:        "file_list",
	Description: "List the contents of a directory. Returns a JSON array of {name, type, size} objects.",
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string", "description": "Absolute or ~ path to the directory"},
		},
		"required": []string{"path"},
	},
}

var fileWriteDef = ai.ToolDefinition{
	Name: "file_write",
	Description: `Create a new file or overwrite an existing file with the given text content.
This is the correct tool to use whenever you need to write a file whole.
To create a new file including its parent directories, set create_parents to true.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":           map[string]any{"type": "string", "description": "Absolute or ~ path to the file"},
			"content":        map[string]any{"type": "string", "description": "File contents (UTF-8)"},
			"create_parents": map[string]any{"type": "boolean", "description": "Create parent directories if they don't exist (default false)"},
		},
		"required": []string{"path", "content"},
	},
}

var fileEditDef = ai.ToolDefinition{
	Name: "file_edit",
	Description: `Replace one or more exact-string occurrences in a file.

Single-hunk form (most common):
  path, old_str, new_str — replace one occurrence of old_str with new_str.

Multi-hunk form (use when making several changes to the same file in one call):
  path, edits: [{old_str, new_str}, …] — apply each replacement in order.
  All hunks are validated before any writes; the call fails atomically if any
  old_str is not found or matches more than once.

In both forms: the match must be exact (including whitespace). Returns an error
if old_str appears zero times (not found) or more than once (ambiguous — include
more surrounding context).
On success, returns the line number(s) where replacements were made.

Field generation order (important for live diff rendering):
  Generate fields in this order: path first, then old_str, then new_str.
  For multi-hunk edits: path first, then each edit's old_str before new_str.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":    map[string]any{"type": "string", "description": "Absolute or ~ path to the file"},
			"old_str": map[string]any{"type": "string", "description": "Exact text to find and replace (single-hunk form)"},
			"new_str": map[string]any{"type": "string", "description": "Replacement text (single-hunk form)"},
			"edits": map[string]any{
				"type":        "array",
				"description": "List of {old_str, new_str} pairs applied in order (multi-hunk form). Mutually exclusive with top-level old_str/new_str.",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"old_str": map[string]any{"type": "string"},
						"new_str": map[string]any{"type": "string"},
					},
					"required": []string{"old_str", "new_str"},
				},
			},
		},
		"required": []string{"path"},
	},
}

var fileDeleteDef = ai.ToolDefinition{
	Name: "file_delete",
	Description: `Delete a file or symlink at the given path.

If path is a symlink, only the symlink itself is removed — the file it
points to is left untouched. To verify the target still exists afterwards
you can read it with file_read.

Never removes directories (use process_start with rm -r for that).`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string", "description": "Absolute or ~ path to the file"},
		},
		"required": []string{"path"},
	},
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const maxFileReadMultiBytes = 8 << 10 // 8 KiB

func resolvePath(p, baseDir string) string {
	if strings.HasPrefix(p, "~/") {
		u, _ := user.Current()
		home := ""
		if u != nil {
			home = u.HomeDir
		}
		p = filepath.Join(home, p[2:])
	}
	if !filepath.IsAbs(p) && baseDir != "" {
		p = filepath.Join(baseDir, p)
	}
	return filepath.Clean(p)
}

func canonicalizePath(p string) string {
	p = resolvePath(p, "")
	if !filepath.IsAbs(p) {
		if cwd, err := os.Getwd(); err == nil {
			p = filepath.Join(cwd, p)
		}
	}
	if resolved, err := filepath.EvalSymlinks(p); err == nil {
		return resolved
	}
	if resolvedDir, err := filepath.EvalSymlinks(filepath.Dir(p)); err == nil {
		return filepath.Join(resolvedDir, filepath.Base(p))
	}
	return filepath.Clean(p)
}

func findMatchLines(text, substr string) []int {
	var lines []int
	offset := 0
	for {
		idx := strings.Index(text[offset:], substr)
		if idx < 0 {
			break
		}
		abs := offset + idx
		lines = append(lines, strings.Count(text[:abs], "\n")+1)
		offset = abs + len(substr)
	}
	return lines
}

func jsonResult(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func (h *Handler) checkSingleRead(ctx context.Context, path string) *toolutil.UnapprovedPathsError {
	approver := h.ctx.ActiveApprover(ctx)
	if approver == nil {
		return nil
	}
	if !approver.IsPathReadApproved(path) {
		return &toolutil.UnapprovedPathsError{Requests: []toolutil.PathAccessRequest{{Path: path, Write: false}}}
	}
	return nil
}

func (h *Handler) checkSingleWrite(ctx context.Context, path string) *toolutil.UnapprovedPathsError {
	approver := h.ctx.ActiveApprover(ctx)
	if approver == nil {
		return nil
	}
	if !approver.IsPathWriteApproved(path) {
		return &toolutil.UnapprovedPathsError{Requests: []toolutil.PathAccessRequest{{Path: path, Write: true}}}
	}
	return nil
}

// ─── Handlers ────────────────────────────────────────────────────────────────

func (h *Handler) handleFileReadMulti(ctx context.Context, args map[string]any) (string, error) {
	rawRegions, _ := args["regions"].([]any)
	if len(rawRegions) == 0 {
		return "", fmt.Errorf("file_read_multi: regions must be a non-empty array")
	}
	type region struct {
		rawPath   string
		path      string
		startLine int
		endLine   int
		hasStart  bool
		hasEnd    bool
	}
	regions := make([]region, 0, len(rawRegions))
	for i, r := range rawRegions {
		rm, ok := r.(map[string]any)
		if !ok {
			return "", fmt.Errorf("file_read_multi: region[%d] is not an object", i)
		}
		rp := toolutil.StringArg(rm, "path")
		if rp == "" {
			return "", fmt.Errorf("file_read_multi: region[%d]: path is required", i)
		}
		reg := region{rawPath: rp, path: canonicalizePath(rp)}
		if v, ok := rm["start_line"]; ok {
			if f, ok2 := toolutil.ToFloat64(v); ok2 {
				reg.startLine = int(f)
			}
			reg.hasStart = true
		}
		if v, ok := rm["end_line"]; ok {
			if f, ok2 := toolutil.ToFloat64(v); ok2 {
				reg.endLine = int(f)
			}
			reg.hasEnd = true
		}
		regions = append(regions, reg)
	}
	approver := h.ctx.ActiveApprover(ctx)
	if approver != nil {
		seen := make(map[string]bool)
		var unapproved []chat.PathAccessRequest
		for _, reg := range regions {
			if !seen[reg.path] && !approver.IsPathReadApproved(reg.path) {
				unapproved = append(unapproved, chat.PathAccessRequest{Path: reg.path, Write: false})
				seen[reg.path] = true
			}
		}
		if len(unapproved) > 0 {
			return "", &toolutil.UnapprovedPathsError{Requests: unapproved}
		}
	}
	var out strings.Builder
	totalBytes := 0
	for i, reg := range regions {
		if i > 0 {
			out.WriteString("\n")
		}
		info, err := os.Stat(reg.path)
		if err != nil {
			fmt.Fprintf(&out, "=== %s [ERROR] ===\n%s\n", reg.rawPath, err.Error())
			continue
		}
		if info.IsDir() {
			fmt.Fprintf(&out, "=== %s [ERROR] ===\ndirectory; use file_list\n", reg.rawPath)
			continue
		}
		data, err := os.ReadFile(reg.path)
		if err != nil {
			fmt.Fprintf(&out, "=== %s [ERROR] ===\n%s\n", reg.rawPath, err.Error())
			continue
		}
		if !utf8.Valid(data) {
			fmt.Fprintf(&out, "=== %s [ERROR] ===\nbinary file; use process_start\n", reg.rawPath)
			continue
		}
		lines := strings.Split(string(data), "\n")
		totalLines := len(lines)
		startLine := 1
		endLine := totalLines
		if reg.hasStart {
			startLine = reg.startLine
		}
		if reg.hasEnd {
			endLine = reg.endLine
		}
		if startLine < 1 {
			startLine = 1
		}
		if endLine > totalLines {
			endLine = totalLines
		}
		if startLine > endLine {
			fmt.Fprintf(&out, "=== %s [ERROR] ===\nstart_line (%d) > end_line (%d)\n", reg.rawPath, startLine, endLine)
			continue
		}
		selected := lines[startLine-1 : endLine]
		rangeLabel := fmt.Sprintf("L%d-L%d", startLine, startLine+len(selected)-1)
		header := fmt.Sprintf("=== %s [%s of %d] ===\n", reg.rawPath, rangeLabel, totalLines)
		out.WriteString(header)
		totalBytes += len(header)
		var sectionBuf strings.Builder
		for i, l := range selected {
			fmt.Fprintf(&sectionBuf, "%4d│%s\n", startLine+i, l)
		}
		section := sectionBuf.String()
		remaining := maxFileReadMultiBytes - totalBytes
		if remaining <= 0 {
			out.WriteString("[output cap reached — omitted]\n")
			break
		}
		if len(section) > remaining {
			out.WriteString(section[:remaining])
			out.WriteString("\n[output cap reached — truncated]\n")
			break
		}
		out.WriteString(section)
		totalBytes += len(section)
	}
	return out.String(), nil
}

func (h *Handler) handleReadImage(ctx context.Context, args map[string]any) (string, []ai.ImagePart, error) {
	rawPath := toolutil.StringArg(args, "path")
	if rawPath == "" {
		return "", nil, fmt.Errorf("read_image: path is required")
	}
	path := canonicalizePath(rawPath)
	if err := h.checkSingleRead(ctx, path); err != nil {
		return "", nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", nil, fmt.Errorf("read_image: %w", err)
	}
	mime := http.DetectContentType(data)
	if mime == "application/octet-stream" {
		ext := strings.ToLower(filepath.Ext(path))
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
			return "", nil, fmt.Errorf("read_image: unsupported image format (extension %s)", ext)
		}
	}
	if !strings.HasPrefix(mime, "image/") {
		return "", nil, fmt.Errorf("read_image: %s is not an image file (detected content type: %s). For reading text documents or PDFs, use file_read_multi instead.", path, mime)
	}
	part := ai.ImagePart{Data: data, MIMEType: mime, Name: filepath.Base(path)}
	return fmt.Sprintf("Image loaded: %s (%s, %d bytes)", filepath.Base(path), mime, len(data)), []ai.ImagePart{part}, nil
}

func (h *Handler) handleFileList(ctx context.Context, args map[string]any) (string, error) {
	rawPath := toolutil.StringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_list: path is required")
	}
	path := canonicalizePath(rawPath)
	if err := h.checkSingleRead(ctx, path); err != nil {
		return "", err
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("file_list: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("file_list: %s is not a directory; use file_read to read files", path)
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return "", fmt.Errorf("file_list: %w", err)
	}
	type entry struct {
		Name string `json:"name"`
		Type string `json:"type"`
		Size *int64 `json:"size,omitempty"`
	}
	result := make([]entry, 0, len(entries))
	for _, e := range entries {
		var kind string
		var size *int64
		switch {
		case e.IsDir():
			kind = "directory"
		case e.Type()&fs.ModeSymlink != 0:
			kind = "symlink"
		default:
			kind = "file"
			if fi, err2 := e.Info(); err2 == nil {
				s := fi.Size()
				size = &s
			}
		}
		result = append(result, entry{Name: e.Name(), Type: kind, Size: size})
	}
	return jsonResult(result), nil
}

func (h *Handler) handleFileWrite(ctx context.Context, args map[string]any) (string, error) {
	rawPath := toolutil.StringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_write: path is required")
	}
	path := canonicalizePath(rawPath)
	content := toolutil.StringArg(args, "content")
	createParents := toolutil.BoolArg(args, "create_parents")
	if err := h.checkSingleWrite(ctx, path); err != nil {
		return "", err
	}
	if createParents {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return "", fmt.Errorf("file_write: creating parent directories: %w", err)
		}
	}
	oldBytes, _ := os.ReadFile(path)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("file_write: %w", err)
	}
	return jsonResult(map[string]any{
		"path":  path,
		"bytes": len(content),
		"diff":  toolutil.ComputeUnifiedDiff(string(oldBytes), content, path),
	}), nil
}

func (h *Handler) handleFileEdit(ctx context.Context, args map[string]any) (string, error) {
	rawPath := toolutil.StringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_edit: path is required")
	}
	path := canonicalizePath(rawPath)
	type hunk struct{ old, new string }
	var hunks []hunk
	if rawEdits, ok := args["edits"].([]any); ok && len(rawEdits) > 0 {
		for i, item := range rawEdits {
			m, ok := item.(map[string]any)
			if !ok {
				return "", fmt.Errorf("file_edit: edits[%d] must be an object", i)
			}
			old, _ := m["old_str"].(string)
			if old == "" {
				return "", fmt.Errorf("file_edit: edits[%d].old_str must not be empty", i)
			}
			newVal, _ := m["new_str"].(string)
			hunks = append(hunks, hunk{old, newVal})
		}
	} else {
		oldStr := toolutil.StringArg(args, "old_str")
		newStr := toolutil.StringArg(args, "new_str")
		if oldStr == "" {
			return "", fmt.Errorf("file_edit: old_str must not be empty; use file_write to overwrite entire files")
		}
		hunks = []hunk{{oldStr, newStr}}
	}
	if err := h.checkSingleWrite(ctx, path); err != nil {
		return "", err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("file_edit: %w", err)
	}
	if !utf8.Valid(data) {
		return "", fmt.Errorf("file_edit: %s appears to be a binary file", path)
	}
	content := string(data)
	for i, hk := range hunks {
		count := strings.Count(content, hk.old)
		switch {
		case count == 0:
			if len(hunks) == 1 {
				return "", fmt.Errorf("file_edit: old_str not found in %s — call file_read on the file first and use the exact text from the file as old_str (watch for whitespace and indentation)", path)
			}
			return "", fmt.Errorf("file_edit: edits[%d] old_str not found in %s", i, path)
		case count > 1:
			lineNums := findMatchLines(content, hk.old)
			if len(hunks) == 1 {
				return "", fmt.Errorf("file_edit: old_str matches %d times in %s (at lines %v); include more surrounding context to make it unique", count, path, lineNums)
			}
			return "", fmt.Errorf("file_edit: edits[%d] old_str matches %d times in %s (at lines %v); include more surrounding context", i, count, path, lineNums)
		}
	}
	type result struct {
		line    int
		added   int
		removed int
	}
	results := make([]result, len(hunks))
	for i, hk := range hunks {
		idx := strings.Index(content, hk.old)
		line := strings.Count(content[:idx], "\n") + 1
		removed := strings.Count(hk.old, "\n") + 1
		added := strings.Count(hk.new, "\n") + 1
		results[i] = result{line, added, removed}
		content = strings.Replace(content, hk.old, hk.new, 1)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("file_edit: writing %s: %w", path, err)
	}
	newContent, _ := os.ReadFile(path)
	diff := toolutil.ComputeUnifiedDiff(string(data), string(newContent), path)
	if len(hunks) == 1 {
		return jsonResult(map[string]any{
			"path":    path,
			"line":    results[0].line,
			"added":   results[0].added,
			"removed": results[0].removed,
			"diff":    diff,
		}), nil
	}
	editsOut := make([]map[string]any, len(results))
	for i, r := range results {
		editsOut[i] = map[string]any{"line": r.line, "added": r.added, "removed": r.removed}
	}
	return jsonResult(map[string]any{"path": path, "edits": editsOut, "diff": diff}), nil
}

func (h *Handler) handleFileDelete(ctx context.Context, args map[string]any) (string, error) {
	rawPath := toolutil.StringArg(args, "path")
	if rawPath == "" {
		return "", fmt.Errorf("file_delete: path is required")
	}
	resolvedPath := canonicalizePath(rawPath)
	if err := h.checkSingleWrite(ctx, resolvedPath); err != nil {
		return "", err
	}
	deletePath := resolvePath(rawPath, "")
	if !filepath.IsAbs(deletePath) {
		if cwd, err := os.Getwd(); err == nil {
			deletePath = filepath.Join(cwd, deletePath)
		}
	}
	deletePath = filepath.Clean(deletePath)
	info, err := os.Lstat(deletePath)
	if err != nil {
		return "", fmt.Errorf("file_delete: %w", err)
	}
	if info.Mode()&os.ModeSymlink == 0 && info.IsDir() {
		return "", fmt.Errorf("file_delete: %s is a directory; use process_start with rm -r to delete directories", deletePath)
	}
	oldBytes, _ := os.ReadFile(resolvedPath)
	if err := os.Remove(deletePath); err != nil {
		return "", fmt.Errorf("file_delete: %w", err)
	}
	return jsonResult(map[string]any{
		"deleted": deletePath,
		"diff":    toolutil.ComputeUnifiedDiff(string(oldBytes), "", deletePath),
	}), nil
}

package file

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ParseEditNote extracts a human-readable annotation from a successful
// file_edit result JSON, e.g. "+3 -2 lines @ line 42".
// Returns plain text; callers may apply styling to the + and - prefixes.
func ParseEditNote(result string) (added, removed, line int, ok bool) {
	var data map[string]any
	if err := json.Unmarshal([]byte(result), &data); err != nil {
		return
	}
	a, _ := data["added"].(float64)
	r, _ := data["removed"].(float64)
	l, _ := data["line"].(float64)
	if a == 0 && r == 0 {
		return
	}
	return int(a), int(r), int(l), true
}

// ParseWriteNote extracts a human-readable annotation from a successful
// file_write result JSON, e.g. "wrote 1.2 KB".
func ParseWriteNote(result string) string {
	var data map[string]any
	if err := json.Unmarshal([]byte(result), &data); err != nil {
		return ""
	}
	bytes, _ := data["bytes"].(float64)
	return fmt.Sprintf("wrote %s", formatBytes(int64(bytes)))
}

// ParseDeleteNote extracts an annotation from a file_delete result JSON.
func ParseDeleteNote(result string) string {
	var data map[string]any
	if err := json.Unmarshal([]byte(result), &data); err != nil {
		return ""
	}
	if _, ok := data["deleted"]; ok {
		return "deleted"
	}
	return ""
}

// EditErrorNote returns a short, user-readable summary of a file_edit error.
func EditErrorNote(err error) string {
	return toolErrorNote("file_edit: ", err)
}

// WriteErrorNote returns a short, user-readable summary of a file_write error.
func WriteErrorNote(err error) string {
	return toolErrorNote("file_write: ", err)
}

func toolErrorNote(prefix string, err error) string {
	if err == nil {
		return ""
	}
	msg := strings.TrimPrefix(err.Error(), prefix)
	if i := strings.Index(msg, " — "); i > 0 {
		msg = msg[:i]
	}
	if len(msg) > 100 {
		msg = msg[:100] + "…"
	}
	return msg
}

func formatBytes(n int64) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

package process

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ParseRunCommandNote returns a short result annotation for a run_command call:
// "exit: 0", "timed out", or "exit: N — <last non-empty output line>".
func ParseRunCommandNote(result string) string {
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

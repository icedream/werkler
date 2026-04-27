// Package pathutil provides small path-manipulation helpers.
package pathutil

import (
	"os"
	"path/filepath"
	"strings"
)

// ExpandTilde replaces a leading ~ with the user's home directory.
// Returns the path unchanged when the home directory cannot be determined.
func ExpandTilde(path string) string {
	if path == "~" {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return home
	}
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return filepath.Join(home, path[2:])
	}
	return path
}

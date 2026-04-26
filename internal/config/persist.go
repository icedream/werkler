package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pelletier/go-toml/v2"
)

// AppendMCPServer appends a new [[mcp.servers]] entry to the config file at
// configPath (defaults to DefaultConfigPath when empty). If a server with the
// same name already exists, it is a no-op and returns nil.
func AppendMCPServer(configPath string, srv MCPServerConfig) error {
	if configPath == "" {
		configPath = DefaultConfigPath()
	}

	// Idempotency: parse existing config to check for duplicates.
	var parsed struct {
		MCP struct {
			Servers []struct {
				Name string `toml:"name"`
			} `toml:"servers"`
		} `toml:"mcp"`
	}
	rawData, readErr := os.ReadFile(configPath)
	if readErr == nil {
		_ = toml.Unmarshal(rawData, &parsed)
		for _, existing := range parsed.MCP.Servers {
			if existing.Name == srv.Name {
				return nil
			}
		}
	}

	if err := os.MkdirAll(filepath.Dir(configPath), 0o700); err != nil {
		return fmt.Errorf("creating config directory: %w", err)
	}

	// Build the TOML fragment for the new server entry.
	var entry strings.Builder
	entry.WriteString("\n[[mcp.servers]]\n")
	fmt.Fprintf(&entry, "name = %s\n", tomlString(srv.Name))
	switch srv.Transport {
	case MCPTransportStreamable, MCPTransportSSE:
		fmt.Fprintf(&entry, "transport = %s\n", tomlString(string(srv.Transport)))
		fmt.Fprintf(&entry, "url = %s\n", tomlString(srv.URL))
	case MCPTransportStdio:
		fmt.Fprintf(&entry, "transport = %s\n", tomlString(string(srv.Transport)))
		fmt.Fprintf(&entry, "command = %s\n", tomlString(srv.Command))
	}

	if readErr != nil {
		// File doesn't exist — create from scratch.
		content := strings.TrimPrefix(entry.String(), "\n")
		return os.WriteFile(configPath, []byte(content), 0o600)
	}

	// Append to existing file.
	text := string(rawData)
	if !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	return os.WriteFile(configPath, []byte(text+entry.String()[1:]), 0o600)
}

// AppendAutoApproveTool appends pattern to mcp.auto_approve_tools in the config
// file at configPath (defaults to DefaultConfigPath when empty).
// If the file does not exist it is created. If pattern is already present it is
// a no-op. Existing comments and formatting are preserved.
func AppendAutoApproveTool(configPath, pattern string) error {
	if configPath == "" {
		configPath = DefaultConfigPath()
	}
	return appendToStringSlice(configPath, "mcp", "auto_approve_tools", pattern)
}

// AppendAutoApprovePath appends pattern to mcp.auto_approve_paths in the config
// file at configPath (defaults to DefaultConfigPath when empty).
// If the file does not exist it is created. If pattern is already present it is
// a no-op. Existing comments and formatting are preserved.
func AppendAutoApprovePath(configPath, pattern string) error {
	if configPath == "" {
		configPath = DefaultConfigPath()
	}
	return appendToStringSlice(configPath, "mcp", "auto_approve_paths", pattern)
}

// appendToStringSlice surgically appends value to a TOML string array at
// section.key, preserving all comments and surrounding formatting.
//
// It uses go-toml/v2 only for the idempotency check (parsing to read the
// current values). The actual file modification is done as a text operation:
//
//   - If the array already exists, value is inserted on a new indented line
//     before the closing "]". Single-line arrays are expanded to multiline
//     before inserting.
//   - If the key is absent but the section exists, a new multiline array is
//     inserted after the section header.
//   - If neither exists, both section and key are appended to the end of the
//     file.
func appendToStringSlice(configPath, section, key, value string) error {
	// --- idempotency check via structured parse ---
	var parsed struct {
		MCP map[string]any `toml:"mcp"`
	}
	rawData, readErr := os.ReadFile(configPath)
	if readErr == nil {
		_ = toml.Unmarshal(rawData, &parsed)
		if arr, ok := parsed.MCP[key]; ok {
			if items, ok := arr.([]any); ok {
				for _, item := range items {
					if s, ok := item.(string); ok && s == value {
						return nil // already present
					}
				}
			}
		}
	}

	// Ensure parent directory exists (needed when creating a new file).
	if err := os.MkdirAll(filepath.Dir(configPath), 0o700); err != nil {
		return fmt.Errorf("creating config directory: %w", err)
	}

	// If the file doesn't exist, create it from scratch.
	if readErr != nil {
		content := fmt.Sprintf("[%s]\n%s = [\n  %s,\n]\n", section, key, tomlString(value))
		return os.WriteFile(configPath, []byte(content), 0o600)
	}

	text := string(rawData)
	result, err := insertIntoText(text, section, key, value)
	if err != nil {
		return err
	}
	return os.WriteFile(configPath, []byte(result), 0o600)
}

// insertIntoText performs the surgical text insertion. It is separated from
// the file I/O so it can be tested directly.
func insertIntoText(text, section, key, value string) (string, error) {
	lines := strings.Split(text, "\n")
	sectionHeader := "[" + section + "]"
	keyPrefix := key + " "   // "auto_approve_tools " — broad match before "="
	keyPrefixEq := key + "=" // tight match without space

	// Locate the target section and the key within it.
	sectionLine := -1
	keyLine := -1
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Detect section headers. Stop searching for key if we've entered a
		// different section after finding our target section.
		if strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]") {
			if trimmed == sectionHeader {
				sectionLine = i
			} else if sectionLine >= 0 {
				// Entered a new section — key was not found in ours.
				break
			}
			continue
		}

		if sectionLine >= 0 {
			bare := strings.TrimSpace(strings.SplitN(trimmed, "#", 2)[0])
			if strings.HasPrefix(bare, keyPrefix) || strings.HasPrefix(bare, keyPrefixEq) {
				keyLine = i
				break
			}
		}
	}

	quoted := tomlString(value)

	switch {
	case keyLine >= 0:
		// Key found — insert into (or expand) the existing array.
		return insertIntoArray(lines, keyLine, quoted)

	case sectionLine >= 0:
		// Section exists but key is absent — insert after the section header.
		newEntry := fmt.Sprintf("%s = [\n  %s,\n]", key, quoted)
		lines = append(lines[:sectionLine+1], append([]string{newEntry}, lines[sectionLine+1:]...)...)
		return strings.Join(lines, "\n"), nil

	default:
		// Neither section nor key — append both to the end.
		suffix := fmt.Sprintf("\n[%s]\n%s = [\n  %s,\n]\n", section, key, quoted)
		if strings.HasSuffix(text, "\n") {
			suffix = suffix[1:] // avoid double blank line
		}
		return text + suffix, nil
	}
}

// insertIntoArray inserts quoted into the array whose opening "[" is on keyLine.
// Handles both single-line ( key = ["a"] ) and multiline arrays.
func insertIntoArray(lines []string, keyLine int, quoted string) (string, error) {
	// Find the extent of the array: from the "[" on keyLine to the closing "]".
	// We join lines starting at keyLine until we accumulate a balanced "[...]".
	depth := 0
	closingLine := -1
	inString := false
	escape := false
	for i := keyLine; i < len(lines); i++ {
		for _, ch := range lines[i] {
			if escape {
				escape = false
				continue
			}
			if ch == '\\' && inString {
				escape = true
				continue
			}
			if ch == '"' {
				inString = !inString
				continue
			}
			if inString {
				continue
			}
			if ch == '[' {
				depth++
			} else if ch == ']' {
				depth--
				if depth == 0 {
					closingLine = i
					break
				}
			}
		}
		if closingLine >= 0 {
			break
		}
	}
	if closingLine < 0 {
		return "", fmt.Errorf("unterminated TOML array at line %d", keyLine+1)
	}

	if closingLine == keyLine {
		// Single-line array: expand to multiline, then append.
		// e.g.  key = ["a"]  →  key = [\n  "a",\n  "new",\n]
		line := lines[keyLine]
		bracket := strings.Index(line, "[")
		if bracket < 0 {
			return "", fmt.Errorf("expected '[' on line %d", keyLine+1)
		}
		prefix := line[:bracket+1]
		inner := strings.TrimSpace(line[bracket+1 : strings.LastIndex(line, "]")])
		// Closing ] aligns with the start of the line (same indentation as the key).
		lineIndent := line[:len(line)-len(strings.TrimLeft(line, " \t"))]
		itemIndent := lineIndent + "  "

		var elems []string
		if inner != "" {
			for _, elem := range strings.Split(inner, ",") {
				if e := strings.TrimSpace(elem); e != "" {
					elems = append(elems, itemIndent+e+",")
				}
			}
		}
		elems = append(elems, itemIndent+quoted+",")

		replacement := prefix + "\n" + strings.Join(elems, "\n") + "\n" + lineIndent + "]"
		// Preserve anything after the closing ] (e.g. inline comment — rare but valid).
		suffix := line[strings.LastIndex(line, "]")+1:]
		lines[keyLine] = replacement + suffix
	} else {
		// Multiline array: find the line of the closing "]" and insert before it.
		closingIndent := strings.TrimRight(lines[closingLine], "]\t ")
		indent := closingIndent + "  "
		if indent == "  " {
			indent = "  " // always at least 2-space indent
		}
		newLine := indent + quoted + ","
		lines = append(lines[:closingLine], append([]string{newLine}, lines[closingLine:]...)...)
	}

	return strings.Join(lines, "\n"), nil
}

// tomlString returns value as a quoted TOML string literal.
func tomlString(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	return `"` + s + `"`
}

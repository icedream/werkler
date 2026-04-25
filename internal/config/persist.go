package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/pelletier/go-toml/v2"
)

// AppendAutoApproveTool appends pattern to mcp.auto_approve_tools in the config
// file at configPath (defaults to DefaultConfigPath when empty).
// If the file does not exist it is created. If pattern is already present it is
// a no-op. Comments in the existing file will be lost on rewrite.
func AppendAutoApproveTool(configPath, pattern string) error {
	if configPath == "" {
		configPath = DefaultConfigPath()
	}
	return appendToStringSlice(configPath, []string{"mcp", "auto_approve_tools"}, pattern)
}

// AppendAutoApprovePath appends pattern to mcp.auto_approve_paths in the config
// file at configPath (defaults to DefaultConfigPath when empty).
// If the file does not exist it is created. If pattern is already present it is
// a no-op. Comments in the existing file will be lost on rewrite.
func AppendAutoApprovePath(configPath, pattern string) error {
	if configPath == "" {
		configPath = DefaultConfigPath()
	}
	return appendToStringSlice(configPath, []string{"mcp", "auto_approve_paths"}, pattern)
}

// appendToStringSlice reads configPath as TOML, appends value to the string
// array at keyPath (creating intermediate tables as needed), and writes it back.
func appendToStringSlice(configPath string, keyPath []string, value string) error {
	raw := map[string]any{}
	if data, err := os.ReadFile(configPath); err == nil {
		if err := toml.Unmarshal(data, &raw); err != nil {
			return fmt.Errorf("parsing config %q: %w", configPath, err)
		}
	}

	// Navigate to/create intermediate tables.
	node := raw
	for _, key := range keyPath[:len(keyPath)-1] {
		switch child := node[key].(type) {
		case map[string]any:
			node = child
		case nil:
			newChild := map[string]any{}
			node[key] = newChild
			node = newChild
		default:
			return fmt.Errorf("config key %q is not a table", key)
		}
	}

	// Append to the slice at the final key (skip if already present).
	finalKey := keyPath[len(keyPath)-1]
	var existing []any
	if v, ok := node[finalKey]; ok {
		existing, _ = v.([]any)
	}
	for _, v := range existing {
		if s, ok := v.(string); ok && s == value {
			return nil
		}
	}
	node[finalKey] = append(existing, value)

	// Ensure parent directory exists.
	if err := os.MkdirAll(filepath.Dir(configPath), 0o700); err != nil {
		return fmt.Errorf("creating config directory: %w", err)
	}

	data, err := toml.Marshal(raw)
	if err != nil {
		return fmt.Errorf("encoding config: %w", err)
	}
	return os.WriteFile(configPath, data, 0o600)
}

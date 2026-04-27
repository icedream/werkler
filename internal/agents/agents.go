// Package agents loads, validates, and saves custom agent definitions.
// Each agent is stored as a TOML file in the agents directory
// (~/.config/werkler/agents by default, overridden via $WERKLER_AGENTS_DIR).
//
// An agent is a named persona with custom instructions injected into the AI
// system prompt and an optional tool allowlist. When Tools is nil the agent
// has access to all tools; when it is an empty slice the agent has access to
// no tools (other than infra tools which are always available regardless).
package agents

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	toml "github.com/pelletier/go-toml/v2"
)

// nameRe is the set of valid agent name characters.
var nameRe = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// reservedNames are built-in tool names that must not be used as agent names.
var reservedNames = []string{
	"use_agent",
	"use_skill",
	"ask_user",
	"task_start",
	"task_complete",
	"connect_server",
}

// Agent represents a loaded agent definition.
type Agent struct {
	// Name is the machine-readable identifier.
	Name string `toml:"name"`
	// Description is a one-line human summary shown in listings and the system prompt.
	Description string `toml:"description"`
	// When describes the situations in which the AI should activate this agent.
	When string `toml:"when"`
	// Instructions is the extra system-prompt section injected when the agent is active.
	Instructions string `toml:"instructions"`
	// Tools is an optional allowlist of tool names. nil = unrestricted; non-nil empty = no tools.
	// Infrastructure tools (use_agent, ask_user, etc.) are always available regardless.
	// MCP server wildcard: "server.*" grants all tools from that server.
	Tools *[]string `toml:"tools,omitempty"`
}

// ToolList returns the Tools value as a plain slice.
// Returns nil when the agent is unrestricted.
func (a *Agent) ToolList() []string {
	if a.Tools == nil {
		return nil
	}
	return *a.Tools
}

// tomlAgent is the on-disk representation. We use a pointer for Tools so that
// absent and empty are distinguishable after decode, and we omit the key
// entirely when saving an unrestricted agent.
type tomlAgent struct {
	Name         string    `toml:"name"`
	Description  string    `toml:"description"`
	When         string    `toml:"when"`
	Instructions string    `toml:"instructions"`
	Tools        *[]string `toml:"tools,omitempty"`
}

// Validate checks that a is well-formed and returns an error describing the
// first problem found.
func Validate(a Agent) error {
	if a.Name == "" {
		return fmt.Errorf("agent name must not be empty")
	}
	if !nameRe.MatchString(a.Name) {
		return fmt.Errorf("agent name %q must match ^[a-zA-Z0-9_-]+$", a.Name)
	}
	if slices.Contains(reservedNames, a.Name) {
		return fmt.Errorf("agent name %q is reserved (conflicts with a built-in tool name)", a.Name)
	}
	if a.Description == "" {
		return fmt.Errorf("agent %q: description must not be empty", a.Name)
	}
	if a.When == "" {
		return fmt.Errorf("agent %q: when must not be empty", a.Name)
	}
	if a.Instructions == "" {
		return fmt.Errorf("agent %q: instructions must not be empty", a.Name)
	}
	return nil
}

// DefaultDir returns the default agents directory.
// It honours $WERKLER_AGENTS_DIR first, then $XDG_CONFIG_HOME/werkler/agents,
// and falls back to ~/.config/werkler/agents.
func DefaultDir() string {
	if d := os.Getenv("WERKLER_AGENTS_DIR"); d != "" {
		return d
	}
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		return filepath.Join(xdg, "werkler", "agents")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".config", "werkler", "agents")
}

// LoadDir scans dir for *.toml files and loads each as an Agent.
// Files that fail to parse, fail validation, or whose name field does not
// match the filename stem are skipped with a warning written to w.
// Duplicate names (in lexicographic file order) are skipped with a warning.
// If dir does not exist, an empty slice is returned without error.
func LoadDir(dir string, w io.Writer) ([]Agent, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading agents dir %s: %w", dir, err)
	}

	seen := make(map[string]bool)
	var out []Agent

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".toml") {
			continue
		}
		a, loadErr := loadFile(filepath.Join(dir, e.Name()))
		if loadErr != nil {
			_, _ = fmt.Fprintf(w, "warning: skipping agent %q: %v\n", e.Name(), loadErr)
			continue
		}
		stem := strings.TrimSuffix(e.Name(), ".toml")
		if a.Name != stem {
			_, _ = fmt.Fprintf(w, "warning: skipping agent %q: name field %q does not match filename stem %q\n",
				e.Name(), a.Name, stem)
			continue
		}
		if seen[a.Name] {
			_, _ = fmt.Fprintf(w, "warning: duplicate agent name %q in %s -- skipping\n", a.Name, e.Name())
			continue
		}
		seen[a.Name] = true
		out = append(out, a)
	}

	return out, nil
}

// loadFile reads and parses a single agent TOML file.
func loadFile(path string) (Agent, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Agent{}, err
	}
	var ta tomlAgent
	if err := toml.Unmarshal(data, &ta); err != nil {
		return Agent{}, fmt.Errorf("parsing %s: %w", filepath.Base(path), err)
	}
	a := Agent(ta)
	if err := Validate(a); err != nil {
		return Agent{}, err
	}
	return a, nil
}

// Save writes agent a to <dir>/<a.Name>.toml, creating dir if necessary.
// It returns an error if Validate(a) fails. It does not check for existing
// files; callers are responsible for presenting an overwrite confirmation.
func Save(dir string, a Agent) error {
	if err := Validate(a); err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating agents dir: %w", err)
	}
	ta := tomlAgent(a)
	data, err := toml.Marshal(ta)
	if err != nil {
		return fmt.Errorf("marshalling agent %q: %w", a.Name, err)
	}
	dest := filepath.Join(dir, a.Name+".toml")
	if err := os.WriteFile(dest, data, 0o644); err != nil {
		return fmt.Errorf("writing agent file: %w", err)
	}
	return nil
}

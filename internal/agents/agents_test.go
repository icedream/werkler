package agents

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValidate(t *testing.T) {
	good := Agent{
		Name:         "my-agent",
		Description:  "Does things",
		When:         "When things need doing",
		Instructions: "Be helpful.",
	}
	if err := Validate(good); err != nil {
		t.Fatalf("unexpected error for valid agent: %v", err)
	}

	tests := []struct {
		name    string
		mutate  func(*Agent)
		wantErr string
	}{
		{"empty name", func(a *Agent) { a.Name = "" }, "must not be empty"},
		{"bad name chars", func(a *Agent) { a.Name = "bad name!" }, "must match"},
		{"path separator", func(a *Agent) { a.Name = "a/b" }, "must match"},
		{"reserved use_agent", func(a *Agent) { a.Name = "use_agent" }, "reserved"},
		{"reserved ask_user", func(a *Agent) { a.Name = "ask_user" }, "reserved"},
		{"empty description", func(a *Agent) { a.Description = "" }, "description must not be empty"},
		{"empty when", func(a *Agent) { a.When = "" }, "when must not be empty"},
		{"empty instructions", func(a *Agent) { a.Instructions = "" }, "instructions must not be empty"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := good
			tc.mutate(&a)
			err := Validate(a)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestSaveAndLoad(t *testing.T) {
	dir := t.TempDir()

	unrestricted := Agent{
		Name:         "test-agent",
		Description:  "A test agent",
		When:         "During tests",
		Instructions: "Do test things.",
	}
	if err := Save(dir, unrestricted); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Verify file exists with correct name.
	if _, err := os.Stat(filepath.Join(dir, "test-agent.toml")); err != nil {
		t.Fatalf("expected test-agent.toml to exist: %v", err)
	}

	// Round-trip via LoadDir.
	loaded, err := LoadDir(dir, os.Stderr)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected 1 agent, got %d", len(loaded))
	}
	got := loaded[0]
	if got.Name != unrestricted.Name || got.Description != unrestricted.Description {
		t.Fatalf("round-trip mismatch: %+v", got)
	}
	// nil Tools means unrestricted.
	if got.Tools != nil {
		t.Fatalf("expected nil Tools for unrestricted agent, got %v", got.Tools)
	}
}

func TestSaveAndLoad_RestrictedTools(t *testing.T) {
	dir := t.TempDir()
	tools := []string{"file_read_multi", "file_list"}
	a := Agent{
		Name:         "restricted",
		Description:  "A restricted agent",
		When:         "When restricted",
		Instructions: "Limited tool access.",
		Tools:        &tools,
	}
	if err := Save(dir, a); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadDir(dir, os.Stderr)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected 1 agent, got %d", len(loaded))
	}
	tl := loaded[0].ToolList()
	if len(tl) != 2 || tl[0] != "file_read" || tl[1] != "file_list" {
		t.Fatalf("unexpected tool list: %v", tl)
	}
}

func TestSaveAndLoad_EmptyTools(t *testing.T) {
	dir := t.TempDir()
	empty := []string{}
	a := Agent{
		Name:         "no-tools",
		Description:  "No tools agent",
		When:         "When no tools needed",
		Instructions: "Conversation only.",
		Tools:        &empty,
	}
	if err := Save(dir, a); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := LoadDir(dir, os.Stderr)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected 1 agent, got %d", len(loaded))
	}
	tl := loaded[0].ToolList()
	// Non-nil but empty -- means "no tools", distinct from nil (unrestricted).
	if tl == nil {
		t.Fatal("expected non-nil empty tool list, got nil")
	}
	if len(tl) != 0 {
		t.Fatalf("expected empty tool list, got %v", tl)
	}
}

func TestLoadDir_SkipsMismatchedName(t *testing.T) {
	dir := t.TempDir()
	// Write a TOML where the name field doesn't match the filename.
	content := `name = "other-name"
description = "desc"
when = "when"
instructions = "instr"
`
	if err := os.WriteFile(filepath.Join(dir, "my-agent.toml"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadDir(dir, os.Stderr)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if len(loaded) != 0 {
		t.Fatalf("expected 0 agents (name mismatch should be skipped), got %d", len(loaded))
	}
}

func TestLoadDir_NonExistent(t *testing.T) {
	loaded, err := LoadDir("/does/not/exist", os.Stderr)
	if err != nil {
		t.Fatalf("expected nil error for non-existent dir, got: %v", err)
	}
	if len(loaded) != 0 {
		t.Fatalf("expected empty slice, got %d agents", len(loaded))
	}
}

func TestDefaultDir(t *testing.T) {
	t.Setenv("WERKLER_AGENTS_DIR", "/custom/agents")
	t.Setenv("XDG_CONFIG_HOME", "")
	d := DefaultDir()
	if d != "/custom/agents" {
		t.Fatalf("expected /custom/agents, got %s", d)
	}
}

package tools

import (
	"context"
	"testing"

	"github.com/icedream/werkler/internal/agents"
)

// TestIsInfraToolName verifies that infrastructure tools are correctly identified.
func TestIsInfraToolName(t *testing.T) {
	infra := []string{"use_agent", "use_skill", "ask_user", "task_start", "task_complete", "connect_server"}
	for _, name := range infra {
		if !IsInfraToolName(name) {
			t.Errorf("expected %q to be an infra tool", name)
		}
	}
	notInfra := []string{"file_read_multi", "file_write", "process_start", "calculate"}
	for _, name := range notInfra {
		if IsInfraToolName(name) {
			t.Errorf("expected %q NOT to be an infra tool", name)
		}
	}
}

// TestSetToolFilter_NilMeansUnrestricted verifies that nil filter allows all tools.
func TestSetToolFilter_NilMeansUnrestricted(t *testing.T) {
	m := newTestManager()
	m.SetToolFilter(nil)
	if !m.isAllowed("file_read_multi") {
		t.Error("expected file_read_multi to be allowed with nil filter")
	}
	if !m.isAllowed("use_agent") {
		t.Error("expected use_agent (infra) to always be allowed")
	}
}

// TestSetToolFilter_AllowlistFilters verifies non-allowlisted tools are blocked.
func TestSetToolFilter_AllowlistFilters(t *testing.T) {
	m := newTestManager()
	m.SetToolFilter([]string{"file_read_multi", "file_list"})

	if !m.isAllowed("file_read_multi") {
		t.Error("file_read_multi should be allowed")
	}
	if !m.isAllowed("file_list") {
		t.Error("file_list should be allowed")
	}
	if m.isAllowed("file_write") {
		t.Error("file_write should NOT be allowed")
	}
	// Infra tools bypass the allowlist.
	if !m.isAllowed("use_agent") {
		t.Error("use_agent (infra) must always be allowed")
	}
	if !m.isAllowed("ask_user") {
		t.Error("ask_user (infra) must always be allowed")
	}
}

// TestSetToolFilter_EmptyMeansNoTools verifies that an empty (non-nil) filter blocks all non-infra tools.
func TestSetToolFilter_EmptyMeansNoTools(t *testing.T) {
	m := newTestManager()
	m.SetToolFilter([]string{})

	if m.isAllowed("file_read_multi") {
		t.Error("file_read_multi should NOT be allowed with empty filter")
	}
	// Infra tools still allowed.
	if !m.isAllowed("use_agent") {
		t.Error("use_agent (infra) must always be allowed")
	}
}

// TestSetToolFilter_Wildcard verifies "server.*" wildcard matching.
func TestSetToolFilter_Wildcard(t *testing.T) {
	m := newTestManager()
	m.SetToolFilter([]string{"fs.*"})

	if !m.isAllowed("fs.read_file") {
		t.Error("fs.read_file should be allowed via wildcard")
	}
	if !m.isAllowed("fs.list_dir") {
		t.Error("fs.list_dir should be allowed via wildcard")
	}
	if m.isAllowed("slack.post") {
		t.Error("slack.post should NOT be allowed")
	}
}

// TestHandleUseAgent_ActivatesCallback verifies that use_agent calls the activation callback.
func TestHandleUseAgent_ActivatesCallback(t *testing.T) {
	m := newTestManager()

	activated := ""
	m.SetAgentActivateNotify(func(name string) {
		activated = name
	})

	testAgents := []agents.Agent{
		{Name: "go-reviewer", Description: "desc", When: "when", Instructions: "instr"},
	}
	m.SetAgents(testAgents)

	result, err := m.handleUseAgent(context.Background(), map[string]any{"name": "go-reviewer"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if activated != "go-reviewer" {
		t.Errorf("expected callback to be called with 'go-reviewer', got %q", activated)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
}

// TestHandleUseAgent_UnknownAgent returns graceful error string.
func TestHandleUseAgent_UnknownAgent(t *testing.T) {
	m := newTestManager()
	m.SetAgents([]agents.Agent{})

	result, err := m.handleUseAgent(context.Background(), map[string]any{"name": "unknown"})
	if err != nil {
		t.Fatalf("unexpected hard error: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty error message string")
	}
}

// newTestManager creates a bare-bones Manager suitable for unit tests.
func newTestManager() *Manager {
	m := &Manager{}
	m.builtins = m.makeBuiltins()
	return m
}

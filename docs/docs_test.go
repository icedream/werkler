package docs_test

import (
	"strings"
	"testing"

	"github.com/icedream/werkler/docs"
)

func TestAll_ReturnsMultipleSections(t *testing.T) {
	sections := docs.All()
	if len(sections) == 0 {
		t.Fatal("All() returned no sections")
	}
	// Every section should have a file name.
	for _, s := range sections {
		if s.File == "" {
			t.Errorf("section with heading %q has empty File", s.Heading)
		}
	}
}

func TestSearch_KeybindingsTopic(t *testing.T) {
	result := docs.Search("keybindings")
	if result == "" {
		t.Fatal("Search returned empty string")
	}
	lower := strings.ToLower(result)
	if !strings.Contains(lower, "key") && !strings.Contains(lower, "tui") {
		t.Errorf("expected keybinding-related content, got: %s", result[:min(200, len(result))])
	}
}

func TestSearch_ConfigurationTopic(t *testing.T) {
	result := docs.Search("configuration providers api_key")
	lower := strings.ToLower(result)
	if !strings.Contains(lower, "provider") && !strings.Contains(lower, "config") {
		t.Errorf("expected configuration content, got: %s", result[:min(200, len(result))])
	}
}

func TestSearch_AutopilotTopic(t *testing.T) {
	result := docs.Search("autopilot cycle limit")
	lower := strings.ToLower(result)
	if !strings.Contains(lower, "autopilot") && !strings.Contains(lower, "cycle") {
		t.Errorf("expected autopilot content, got: %s", result[:min(200, len(result))])
	}
}

func TestSearch_NoMatchReturnsTOC(t *testing.T) {
	result := docs.Search("xyzzy frobnicator 12345qwerty")
	if !strings.Contains(result, "Available documentation sections") {
		t.Errorf("expected TOC fallback for no-match query, got: %s", result[:min(300, len(result))])
	}
}

func TestSearch_EmptyQueryReturnsTOC(t *testing.T) {
	result := docs.Search("")
	if !strings.Contains(result, "Available documentation sections") {
		t.Errorf("expected TOC for empty query, got: %s", result[:min(300, len(result))])
	}
}

func TestSearch_ResultCappedAt4000Chars(t *testing.T) {
	// Use a very broad query that matches many sections.
	result := docs.Search("the a to in of")
	if len(result) > 4100 { // small slack for "… (truncated)" suffix
		t.Errorf("result length %d exceeds 4000 char cap", len(result))
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

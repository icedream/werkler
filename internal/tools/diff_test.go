package tools

import (
	"strings"
	"testing"
)

func TestComputeUnifiedDiff_Identical(t *testing.T) {
	if got := ComputeUnifiedDiff("hello\n", "hello\n", "f.txt"); got != "" {
		t.Fatalf("expected empty diff for identical content, got %q", got)
	}
}

func TestComputeUnifiedDiff_AddLines(t *testing.T) {
	old := "line1\nline2\n"
	new := "line1\nline2\nline3\n"
	diff := ComputeUnifiedDiff(old, new, "f.txt")
	if !strings.Contains(diff, "+line3") {
		t.Fatalf("expected +line3 in diff, got:\n%s", diff)
	}
	if !strings.Contains(diff, "--- a/f.txt") {
		t.Fatalf("expected header in diff, got:\n%s", diff)
	}
}

func TestComputeUnifiedDiff_RemoveLines(t *testing.T) {
	old := "a\nb\nc\n"
	new := "a\nc\n"
	diff := ComputeUnifiedDiff(old, new, "f.txt")
	if !strings.Contains(diff, "-b") {
		t.Fatalf("expected -b in diff, got:\n%s", diff)
	}
}

func TestComputeUnifiedDiff_ReplaceLines(t *testing.T) {
	old := "foo\nbar\nbaz\n"
	new := "foo\nQUX\nbaz\n"
	diff := ComputeUnifiedDiff(old, new, "f.txt")
	if !strings.Contains(diff, "-bar") || !strings.Contains(diff, "+QUX") {
		t.Fatalf("expected -bar and +QUX in diff, got:\n%s", diff)
	}
}

func TestComputeUnifiedDiff_Binary(t *testing.T) {
	binary := "he\x00llo"
	if got := ComputeUnifiedDiff(binary, "hello", "f.bin"); got != "" {
		t.Fatalf("expected empty diff for binary old content, got %q", got)
	}
	if got := ComputeUnifiedDiff("hello", binary, "f.bin"); got != "" {
		t.Fatalf("expected empty diff for binary new content, got %q", got)
	}
}

func TestComputeUnifiedDiff_NewFile(t *testing.T) {
	diff := ComputeUnifiedDiff("", "line1\nline2\n", "new.txt")
	if !strings.Contains(diff, "+line1") || !strings.Contains(diff, "+line2") {
		t.Fatalf("expected both added lines, got:\n%s", diff)
	}
}

func TestComputeUnifiedDiff_DeletedFile(t *testing.T) {
	diff := ComputeUnifiedDiff("old\ncontent\n", "", "del.txt")
	if !strings.Contains(diff, "-old") || !strings.Contains(diff, "-content") {
		t.Fatalf("expected removed lines, got:\n%s", diff)
	}
}

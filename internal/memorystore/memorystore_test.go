package memorystore

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// newTestStore creates a MemoryStore using a temp directory as the storage root,
// bypassing the real ~/.config/werkler/memory path.
func newTestStore(t *testing.T) *MemoryStore {
	t.Helper()
	dir := t.TempDir()
	s := &MemoryStore{
		dir:    dir,
		cached: make(map[string]string),
	}
	_ = s.loadAll()
	return s
}

// --- validateSlug ---

func TestValidateSlug(t *testing.T) {
	cases := []struct {
		name  string
		valid bool
	}{
		{"general", true},
		{"api-notes", true},
		{"a", true},
		{"abc123", true},
		{"a-b-c", true},
		{"", false},
		{"-leading", false},
		{"has space", false},
		{"UPPER", false},
		{"under_score", false},
		{strings.Repeat("a", 64), true},
		{strings.Repeat("a", 65), false},
		{"has/slash", false},
		{"has.dot", false},
	}
	for _, tc := range cases {
		err := validateSlug(tc.name)
		if tc.valid && err != nil {
			t.Errorf("validateSlug(%q) = %v, want nil", tc.name, err)
		}
		if !tc.valid && err == nil {
			t.Errorf("validateSlug(%q) = nil, want error", tc.name)
		}
	}
}

// --- WriteNamed / ReadNamed ---

func TestWriteRead(t *testing.T) {
	s := newTestStore(t)
	if err := s.WriteNamed("general", "hello world"); err != nil {
		t.Fatal(err)
	}
	got, err := s.ReadNamed("general")
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello world" {
		t.Errorf("got %q, want %q", got, "hello world")
	}
}

func TestWriteTrimsSpace(t *testing.T) {
	s := newTestStore(t)
	_ = s.WriteNamed("general", "  trimmed  ")
	got, _ := s.ReadNamed("general")
	if got != "trimmed" {
		t.Errorf("got %q, want %q", got, "trimmed")
	}
}

func TestWriteRejectsTooLarge(t *testing.T) {
	s := newTestStore(t)
	big := strings.Repeat("x", MaxBytesPerFile+1)
	err := s.WriteNamed("general", big)
	if err == nil {
		t.Fatal("expected error for oversized content, got nil")
	}
}

func TestWriteRejectsInvalidSlug(t *testing.T) {
	s := newTestStore(t)
	if err := s.WriteNamed("Bad Name!", "content"); err == nil {
		t.Fatal("expected slug validation error")
	}
}

func TestWriteRejectsTooManyFiles(t *testing.T) {
	s := newTestStore(t)
	for i := range MaxFiles {
		_ = s.WriteNamed("m"+strings.Repeat("a", i+1), "content")
	}
	// One more should fail.
	err := s.WriteNamed("overflow", "content")
	if err == nil {
		t.Fatal("expected too-many-files error, got nil")
	}
}

func TestReadNotExist(t *testing.T) {
	s := newTestStore(t)
	got, err := s.ReadNamed("nosuchfile")
	if err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Errorf("expected empty string, got %q", got)
	}
}

// --- DeleteNamed ---

func TestDeleteNamed(t *testing.T) {
	s := newTestStore(t)
	_ = s.WriteNamed("notes", "some notes")
	if err := s.DeleteNamed("notes"); err != nil {
		t.Fatal(err)
	}
	// Should be gone from cache.
	if s.Exists() {
		t.Error("Exists() should be false after deleting the only entry")
	}
	// File should be gone from disk.
	if _, err := os.Stat(filepath.Join(s.dir, "notes.md")); !os.IsNotExist(err) {
		t.Error("file should not exist on disk after delete")
	}
}

func TestDeleteNotExist(t *testing.T) {
	s := newTestStore(t)
	if err := s.DeleteNamed("nosuchfile"); err != nil {
		t.Errorf("DeleteNamed on non-existent file should be nil, got %v", err)
	}
}

// --- List / CachedAll ---

func TestListSorted(t *testing.T) {
	s := newTestStore(t)
	_ = s.WriteNamed("zzz", "c")
	_ = s.WriteNamed("aaa", "a")
	_ = s.WriteNamed("mmm", "b")

	list := s.List()
	if len(list) != 3 {
		t.Fatalf("expected 3, got %d", len(list))
	}
	if list[0].Name != "aaa" || list[1].Name != "mmm" || list[2].Name != "zzz" {
		t.Errorf("unexpected order: %v", list)
	}
}

func TestCachedAllSorted(t *testing.T) {
	s := newTestStore(t)
	_ = s.WriteNamed("z", "last")
	_ = s.WriteNamed("a", "first")

	all := s.CachedAll()
	if len(all) != 2 || all[0].Name != "a" || all[1].Name != "z" {
		t.Errorf("unexpected CachedAll: %v", all)
	}
}

// --- Exists ---

func TestExists(t *testing.T) {
	s := newTestStore(t)
	if s.Exists() {
		t.Error("new store should not exist")
	}
	_ = s.WriteNamed("general", "content")
	if !s.Exists() {
		t.Error("store should exist after write")
	}
}

// --- BuildInjectionSection ---

func TestBuildInjectionSectionEmpty(t *testing.T) {
	s := newTestStore(t)
	if got := s.BuildInjectionSection(); got != "" {
		t.Errorf("expected empty string, got %q", got)
	}
}

func TestBuildInjectionSectionBasic(t *testing.T) {
	s := newTestStore(t)
	_ = s.WriteNamed("general", "key facts")
	sec := s.BuildInjectionSection()
	if !strings.Contains(sec, "## Project memory") {
		t.Error("missing header")
	}
	if !strings.Contains(sec, "### general") {
		t.Error("missing section header")
	}
	if !strings.Contains(sec, "key facts") {
		t.Error("missing content")
	}
}

func TestBuildInjectionSectionBudgetOverflow(t *testing.T) {
	s := newTestStore(t)
	// Write a memory larger than InjectBudget.
	big := strings.Repeat("x", InjectBudget+1)
	// Can't use WriteNamed directly since it enforces MaxBytesPerFile;
	// write directly to disk to simulate a large file.
	path := filepath.Join(s.dir, "big.md")
	_ = os.WriteFile(path, []byte(big), 0600)
	s.mu.Lock()
	s.cached["big"] = big
	s.mu.Unlock()

	sec := s.BuildInjectionSection()
	if strings.Contains(sec, big) {
		t.Error("oversized content should not appear verbatim in section")
	}
	if !strings.Contains(sec, "big") {
		t.Error("overflow memory name should still appear in section")
	}
	if !strings.Contains(sec, "memory_read") {
		t.Error("overflow section should suggest memory_read")
	}
}

// --- Migration ---

func TestMigrateLegacyNoFile(t *testing.T) {
	dir := t.TempDir()
	legacyPath := filepath.Join(dir, "abc.md")
	targetDir := filepath.Join(dir, "abc")
	// No legacy file: should be a no-op.
	if err := migrateLegacy(legacyPath, targetDir); err != nil {
		t.Fatal(err)
	}
}

func TestMigrateLegacyMoves(t *testing.T) {
	dir := t.TempDir()
	legacyPath := filepath.Join(dir, "abc.md")
	targetDir := filepath.Join(dir, "abc")
	_ = os.WriteFile(legacyPath, []byte("legacy content"), 0600)

	if err := migrateLegacy(legacyPath, targetDir); err != nil {
		t.Fatal(err)
	}
	// Legacy file should be gone.
	if _, err := os.Stat(legacyPath); !os.IsNotExist(err) {
		t.Error("legacy file should have been removed")
	}
	// general.md should contain the content.
	data, err := os.ReadFile(filepath.Join(targetDir, "general.md"))
	if err != nil || string(data) != "legacy content" {
		t.Errorf("unexpected general.md content: %v / %s", err, data)
	}
}

func TestMigrateLegacyGeneralAlreadyExists(t *testing.T) {
	dir := t.TempDir()
	legacyPath := filepath.Join(dir, "abc.md")
	targetDir := filepath.Join(dir, "abc")
	_ = os.MkdirAll(targetDir, 0700)
	_ = os.WriteFile(legacyPath, []byte("legacy"), 0600)
	_ = os.WriteFile(filepath.Join(targetDir, "general.md"), []byte("existing"), 0600)

	if err := migrateLegacy(legacyPath, targetDir); err != nil {
		t.Fatal(err)
	}
	// general.md should be unchanged.
	data, _ := os.ReadFile(filepath.Join(targetDir, "general.md"))
	if string(data) != "existing" {
		t.Errorf("general.md should not be overwritten; got %q", data)
	}
	// Legacy file should have been removed.
	if _, err := os.Stat(legacyPath); !os.IsNotExist(err) {
		t.Error("legacy file should have been removed")
	}
}

func TestMigrateLegacyIdempotent(t *testing.T) {
	dir := t.TempDir()
	legacyPath := filepath.Join(dir, "abc.md")
	targetDir := filepath.Join(dir, "abc")
	_ = os.WriteFile(legacyPath, []byte("content"), 0600)

	// Call twice: first call migrates, second call sees no legacy file.
	if err := migrateLegacy(legacyPath, targetDir); err != nil {
		t.Fatal(err)
	}
	if err := migrateLegacy(legacyPath, targetDir); err != nil {
		t.Fatal("second migration call should be a no-op, got:", err)
	}
}

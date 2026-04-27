package memorystore

import (
	"crypto/sha256"
	"fmt"
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

// --- loadAncestors / ancestor injection ---

// writeHashedDir creates a hashed memory directory for dirPath under baseDir
// and writes the given files into it. Returns the created directory path.
func writeHashedDir(t *testing.T, baseDir, dirPath string, files map[string]string) string {
	t.Helper()
	hash := sha256.Sum256([]byte(dirPath))
	dir := filepath.Join(baseDir, fmt.Sprintf("%x", hash))
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name+".md"), []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestLoadAncestors_NoParentMemories(t *testing.T) {
	baseDir := t.TempDir()
	cwd := "/a/b/c/d"
	ancs := loadAncestors(cwd, baseDir)
	if len(ancs) != 0 {
		t.Errorf("expected no ancestors, got %d", len(ancs))
	}
}

func TestLoadAncestors_ImmediateParent(t *testing.T) {
	baseDir := t.TempDir()
	cwd := "/a/b/c/d"
	parent := "/a/b/c"
	writeHashedDir(t, baseDir, parent, map[string]string{"general": "parent notes"})

	ancs := loadAncestors(cwd, baseDir)
	if len(ancs) != 1 {
		t.Fatalf("expected 1 ancestor, got %d", len(ancs))
	}
	if ancs[0].RelPath != ".." {
		t.Errorf("expected RelPath '..', got %q", ancs[0].RelPath)
	}
	if len(ancs[0].Entries) != 1 || ancs[0].Entries[0].Name != "general" {
		t.Errorf("unexpected entries: %v", ancs[0].Entries)
	}
}

func TestLoadAncestors_MultipleParents(t *testing.T) {
	baseDir := t.TempDir()
	cwd := "/a/b/c/d"
	writeHashedDir(t, baseDir, "/a/b/c", map[string]string{"close": "close parent"})
	writeHashedDir(t, baseDir, "/a/b", map[string]string{"far": "far parent"})
	// /a has no memory — should be skipped.

	ancs := loadAncestors(cwd, baseDir)
	if len(ancs) != 2 {
		t.Fatalf("expected 2 ancestors, got %d: %v", len(ancs), ancs)
	}
	// Most immediate parent first.
	if ancs[0].RelPath != ".." {
		t.Errorf("first ancestor RelPath = %q, want '..'", ancs[0].RelPath)
	}
	if ancs[1].RelPath != "../.." {
		t.Errorf("second ancestor RelPath = %q, want '../..'", ancs[1].RelPath)
	}
}

func TestBuildInjectionSection_WithAncestors(t *testing.T) {
	s := newTestStore(t)
	_ = s.WriteNamed("local", "local notes")
	s.ancestors = []AncestorMemory{
		{RelPath: "..", Entries: []NamedContent{{Name: "shared", Content: "shared notes"}}},
	}

	sec := s.BuildInjectionSection()

	if !strings.Contains(sec, "## Project memory") {
		t.Error("missing CWD memory header")
	}
	if !strings.Contains(sec, "local notes") {
		t.Error("missing local memory content")
	}
	if !strings.Contains(sec, "## Parent directory memory (..)") {
		t.Error("missing ancestor memory header")
	}
	if !strings.Contains(sec, "shared notes") {
		t.Error("missing ancestor memory content")
	}
}

func TestBuildInjectionSection_AncestorOnlyNoLocal(t *testing.T) {
	s := newTestStore(t)
	s.ancestors = []AncestorMemory{
		{RelPath: "..", Entries: []NamedContent{{Name: "root", Content: "root context"}}},
	}

	sec := s.BuildInjectionSection()
	if strings.Contains(sec, "## Project memory") {
		t.Error("should not emit CWD header when there are no local memories")
	}
	if !strings.Contains(sec, "## Parent directory memory (..)") {
		t.Error("missing ancestor header")
	}
	if !strings.Contains(sec, "root context") {
		t.Error("missing ancestor content")
	}
}

func TestBuildInjectionSection_EmptyWithNoAncestors(t *testing.T) {
	s := newTestStore(t)
	if got := s.BuildInjectionSection(); got != "" {
		t.Errorf("expected empty string, got %q", got)
	}
}

// --- Promote ---

// newTestStoreForCWD creates a MemoryStore rooted at a specific absolute path
// within baseDir, simulating what New() does without touching the real config dir.
func newTestStoreForCWD(t *testing.T, baseDir, absPath string) *MemoryStore {
	t.Helper()
	hash := sha256.Sum256([]byte(absPath))
	dir := filepath.Join(baseDir, fmt.Sprintf("%x", hash))
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	s := &MemoryStore{
		dir:     dir,
		abs:     absPath,
		baseDir: baseDir,
		cached:  make(map[string]string),
	}
	_ = s.loadAll()
	return s
}

func TestPromote_MovesToParent(t *testing.T) {
	baseDir := t.TempDir()
	childPath := "/workspace/project/subdir"
	parentPath := "/workspace/project"

	s := newTestStoreForCWD(t, baseDir, childPath)
	if err := s.WriteNamed("notes", "important stuff"); err != nil {
		t.Fatal(err)
	}

	if err := s.Promote("notes", ".."); err != nil {
		t.Fatalf("Promote failed: %v", err)
	}

	// Memory should be gone from child store.
	content, _ := s.ReadNamed("notes")
	if content != "" {
		t.Error("memory should be removed from child store after promote")
	}

	// Memory should exist in parent's hash directory.
	parentEntries, _ := readDirMemories(func() string {
		hash := sha256.Sum256([]byte(parentPath))
		return filepath.Join(baseDir, fmt.Sprintf("%x", hash))
	}())
	if len(parentEntries) != 1 || parentEntries[0].Name != "notes" || parentEntries[0].Content != "important stuff" {
		t.Errorf("unexpected parent entries: %v", parentEntries)
	}
}

func TestPromote_RejectsCurrentDir(t *testing.T) {
	baseDir := t.TempDir()
	s := newTestStoreForCWD(t, baseDir, "/workspace/project")
	_ = s.WriteNamed("notes", "content")

	if err := s.Promote("notes", "."); err == nil {
		t.Error("expected error when promoting to current directory")
	}
}

func TestPromote_RejectsChildDir(t *testing.T) {
	baseDir := t.TempDir()
	s := newTestStoreForCWD(t, baseDir, "/workspace/project")
	_ = s.WriteNamed("notes", "content")

	if err := s.Promote("notes", "subdir"); err == nil {
		t.Error("expected error when promoting to a child directory")
	}
}

func TestPromote_RejectsMissingMemory(t *testing.T) {
	baseDir := t.TempDir()
	s := newTestStoreForCWD(t, baseDir, "/workspace/project/subdir")

	if err := s.Promote("doesnotexist", ".."); err == nil {
		t.Error("expected error when promoting non-existent memory")
	}
}

func TestAncestorPaths(t *testing.T) {
	baseDir := t.TempDir()
	cwd := "/a/b/c/d"
	writeHashedDir(t, baseDir, "/a/b/c", map[string]string{"x": "content"})
	writeHashedDir(t, baseDir, "/a/b", map[string]string{"y": "content"})

	ancs := loadAncestors(cwd, baseDir)
	s := &MemoryStore{ancestors: ancs}
	paths := s.AncestorPaths()

	if len(paths) != 2 || paths[0] != ".." || paths[1] != "../.." {
		t.Errorf("unexpected AncestorPaths: %v", paths)
	}
}

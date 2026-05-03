package tools

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newFileDeleteTestManager() *Manager {
	m := &Manager{}
	m.builtins = m.makeBuiltins()
	return m
}

func TestHandleFileDelete_RemovesSymlinkNotTarget(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target.txt")
	link := filepath.Join(dir, "link.txt")

	require.NoError(t, os.WriteFile(target, []byte("hello"), 0o644))
	require.NoError(t, os.Symlink(target, link))

	m := newFileDeleteTestManager()
	_, err := m.handleFileDelete(t.Context(), map[string]any{"path": link})
	require.NoError(t, err)

	// The symlink must be gone.
	_, lerr := os.Lstat(link)
	assert.True(t, os.IsNotExist(lerr), "symlink should be removed")

	// The target must still exist.
	_, terr := os.Stat(target)
	assert.NoError(t, terr, "target file must not be deleted")
}

func TestHandleFileDelete_RemovesRegularFile(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "file.txt")
	require.NoError(t, os.WriteFile(f, []byte("data"), 0o644))

	m := newFileDeleteTestManager()
	_, err := m.handleFileDelete(t.Context(), map[string]any{"path": f})
	require.NoError(t, err)

	_, ferr := os.Lstat(f)
	assert.True(t, os.IsNotExist(ferr), "file should be removed")
}

func TestHandleFileDelete_RejectsDirectory(t *testing.T) {
	dir := t.TempDir()
	sub := filepath.Join(dir, "sub")
	require.NoError(t, os.Mkdir(sub, 0o755))

	m := newFileDeleteTestManager()
	_, err := m.handleFileDelete(t.Context(), map[string]any{"path": sub})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "directory")
}

func TestHandleFileDelete_SymlinkToDir_RemovesSymlinkNotDir(t *testing.T) {
	dir := t.TempDir()
	realDir := filepath.Join(dir, "realdir")
	link := filepath.Join(dir, "dirlink")
	require.NoError(t, os.Mkdir(realDir, 0o755))
	require.NoError(t, os.Symlink(realDir, link))

	m := newFileDeleteTestManager()
	_, err := m.handleFileDelete(t.Context(), map[string]any{"path": link})
	// A symlink-to-directory must be removable (the old code mistakenly
	// rejected it with "is a directory").
	require.NoError(t, err)

	_, lerr := os.Lstat(link)
	assert.True(t, os.IsNotExist(lerr), "symlink should be removed")

	_, derr := os.Stat(realDir)
	assert.NoError(t, derr, "real directory must still exist")
}

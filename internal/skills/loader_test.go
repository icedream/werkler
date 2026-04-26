package skills_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/icedream/werkler/internal/skills"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeSkill(t *testing.T, dir, name, content string) string {
	t.Helper()
	d := filepath.Join(dir, name)
	require.NoError(t, os.MkdirAll(d, 0o700))
	require.NoError(t, os.WriteFile(filepath.Join(d, "SKILL.md"), []byte(content), 0o600))
	return d
}

func TestLoad_ValidSkill(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, dir, "go-guidelines", "---\nname: go-guidelines\ndescription: Go best practices\n---\n# Go Guidelines\n\nBe good.\n")

	s, err := skills.Load(filepath.Join(dir, "go-guidelines"))
	require.NoError(t, err)
	assert.Equal(t, "go-guidelines", s.Name)
	assert.Equal(t, "Go best practices", s.Description)
	assert.Contains(t, s.Content, "# Go Guidelines")
	assert.Equal(t, filepath.Join(dir, "go-guidelines"), s.Dir)
}

func TestLoad_MissingFrontmatter(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, dir, "bad", "# No frontmatter here\n")

	_, err := skills.Load(filepath.Join(dir, "bad"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "front matter")
}

func TestLoad_MissingName(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, dir, "noname", "---\ndescription: some desc\n---\ncontent\n")

	_, err := skills.Load(filepath.Join(dir, "noname"))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "name")
}

func TestLoad_NoSKILLmd(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.Mkdir(filepath.Join(dir, "empty"), 0o700))

	_, err := skills.Load(filepath.Join(dir, "empty"))
	require.Error(t, err)
}

func TestLoad_ShellExpansion(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, dir, "echo-skill", "---\nname: echo-skill\ndescription: test\n---\n!`echo hello world`\nstatic line\n")

	s, err := skills.Load(filepath.Join(dir, "echo-skill"))
	require.NoError(t, err)
	assert.Contains(t, s.Content, "hello world")
	assert.Contains(t, s.Content, "static line")
	// The !`...` line itself should not appear
	assert.NotContains(t, s.Content, "!`echo")
}

func TestLoad_ShellExpansionFailure(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, dir, "fail-skill", "---\nname: fail-skill\ndescription: test\n---\n!`exit 1`\nafter\n")

	s, err := skills.Load(filepath.Join(dir, "fail-skill"))
	require.NoError(t, err)
	assert.Contains(t, s.Content, "[skill command failed:")
	assert.Contains(t, s.Content, "after")
}

func TestLoadDir_HappyPath(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, dir, "skill-a", "---\nname: skill-a\ndescription: alpha\n---\nalpha content\n")
	writeSkill(t, dir, "skill-b", "---\nname: skill-b\ndescription: beta\n---\nbeta content\n")
	// non-dir file should be ignored
	require.NoError(t, os.WriteFile(filepath.Join(dir, "README.md"), []byte("ignore me"), 0o600))

	var warn strings.Builder
	loaded, err := skills.LoadDir(dir, &warn)
	require.NoError(t, err)
	assert.Len(t, loaded, 2)
	assert.Empty(t, warn.String())
}

func TestLoadDir_NonExistent(t *testing.T) {
	loaded, err := skills.LoadDir(filepath.Join(t.TempDir(), "no-such-dir"), os.Stderr)
	require.NoError(t, err)
	assert.Empty(t, loaded)
}

func TestLoadDir_BadSkillSkipped(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, dir, "good", "---\nname: good\ndescription: ok\n---\ncontent\n")
	writeSkill(t, dir, "bad", "no frontmatter")

	var warn strings.Builder
	loaded, err := skills.LoadDir(dir, &warn)
	require.NoError(t, err)
	assert.Len(t, loaded, 1)
	assert.Equal(t, "good", loaded[0].Name)
	assert.Contains(t, warn.String(), "bad")
}

func TestLoadDir_DuplicateNameSkipped(t *testing.T) {
	dir := t.TempDir()
	writeSkill(t, dir, "dir-a", "---\nname: clash\ndescription: first\n---\nfirst\n")
	writeSkill(t, dir, "dir-b", "---\nname: clash\ndescription: second\n---\nsecond\n")

	var warn strings.Builder
	loaded, err := skills.LoadDir(dir, &warn)
	require.NoError(t, err)
	assert.Len(t, loaded, 1)
	assert.Contains(t, warn.String(), "clash")
}

func TestExpandTilde(t *testing.T) {
	home, err := os.UserHomeDir()
	require.NoError(t, err)

	assert.Equal(t, home, skills.ExpandTilde("~"))
	assert.Equal(t, filepath.Join(home, ".agents", "skills"), skills.ExpandTilde("~/.agents/skills"))
	assert.Equal(t, "/absolute/path", skills.ExpandTilde("/absolute/path"))
	assert.Equal(t, "relative/path", skills.ExpandTilde("relative/path"))
}

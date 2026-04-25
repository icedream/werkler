package config

import (
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInsertIntoText_MultilineArray_InsertsBeforeClose(t *testing.T) {
	input := `# top comment
[mcp]
# Tools allowed
auto_approve_tools = [
  "git__log",
]
`
	result, err := insertIntoText(input, "mcp", "auto_approve_tools", "fs__read_file")
	require.NoError(t, err)
	assert.Contains(t, result, `"fs__read_file",`)
	assert.Contains(t, result, `"git__log",`)
	assert.Contains(t, result, "# top comment")
	assert.Contains(t, result, "# Tools allowed")
	// New entry should appear before the closing bracket.
	idx1 := strings.Index(result, `"fs__read_file"`)
	idx2 := strings.LastIndex(result, "]")
	assert.Less(t, idx1, idx2, "new entry should be before closing ]")
}

func TestInsertIntoText_SingleLineArray_Expands(t *testing.T) {
	input := `[mcp]
auto_approve_tools = ["git__log"]
`
	result, err := insertIntoText(input, "mcp", "auto_approve_tools", "fs__read_file")
	require.NoError(t, err)
	assert.Contains(t, result, `"fs__read_file"`)
	assert.Contains(t, result, `"git__log"`)
}

func TestInsertIntoText_KeyAbsent_SectionPresent(t *testing.T) {
	input := `# header
[mcp]
# existing key
some_other = "val"

[ai]
model = "gpt-4o"
`
	result, err := insertIntoText(input, "mcp", "auto_approve_tools", "my_tool")
	require.NoError(t, err)
	assert.Contains(t, result, "auto_approve_tools")
	assert.Contains(t, result, `"my_tool"`)
	assert.Contains(t, result, "# header")
	// New key should be in [mcp], not after [ai].
	mcpIdx := strings.Index(result, "[mcp]")
	aiIdx := strings.Index(result, "[ai]")
	toolIdx := strings.Index(result, "auto_approve_tools")
	assert.Greater(t, toolIdx, mcpIdx)
	assert.Less(t, toolIdx, aiIdx)
}

func TestInsertIntoText_SectionAbsent_AppendsToEnd(t *testing.T) {
	input := `[ai]
model = "gpt-4o"
`
	result, err := insertIntoText(input, "mcp", "auto_approve_tools", "my_tool")
	require.NoError(t, err)
	assert.Contains(t, result, "[mcp]")
	assert.Contains(t, result, "auto_approve_tools")
	assert.Contains(t, result, `"my_tool"`)
	// [ai] section should still be intact.
	assert.Contains(t, result, `model = "gpt-4o"`)
}

func TestInsertIntoText_CommentsPreserved(t *testing.T) {
	input := `# Werkler configuration

[ai]
model = "gpt-4o"

# MCP server settings
[mcp]
# Tools always allowed without prompting
auto_approve_tools = [
  "git__log",
]
`
	result, err := insertIntoText(input, "mcp", "auto_approve_tools", "fs__read_file")
	require.NoError(t, err)
	assert.Contains(t, result, "# Werkler configuration")
	assert.Contains(t, result, "# MCP server settings")
	assert.Contains(t, result, "# Tools always allowed without prompting")
	assert.Contains(t, result, `"git__log"`)
	assert.Contains(t, result, `"fs__read_file"`)
}

func TestAppendAutoApproveTool_NewFile(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "werkler-*.toml")
	require.NoError(t, err)
	require.NoError(t, f.Close())
	require.NoError(t, os.Remove(f.Name())) // start from non-existent

	require.NoError(t, AppendAutoApproveTool(f.Name(), "fs__read_file"))

	data, err := os.ReadFile(f.Name())
	require.NoError(t, err)
	assert.Contains(t, string(data), "auto_approve_tools")
	assert.Contains(t, string(data), `"fs__read_file"`)
}

func TestAppendAutoApproveTool_Idempotent(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/config.toml"
	require.NoError(t, AppendAutoApproveTool(path, "fs__read_file"))
	first, _ := os.ReadFile(path)
	require.NoError(t, AppendAutoApproveTool(path, "fs__read_file"))
	second, _ := os.ReadFile(path)
	assert.Equal(t, string(first), string(second), "second call should not modify the file")
}

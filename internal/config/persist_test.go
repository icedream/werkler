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

func TestRemoveMCPServerBlock_FirstServer(t *testing.T) {
	input := `[[mcp.servers]]
name = "github"
transport = "streamable-http"
url = "https://api.githubcopilot.com/mcp/"

[[mcp.servers]]
name = "gitlab"
transport = "streamable-http"
url = "https://gitlab.example.com/mcp/"
`
	result := removeMCPServerBlock(input, "github")
	assert.NotContains(t, result, `name = "github"`)
	assert.Contains(t, result, `name = "gitlab"`)
}

func TestRemoveMCPServerBlock_MiddleServer(t *testing.T) {
	input := `[[mcp.servers]]
name = "a"
transport = "streamable-http"
url = "https://a.example.com/"

[[mcp.servers]]
name = "b"
transport = "streamable-http"
url = "https://b.example.com/"

[[mcp.servers]]
name = "c"
transport = "streamable-http"
url = "https://c.example.com/"
`
	result := removeMCPServerBlock(input, "b")
	assert.NotContains(t, result, `name = "b"`)
	assert.Contains(t, result, `name = "a"`)
	assert.Contains(t, result, `name = "c"`)
}

func TestRemoveMCPServerBlock_LastServer(t *testing.T) {
	input := `[[mcp.servers]]
name = "first"
transport = "streamable-http"
url = "https://first.example.com/"

[[mcp.servers]]
name = "last"
transport = "streamable-http"
url = "https://last.example.com/"
`
	result := removeMCPServerBlock(input, "last")
	assert.NotContains(t, result, `name = "last"`)
	assert.Contains(t, result, `name = "first"`)
}

func TestRemoveMCPServerBlock_NotFound(t *testing.T) {
	input := `[[mcp.servers]]
name = "existing"
transport = "streamable-http"
url = "https://existing.example.com/"
`
	result := removeMCPServerBlock(input, "nonexistent")
	assert.Equal(t, input, result)
}

func TestRemoveMCPServerBlock_WithEnvSubTable(t *testing.T) {
	input := `[[mcp.servers]]
name = "first"
transport = "streamable-http"
url = "https://first.example.com/"

[[mcp.servers]]
name = "withenv"
transport = "stdio"
command = "myserver"

[mcp.servers.env]
API_KEY = "secret"
TOKEN = "tok"

[[mcp.servers]]
name = "third"
transport = "streamable-http"
url = "https://third.example.com/"
`
	result := removeMCPServerBlock(input, "withenv")
	assert.NotContains(t, result, `name = "withenv"`)
	assert.NotContains(t, result, `API_KEY`)
	assert.Contains(t, result, `name = "first"`)
	assert.Contains(t, result, `name = "third"`)
}

func TestRemoveMCPServerBlock_SingleServer(t *testing.T) {
	input := `[[mcp.servers]]
name = "only"
transport = "streamable-http"
url = "https://only.example.com/"
`
	result := removeMCPServerBlock(input, "only")
	assert.NotContains(t, result, `name = "only"`)
}

func TestRemoveMCPServerBlock_EOFWithoutTrailingNewline(t *testing.T) {
	input := "[[mcp.servers]]\nname = \"a\"\ntransport = \"streamable-http\"\nurl = \"https://a.example.com/\""
	result := removeMCPServerBlock(input, "a")
	assert.NotContains(t, result, `name = "a"`)
}

func TestAppendMCPServer_PersistsOAuthAndHint(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/config.toml"

	srv := MCPServerConfig{
		Name:      "acme",
		Transport: MCPTransportStreamable,
		URL:       "https://mcp.acme.example.com/mcp",
		OAuth:     true,
		Hint:      "Manage Acme widgets and deployments",
	}
	require.NoError(t, AppendMCPServer(path, srv))

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	text := string(data)

	assert.Contains(t, text, "oauth = true")
	assert.Contains(t, text, `hint = "Manage Acme widgets and deployments"`)
	assert.Contains(t, text, `url = "https://mcp.acme.example.com/mcp"`)
}

func TestAppendMCPServer_NoOAuthFieldWhenFalse(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/config.toml"

	srv := MCPServerConfig{
		Name:      "plain",
		Transport: MCPTransportStreamable,
		URL:       "https://mcp.plain.example.com/mcp",
	}
	require.NoError(t, AppendMCPServer(path, srv))

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.NotContains(t, string(data), "oauth")
	assert.NotContains(t, string(data), "hint")
}

func TestAppendMCPServer_OAuthClientIDPersisted(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/config.toml"

	srv := MCPServerConfig{
		Name:          "github",
		Transport:     MCPTransportStreamable,
		URL:           "https://api.github.com/mcp",
		OAuth:         true,
		OAuthClientID: "Iv1.abc123",
	}
	require.NoError(t, AppendMCPServer(path, srv))

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	text := string(data)
	assert.Contains(t, text, "oauth = true")
	assert.Contains(t, text, `oauth_client_id = "Iv1.abc123"`)
	assert.NotContains(t, text, "oauth_client_secret")
}

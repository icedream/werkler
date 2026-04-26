package mcp

import (
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
)

// --- sanitize ---

func TestSanitize_SafeCharacters(t *testing.T) {
	assert.Equal(t, "hello_world-123", sanitize("hello_world-123"))
}

func TestSanitize_Dots(t *testing.T) {
	assert.Equal(t, "foo_bar", sanitize("foo.bar"))
}

func TestSanitize_Slashes(t *testing.T) {
	assert.Equal(t, "a_b_c", sanitize("a/b/c"))
}

func TestSanitize_Spaces(t *testing.T) {
	assert.Equal(t, "foo_bar", sanitize("foo bar"))
}

func TestSanitize_MixedUnsafe(t *testing.T) {
	assert.Equal(t, "get_file_contents", sanitize("get/file.contents"))
}

func TestSanitize_Empty(t *testing.T) {
	assert.Equal(t, "", sanitize(""))
}

// --- splitToolName ---

func TestSplitToolName_Valid(t *testing.T) {
	server, tool, ok := splitToolName("myserver__mytool")
	assert.True(t, ok)
	assert.Equal(t, "myserver", server)
	assert.Equal(t, "mytool", tool)
}

func TestSplitToolName_NoSeparator(t *testing.T) {
	_, _, ok := splitToolName("notool")
	assert.False(t, ok)
}

func TestSplitToolName_MultipleDoubleSeparators(t *testing.T) {
	// Splits on first occurrence only; tool part may itself contain __.
	server, tool, ok := splitToolName("server__tool__extra")
	assert.True(t, ok)
	assert.Equal(t, "server", server)
	assert.Equal(t, "tool__extra", tool)
}

func TestSplitToolName_EmptyServer(t *testing.T) {
	// A leading __ gives an empty server name — still splits.
	server, tool, ok := splitToolName("__tool")
	assert.True(t, ok)
	assert.Equal(t, "", server)
	assert.Equal(t, "tool", tool)
}

// --- renderResult ---

func TestRenderResult_SingleText(t *testing.T) {
	result := &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: "hello"}},
	}
	out, isErr := renderResult(result)
	assert.Equal(t, "hello", out)
	assert.False(t, isErr)
}

func TestRenderResult_MultipleText(t *testing.T) {
	result := &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: "line1"},
			&mcp.TextContent{Text: "line2"},
		},
	}
	out, isErr := renderResult(result)
	assert.Equal(t, "line1\nline2", out)
	assert.False(t, isErr)
}

func TestRenderResult_ErrorFlag(t *testing.T) {
	result := &mcp.CallToolResult{
		IsError: true,
		Content: []mcp.Content{&mcp.TextContent{Text: "something went wrong"}},
	}
	out, isErr := renderResult(result)
	assert.True(t, isErr)
	assert.Equal(t, "something went wrong", out)
}

func TestRenderResult_ErrorFlagNoContent(t *testing.T) {
	result := &mcp.CallToolResult{IsError: true}
	out, isErr := renderResult(result)
	assert.True(t, isErr)
	assert.Equal(t, "(tool returned an error with no message)", out)
}

func TestRenderResult_EmptyContent(t *testing.T) {
	result := &mcp.CallToolResult{}
	out, isErr := renderResult(result)
	assert.Equal(t, "", out)
	assert.False(t, isErr)
}

func TestRenderResult_NonTextContent_JSONEncoded(t *testing.T) {
	// ImageContent is not TextContent; it should be JSON-encoded.
	result := &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.ImageContent{Data: []byte("abc"), MIMEType: "image/png"}},
	}
	out, isErr := renderResult(result)
	assert.Contains(t, out, "image/png")
	assert.False(t, isErr)
}

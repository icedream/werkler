package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// callRunCommand is a test helper that calls handleRunCommand and unmarshals
// the JSON result into a map.
func callRunCommand(t *testing.T, args map[string]any) map[string]any {
	t.Helper()
	m := newTestManager()
	result, err := m.handleRunCommand(context.Background(), args)
	require.NoError(t, err)
	var out map[string]any
	require.NoError(t, json.Unmarshal([]byte(result), &out))
	return out
}

// TestRunCommand_BasicEcho verifies a simple echo command returns correct output.
func TestRunCommand_BasicEcho(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"command": "echo",
		"args":    []any{"hello", "world"},
		"title":   "echo test",
	})
	assert.Nil(t, out["error"])
	assert.EqualValues(t, 0, out["exit_code"])
	assert.False(t, out["timed_out"].(bool))
	assert.Contains(t, out["stdout"].(string), "hello world")
	assert.Contains(t, out["combined_output"].(string), "hello world")
}

// TestRunCommand_ShellMode verifies shell=true enables pipe syntax.
func TestRunCommand_ShellMode(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"command": "echo hello | tr 'a-z' 'A-Z'",
		"shell":   true,
		"title":   "shell pipe test",
	})
	assert.Nil(t, out["error"])
	assert.EqualValues(t, 0, out["exit_code"])
	assert.Contains(t, out["stdout"].(string), "HELLO")
}

// TestRunCommand_ShellWithArgsFails verifies that args + shell=true returns a structured error.
func TestRunCommand_ShellWithArgsFails(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"command": "echo foo",
		"args":    []any{"bar"},
		"shell":   true,
		"title":   "shell+args test",
	})
	require.NotNil(t, out["error"])
	assert.Contains(t, out["error"].(string), "args must not be provided when shell is true")
}

// TestRunCommand_NonzeroExit verifies non-zero exit codes are reported correctly.
func TestRunCommand_NonzeroExit(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"command": "bash",
		"args":    []any{"-c", "exit 42"},
		"title":   "exit 42 test",
	})
	assert.Nil(t, out["error"])
	assert.EqualValues(t, 42, out["exit_code"])
	assert.False(t, out["timed_out"].(bool))
}

// TestRunCommand_StderrCaptured verifies stderr is captured separately.
func TestRunCommand_StderrCaptured(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"command": "bash",
		"args":    []any{"-c", "echo to-stdout; echo to-stderr >&2"},
		"title":   "stderr test",
	})
	assert.Nil(t, out["error"])
	assert.Contains(t, out["stdout"].(string), "to-stdout")
	assert.Contains(t, out["stderr"].(string), "to-stderr")
	// Both streams should appear in combined output.
	combined := out["combined_output"].(string)
	assert.Contains(t, combined, "to-stdout")
	assert.Contains(t, combined, "to-stderr")
}

// TestRunCommand_EnvOverride verifies env vars are passed to the child.
func TestRunCommand_EnvOverride(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"command": "bash",
		"args":    []any{"-c", "echo $MY_TEST_VAR"},
		"env":     map[string]any{"MY_TEST_VAR": "werkler-test-value"},
		"title":   "env test",
	})
	assert.Nil(t, out["error"])
	assert.Contains(t, out["stdout"].(string), "werkler-test-value")
}

// TestRunCommand_EnvUnset verifies a null env value removes the variable.
func TestRunCommand_EnvUnset(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"command": "bash",
		"args":    []any{"-c", "echo ${UNSET_VAR:-was-unset}"},
		"env":     map[string]any{"UNSET_VAR": nil},
		"title":   "env unset test",
	})
	assert.Nil(t, out["error"])
	assert.Contains(t, out["stdout"].(string), "was-unset")
}

// TestRunCommand_Timeout verifies that a hanging process is killed and timed_out is set.
func TestRunCommand_Timeout(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"command":         "sleep",
		"args":            []any{"300"},
		"timeout_seconds": float64(1),
		"title":           "timeout test",
	})
	assert.Nil(t, out["error"])
	assert.True(t, out["timed_out"].(bool), "expected timed_out to be true")
	// exit_code should be non-zero (signal kill -- 128+9=137 typically)
	assert.NotEqualValues(t, 0, out["exit_code"])
}

// TestRunCommand_MissingCommand verifies a structured error for missing command.
func TestRunCommand_MissingCommand(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"title": "missing command test",
	})
	require.NotNil(t, out["error"])
	assert.Contains(t, out["error"].(string), "command is required")
}

// TestRunCommand_UnknownCommand verifies a structured error for ENOENT.
func TestRunCommand_UnknownCommand(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"command": "this-command-does-not-exist-werkler-test",
		"title":   "unknown command test",
	})
	require.NotNil(t, out["error"])
}

// TestRunCommand_BadCwd verifies a structured error for a non-existent cwd.
func TestRunCommand_BadCwd(t *testing.T) {
	out := callRunCommand(t, map[string]any{
		"command": "echo",
		"args":    []any{"hi"},
		"cwd":     "/this/path/does/not/exist/werkler-test",
		"title":   "bad cwd test",
	})
	require.NotNil(t, out["error"])
	assert.Contains(t, out["error"].(string), "cwd does not exist")
}

// TestRunCommand_OutputCap verifies that output beyond the cap is truncated.
func TestRunCommand_OutputCap(t *testing.T) {
	// Generate output well beyond the 512 KiB cap.
	// dd produces exactly count*bs bytes. We generate 1 MiB of 'a' bytes.
	out := callRunCommand(t, map[string]any{
		"command": "bash",
		"args":    []any{"-c", "dd if=/dev/zero bs=1024 count=1024 2>/dev/null | tr '\\0' 'a'"},
		"title":   "output cap test",
	})
	assert.Nil(t, out["error"])
	stdout := out["stdout"].(string)
	// Should be capped -- check it contains a truncation notice.
	assert.True(t, strings.Contains(stdout, "[truncated:"), "expected truncation notice in stdout")
}

// TestCapBuffer_WritesUpToCap checks that capBuffer stops accepting bytes after the cap.
func TestCapBuffer_WritesUpToCap(t *testing.T) {
	b := newCapBuffer(10)
	n, err := b.Write([]byte("hello"))
	require.NoError(t, err)
	assert.Equal(t, 5, n)

	n, err = b.Write([]byte("worldXYZ")) // 8 bytes; only 5 fit
	require.NoError(t, err)
	assert.Equal(t, 8, n) // Write still reports all bytes consumed (io.Writer contract)

	s := b.String()
	assert.Contains(t, s, "helloworld")
	assert.Contains(t, s, "[truncated: 3 bytes omitted]")
}

// TestCapBuffer_NoTruncationNoticeWhenFits checks the normal (no-truncation) path.
func TestCapBuffer_NoTruncationNoticeWhenFits(t *testing.T) {
	b := newCapBuffer(100)
	_, _ = b.Write([]byte("hello"))
	s := b.String()
	assert.Equal(t, "hello", s)
}

package ui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func noopAction(args string) (string, []tea.Cmd) { return "", nil }

func withExternalCommands(t *testing.T, cmds []ExternalCommand) {
	t.Helper()
	orig := externalCommands
	t.Cleanup(func() { externalCommands = orig })
	externalCommands = cmds
}

// TestRegisterCommand_AppearsInFilteredCmds verifies that a registered command
// shows up when the user types "/" in stateIdle.
func TestRegisterCommand_AppearsInFilteredCmds(t *testing.T) {
	withExternalCommands(t, nil)
	RegisterCommand(ExternalCommand{
		Name:        "testcmd",
		Description: "a test command",
		Action:      noopAction,
	})

	m := baseModel(t)
	m.input.SetValue("/testcmd")

	filtered := m.filteredCmds()
	require.Len(t, filtered, 1)
	assert.Equal(t, "testcmd", filtered[0].name)
	assert.Equal(t, "a test command", filtered[0].description)
}

// TestMatchExternal covers exact match, args match, and no-match cases.
func TestMatchExternal(t *testing.T) {
	withExternalCommands(t, nil)
	RegisterCommand(ExternalCommand{Name: "greet", Description: "greet", Action: noopAction})

	ext, args, ok := matchExternal("/greet")
	require.True(t, ok)
	assert.Equal(t, "greet", ext.Name)
	assert.Equal(t, "", args)

	ext2, args2, ok2 := matchExternal("/greet world")
	require.True(t, ok2)
	assert.Equal(t, "greet", ext2.Name)
	assert.Equal(t, "world", args2)

	_, _, ok3 := matchExternal("/unknown")
	assert.False(t, ok3)

	_, _, ok4 := matchExternal("/greetextra") // prefix but not a match
	assert.False(t, ok4)
}

// TestRegisterCommand_SafeWhileBusyFiltering verifies that unsafe commands are
// hidden from filteredCmds when the model is busy.
func TestRegisterCommand_SafeWhileBusyFiltering(t *testing.T) {
	withExternalCommands(t, nil)
	RegisterCommand(ExternalCommand{Name: "safe", Description: "safe", SafeWhileBusy: true, Action: noopAction})
	RegisterCommand(ExternalCommand{Name: "unsafe", Description: "unsafe", SafeWhileBusy: false, Action: noopAction})

	m := baseModel(t)
	m.state = stateThinking
	m.input.SetValue("/")

	filtered := m.filteredCmds()
	names := make([]string, len(filtered))
	for i, c := range filtered {
		names[i] = c.name
	}
	assert.Contains(t, names, "safe")
	assert.NotContains(t, names, "unsafe")
}

// TestRegisterCommand_ActionInvoked verifies that the action is called with the
// correct args and its message is appended to items.
func TestRegisterCommand_ActionInvoked(t *testing.T) {
	withExternalCommands(t, nil)
	var got string
	RegisterCommand(ExternalCommand{
		Name:        "echo",
		Description: "echo args",
		Action: func(args string) (string, []tea.Cmd) {
			got = args
			return "echoed: " + args, nil
		},
	})

	m := baseModel(t)
	m.state = stateIdle
	m.input.SetValue("/echo hello world")
	m.collapsedHandles = make(map[string]bool)

	next, _ := update(t, m, tea.KeyMsg{Type: tea.KeyEnter})
	assert.Equal(t, "hello world", got)
	require.NotEmpty(t, next.items)
	last := next.items[len(next.items)-1]
	assert.Equal(t, itemInfo, last.kind)
	assert.Equal(t, "echoed: hello world", last.content)
}

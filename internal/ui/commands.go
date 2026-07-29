package ui

import (
	"strings"

	tea "charm.land/bubbletea/v2"
)

// ExternalCommand defines a /command that can be registered from outside the
// ui package, e.g. from the application entry point or a plugin package.
//
// The action receives no *Model pointer so external packages do not need to
// import or depend on Model internals.  Use closures to capture any external
// state the command needs.
type ExternalCommand struct {
	// Name is the command name without the leading slash, e.g. "mycommand".
	Name string
	// Description is shown in /help output and the completion popup.
	Description string
	// Action is called when the command is invoked.  args is the trimmed text
	// after the command name (empty when invoked via the completion popup with
	// no arguments).  Returning a non-empty msg appends it as an itemInfo
	// bubble; cmds are forwarded to the bubbletea runtime.
	Action func(args string) (msg string, cmds []tea.Cmd)
	// SafeWhileBusy, when true, allows this command to be run while the AI is
	// actively thinking / streaming / calling tools.
	SafeWhileBusy bool
	// Available, if non-nil, controls whether the command appears in the
	// completion popup.  It receives no context; use a closure to capture state.
	Available func() bool
}

// externalCommands holds all commands registered via RegisterCommand.
var externalCommands []ExternalCommand

// RegisterCommand registers an external /command that will appear alongside
// the built-in commands in the completion popup and be executed when the user
// types it.  Must be called before RunTUI starts.
func RegisterCommand(cmd ExternalCommand) {
	externalCommands = append(externalCommands, cmd)
}

// wrapExternal converts an ExternalCommand into an internal slashCommand, using
// args as the argument string (empty when invoked via the completion popup).
func wrapExternal(ext ExternalCommand, args string) slashCommand {
	return slashCommand{
		name:          ext.Name,
		description:   ext.Description,
		safeWhileBusy: ext.SafeWhileBusy,
		available: func(_ *Model) bool {
			if ext.Available != nil {
				return ext.Available()
			}
			return true
		},
		action: func(m *Model) []tea.Cmd {
			msg, extCmds := ext.Action(args)
			if msg != "" {
				m.items = append(m.items, displayItem{kind: itemInfo, content: msg})
				m.rebuildContent()
			}
			return extCmds
		},
	}
}

// matchExternal returns the first ExternalCommand whose name matches text
// (with or without trailing args), and the extracted args string.
// Returns ok=false when no command matches.
func matchExternal(text string) (ext ExternalCommand, args string, ok bool) {
	for _, e := range externalCommands {
		cmdFull := "/" + e.Name
		if text == cmdFull {
			return e, "", true
		}
		if len(text) > len(cmdFull) && text[:len(cmdFull)] == cmdFull && text[len(cmdFull)] == ' ' {
			return e, strings.TrimSpace(text[len(cmdFull):]), true
		}
	}
	return ExternalCommand{}, "", false
}

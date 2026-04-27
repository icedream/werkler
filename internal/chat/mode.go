package chat

import (
	"fmt"
	"strings"

	"github.com/icedream/werkler/internal/config"
)

// ResolvedMode is a fully-merged mode configuration, ready to apply to a session.
type ResolvedMode struct {
	Name               string
	IsDefault          bool
	SystemPromptExtra  string
	Autopilot          *bool // nil = don't override the session setting
	AutopilotMaxCycles int   // 0 = don't override
	AutoApproveTools   []string
}

// builtinModeList is the canonical ordered list of built-in modes.
var builtinModeList = []ResolvedMode{
	{
		Name:      "default",
		IsDefault: true,
	},
	{
		Name:              "plan",
		SystemPromptExtra: planModePrompt,
	},
	{
		Name:              "document",
		SystemPromptExtra: documentModePrompt,
	},
}

// builtinModes is a map of built-in mode name → ResolvedMode for fast lookup.
var builtinModes map[string]ResolvedMode

func init() {
	builtinModes = make(map[string]ResolvedMode, len(builtinModeList))
	for _, m := range builtinModeList {
		builtinModes[m.Name] = m
	}
}

const planModePrompt = `## Mode: Plan

You are in planning mode. Prioritise:
- Breaking down tasks into clear, ordered steps before writing any code.
- Drafting well-structured tickets with: Summary, Background, Acceptance Criteria, Technical Notes.
- Reviewing and critiquing designs — identify gaps, risks, and edge cases.
- Identifying dependencies and sequencing work correctly.

Prefer structured output (numbered steps, tables, criteria lists). Do not write implementation code unless explicitly asked. Stop and confirm with the user before committing to a full plan.`

const documentModePrompt = `## Mode: Document

You are in documentation mode. Prioritise:
- Writing clear, accurate, and well-structured documentation.
- Tailoring language and depth to the intended audience.
- Consistent formatting, terminology, and tone throughout.
- Producing self-contained documents that do not require this chat context to be understood.

Prefer clear prose with appropriate headings, lists, and code examples where helpful.`

// DefaultMode returns the built-in default mode.
func DefaultMode() ResolvedMode {
	return builtinModes["default"]
}

// BuiltinModeNames returns the names of all built-in modes in display order.
func BuiltinModeNames() []string {
	names := make([]string, len(builtinModeList))
	for i, m := range builtinModeList {
		names[i] = m.Name
	}
	return names
}

// ResolveMode returns the ResolvedMode for name, merging user config overrides
// with built-in definitions. An empty name returns the default mode.
func ResolveMode(name string, userModes []config.ModeConfig) (ResolvedMode, error) {
	if name == "" || name == "default" {
		return builtinModes["default"], nil
	}

	// User modes shadow built-ins with the same name.
	for _, um := range userModes {
		if um.Name == name {
			return resolveUserMode(um, userModes)
		}
	}

	// Fall back to built-in.
	if bm, ok := builtinModes[name]; ok {
		return bm, nil
	}

	available := append(BuiltinModeNames(), userModeNames(userModes)...)
	return ResolvedMode{}, fmt.Errorf("unknown mode %q — available: %s", name, strings.Join(available, ", "))
}

// AllModes returns all available modes: built-ins in order (user overrides
// substituted in place), followed by additional user-defined modes.
func AllModes(userModes []config.ModeConfig) ([]ResolvedMode, error) {
	out := make([]ResolvedMode, 0, len(builtinModeList)+len(userModes))

	// Built-ins first; substitute any user mode with the same name.
	for _, bm := range builtinModeList {
		substituted := false
		for _, um := range userModes {
			if um.Name == bm.Name {
				resolved, err := resolveUserMode(um, userModes)
				if err != nil {
					return nil, err
				}
				out = append(out, resolved)
				substituted = true
				break
			}
		}
		if !substituted {
			out = append(out, bm)
		}
	}

	// Append user modes whose names are not built-ins.
	for _, um := range userModes {
		if _, isBuiltin := builtinModes[um.Name]; !isBuiltin {
			resolved, err := resolveUserMode(um, userModes)
			if err != nil {
				return nil, err
			}
			out = append(out, resolved)
		}
	}

	return out, nil
}

// resolveUserMode merges a user-defined mode onto its optional base.
func resolveUserMode(um config.ModeConfig, allUserModes []config.ModeConfig) (ResolvedMode, error) {
	base := ResolvedMode{Name: um.Name}

	if um.Base != "" {
		var err error
		base, err = ResolveMode(um.Base, allUserModes)
		if err != nil {
			return ResolvedMode{}, fmt.Errorf("mode %q: resolving base %q: %w", um.Name, um.Base, err)
		}
		base.Name = um.Name
		base.IsDefault = false
	}

	if um.SystemPromptExtra != "" {
		base.SystemPromptExtra = um.SystemPromptExtra
	}
	if um.Autopilot != nil {
		v := *um.Autopilot
		base.Autopilot = &v
	}
	if um.AutopilotMaxCycles > 0 {
		base.AutopilotMaxCycles = um.AutopilotMaxCycles
	}
	if len(um.AutoApproveTools) > 0 {
		merged := make([]string, len(base.AutoApproveTools)+len(um.AutoApproveTools))
		copy(merged, base.AutoApproveTools)
		copy(merged[len(base.AutoApproveTools):], um.AutoApproveTools)
		base.AutoApproveTools = merged
	}

	return base, nil
}

func userModeNames(userModes []config.ModeConfig) []string {
	names := make([]string, 0, len(userModes))
	for _, um := range userModes {
		if _, isBuiltin := builtinModes[um.Name]; !isBuiltin {
			names = append(names, um.Name)
		}
	}
	return names
}

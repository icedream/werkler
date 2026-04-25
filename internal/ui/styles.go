package ui

import (
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
)

// thinkingSpinner is a single circle (●) that pulses through a purple→pink→purple
// color gradient, readable at any font size.
var thinkingSpinner = spinner.Spinner{
	Frames: func() []string {
		colors := []lipgloss.Color{"57", "63", "99", "135", "171", "207", "171", "135", "99", "63"}
		f := make([]string, len(colors))
		for i, c := range colors {
			f[i] = lipgloss.NewStyle().Foreground(c).Render("●")
		}
		return f
	}(),
	FPS: time.Second / 10,
}

var (
	headerStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("212")).
			Padding(0, 1)

	separatorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("238"))

	userPrefixStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("39"))

	assistantPrefixStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("212"))

	toolNameStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("220"))

	toolApprovedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("42"))

	toolDeniedStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("196"))

	toolPendingStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("244"))

	statusStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("244"))

	approvalPromptStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("220")).
				Bold(true)

	keyHintStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("39")).
			Bold(true)

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("196")).
			Bold(true)

	infoStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("244")).
			Italic(true)

	completionItemStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("250"))

	completionSelectedStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("212"))

	completionNameStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("39")).
				Bold(true)

	completionSelectedNameStyle = lipgloss.NewStyle().
					Foreground(lipgloss.Color("212")).
					Bold(true)

	inputPrefixStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("244"))

	processHandleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("36"))

	queueCountStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("39")).
			Bold(true)

	choiceSelectedStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("212"))

	choiceRecommendedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("42")).
				Italic(true)

	allowAllWarningStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.Color("202")) // orange — visible warning
)

func separator(width int) string {
	return separatorStyle.Render(strings.Repeat("─", width))
}

func newGlamourRenderer(width int, style string) *glamour.TermRenderer {
	if width < 20 {
		width = 20
	}
	r, err := glamour.NewTermRenderer(
		glamour.WithStylePath(style),
		glamour.WithWordWrap(width),
	)
	if err != nil {
		return nil
	}
	return r
}

func renderMarkdown(r *glamour.TermRenderer, content string) string {
	if r == nil {
		return content
	}
	rendered, err := r.Render(content)
	if err != nil {
		return content
	}
	return strings.TrimRight(rendered, "\n")
}

// Package docs embeds werkler's documentation markdown files and provides
// a lightweight keyword search over their content.
package docs

import (
	"embed"
	"fmt"
	"io/fs"
	"strings"
	"unicode"
)

//go:embed *.md
var embeddedFS embed.FS

// Section is a single heading + body block parsed from a markdown file.
type Section struct {
	File    string // e.g. "tui.md"
	Heading string // e.g. "## Keybindings"
	Body    string // text beneath this heading until the next same-or-higher heading
}

// All parses all embedded markdown files into Sections split on ## / ### headings.
// The text before the first heading in each file is treated as a preamble section
// with an empty Heading.
func All() []Section {
	var out []Section

	entries, _ := fs.ReadDir(embeddedFS, ".")
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		data, err := embeddedFS.ReadFile(e.Name())
		if err != nil {
			continue
		}
		out = append(out, parseSections(e.Name(), string(data))...)
	}
	return out
}

// Search returns documentation sections relevant to the given query.
// It scores each section by keyword overlap (heading matches weighted 3×,
// body matches weighted 1×), returns the top-4 scoring sections, and caps
// the total output at 4000 characters. If no section scores above zero,
// a table of contents listing all headings is returned instead.
func Search(query string) string {
	tokens := tokenise(query)
	if len(tokens) == 0 {
		return tableOfContents()
	}

	sections := All()
	type scored struct {
		s     Section
		score int
	}
	scores := make([]scored, len(sections))
	for i, s := range sections {
		headTokens := tokenise(s.Heading)
		bodyTokens := tokenise(s.Body)
		var sc int
		for _, t := range tokens {
			for _, h := range headTokens {
				if h == t {
					sc += 3
				}
			}
			for _, b := range bodyTokens {
				if b == t {
					sc++
				}
			}
		}
		scores[i] = scored{s, sc}
	}

	// Simple insertion sort — section count is small (~60).
	for i := 1; i < len(scores); i++ {
		for j := i; j > 0 && scores[j].score > scores[j-1].score; j-- {
			scores[j], scores[j-1] = scores[j-1], scores[j]
		}
	}

	if scores[0].score == 0 {
		return tableOfContents()
	}

	const maxSections = 4
	const maxChars = 4000

	var sb strings.Builder
	for i := 0; i < len(scores) && i < maxSections; i++ {
		if scores[i].score == 0 {
			break
		}
		s := scores[i].s
		chunk := formatSection(s)
		if sb.Len()+len(chunk) > maxChars {
			// Include a truncated version if we have room for at least a header.
			remaining := maxChars - sb.Len()
			if remaining > len(fmt.Sprintf("### %s / %s\n", s.File, s.Heading))+50 {
				sb.WriteString(chunk[:remaining])
				sb.WriteString("\n… (truncated)")
			}
			break
		}
		sb.WriteString(chunk)
	}
	return strings.TrimSpace(sb.String())
}

// tableOfContents returns a plain list of all section headings grouped by file.
func tableOfContents() string {
	sections := All()
	var sb strings.Builder
	sb.WriteString("No exact match found. Available documentation sections:\n\n")
	lastFile := ""
	for _, s := range sections {
		if s.File != lastFile {
			sb.WriteString("\n**")
			sb.WriteString(s.File)
			sb.WriteString("**\n")
			lastFile = s.File
		}
		if s.Heading != "" {
			sb.WriteString("  ")
			sb.WriteString(s.Heading)
			sb.WriteString("\n")
		}
	}
	return strings.TrimSpace(sb.String())
}

// formatSection renders a section as a labelled markdown block.
func formatSection(s Section) string {
	var sb strings.Builder
	sb.WriteString("---\n")
	sb.WriteString("**Source: ")
	sb.WriteString(s.File)
	sb.WriteString("**\n")
	if s.Heading != "" {
		sb.WriteString(s.Heading)
		sb.WriteString("\n")
	}
	sb.WriteString(strings.TrimSpace(s.Body))
	sb.WriteString("\n\n")
	return sb.String()
}

// parseSections splits markdown content into sections on ## and ### headings.
func parseSections(filename, content string) []Section {
	lines := strings.Split(content, "\n")
	var sections []Section
	var curHeading string
	var curBody strings.Builder

	flush := func() {
		body := strings.TrimSpace(curBody.String())
		if body != "" || curHeading != "" {
			sections = append(sections, Section{
				File:    filename,
				Heading: curHeading,
				Body:    body,
			})
		}
		curBody.Reset()
	}

	for _, line := range lines {
		if strings.HasPrefix(line, "## ") || strings.HasPrefix(line, "### ") {
			flush()
			curHeading = strings.TrimSpace(line)
		} else {
			curBody.WriteString(line)
			curBody.WriteByte('\n')
		}
	}
	flush()
	return sections
}

// tokenise splits s into lowercase word tokens, stripping punctuation.
func tokenise(s string) []string {
	s = strings.ToLower(s)
	return strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

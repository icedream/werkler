// Package skills loads agent skill definitions from a directory of SKILL.md files.
// Each subdirectory of the skills dir may contain a SKILL.md with YAML front matter
// (name, description) followed by markdown content.
//
// Lines in the content matching the pattern !`command` are treated as shell
// commands: they are executed once at load time (in the process working directory)
// and their stdout replaces the line in the skill's content.
package skills

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Skill represents a loaded skill definition.
type Skill struct {
	// Name is the machine-readable identifier from the front matter.
	Name string
	// Description is the human-readable summary from the front matter.
	Description string
	// Content is the resolved markdown body, with !`...` lines expanded.
	Content string
	// Dir is the directory that contained SKILL.md.
	Dir string
}

// skillFrontmatter holds the YAML fields parsed from SKILL.md front matter.
type skillFrontmatter struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

// DefaultDir returns the default skills directory (~/.agents/skills).
func DefaultDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".agents", "skills")
}

// ExpandTilde replaces a leading ~ with the user's home directory.
func ExpandTilde(path string) string {
	if path == "~" {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return home
	}
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return filepath.Join(home, path[2:])
	}
	return path
}

// cmdLineRe matches lines of the form !`command` where command is the shell
// command to execute, capturing the command in group 1.
var cmdLineRe = regexp.MustCompile("^!`(.+)`$")

// execCmdLine runs the shell command with a 10-second timeout from the current
// working directory, capping output at 64 KiB. On failure it returns a
// human-readable error placeholder.
func execCmdLine(cmd string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	c := exec.CommandContext(ctx, "sh", "-c", cmd) //nolint:gosec // user-authored skill commands
	r, err := c.Output()
	if err != nil {
		return fmt.Sprintf("[skill command failed: %v]", err)
	}

	const maxBytes = 64 * 1024
	if len(r) > maxBytes {
		r = r[:maxBytes]
	}
	return strings.TrimRight(string(r), "\n")
}

// expandContent processes the raw content, replacing !`cmd` lines with their output.
func expandContent(raw string) string {
	var buf strings.Builder
	scanner := bufio.NewScanner(strings.NewReader(raw))
	for scanner.Scan() {
		line := scanner.Text()
		if m := cmdLineRe.FindStringSubmatch(line); m != nil {
			buf.WriteString(execCmdLine(m[1]))
		} else {
			buf.WriteString(line)
		}
		buf.WriteByte('\n')
	}
	return buf.String()
}

// parseFrontmatter splits a SKILL.md file into front matter and content.
// Returns an error if the front matter is missing or malformed.
func parseFrontmatter(data []byte) (skillFrontmatter, string, error) {
	const sep = "---"

	// Require the file to begin with the opening ---
	if !bytes.HasPrefix(data, []byte("---\n")) && !bytes.HasPrefix(data, []byte("---\r\n")) {
		return skillFrontmatter{}, "", fmt.Errorf("SKILL.md must begin with YAML front matter (---)")
	}

	// Find closing ---
	rest := data[4:] // skip opening "---\n"
	idx := bytes.Index(rest, []byte("\n"+sep))
	if idx == -1 {
		return skillFrontmatter{}, "", fmt.Errorf("SKILL.md front matter is not closed")
	}

	yamlBytes := rest[:idx]
	content := string(rest[idx+1+len(sep):])
	if strings.HasPrefix(content, "\n") {
		content = content[1:]
	} else if strings.HasPrefix(content, "\r\n") {
		content = content[2:]
	}

	var fm skillFrontmatter
	if err := yaml.Unmarshal(yamlBytes, &fm); err != nil {
		return skillFrontmatter{}, "", fmt.Errorf("parsing front matter: %w", err)
	}
	if fm.Name == "" {
		return skillFrontmatter{}, "", fmt.Errorf("SKILL.md front matter must have a non-empty 'name'")
	}

	return fm, content, nil
}

// Load parses the SKILL.md in dir and returns a Skill with content expanded.
func Load(dir string) (*Skill, error) {
	data, err := os.ReadFile(filepath.Join(dir, "SKILL.md"))
	if err != nil {
		return nil, err
	}

	fm, rawContent, err := parseFrontmatter(data)
	if err != nil {
		return nil, fmt.Errorf("skill %s: %w", filepath.Base(dir), err)
	}

	return &Skill{
		Name:        fm.Name,
		Description: fm.Description,
		Content:     expandContent(rawContent),
		Dir:         dir,
	}, nil
}

// LoadDir scans dir for skill subdirectories and loads each SKILL.md.
// Directories without a SKILL.md or with malformed metadata are skipped
// with a warning written to w (typically os.Stderr).
// If dir does not exist, an empty slice is returned without error.
func LoadDir(dir string, w io.Writer) ([]Skill, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading skills dir %s: %w", dir, err)
	}

	seen := make(map[string]bool)
	var skills []Skill

	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		s, loadErr := Load(filepath.Join(dir, e.Name()))
		if loadErr != nil {
			if !os.IsNotExist(loadErr) {
				_, _ = fmt.Fprintf(w, "warning: skipping skill %q: %v\n", e.Name(), loadErr)
			}
			continue
		}
		if seen[s.Name] {
			_, _ = fmt.Fprintf(w, "warning: duplicate skill name %q in %s — skipping\n", s.Name, e.Name())
			continue
		}
		seen[s.Name] = true
		skills = append(skills, *s)
	}

	return skills, nil
}

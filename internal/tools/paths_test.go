package tools

import (
	"strings"
	"testing"
)

func TestIsPathLike(t *testing.T) {
	cases := []struct {
		s    string
		want bool
	}{
		{"/usr/bin/go", true},
		{"./main.go", true},
		{"../pkg/foo.go", true},
		{"~/projects/repo", true},
		// Go package wildcards — must NOT be treated as paths
		{"./...", false},
		{"./foo/...", false},
		// Plain identifiers
		{"build", false},
		{"-v", false},
		{"./...", false},
	}
	for _, tc := range cases {
		got := isPathLike(tc.s)
		if got != tc.want {
			t.Errorf("isPathLike(%q) = %v, want %v", tc.s, got, tc.want)
		}
	}
}

func TestContainsEllipsis(t *testing.T) {
	cases := []struct {
		s    string
		want bool
	}{
		{"./...", true},
		{"./foo/...", true},
		{"/usr/bin/go", false},
		{"./main.go", false},
		{"../pkg", false},
	}
	for _, tc := range cases {
		got := containsEllipsis(tc.s)
		if got != tc.want {
			t.Errorf("containsEllipsis(%q) = %v, want %v", tc.s, got, tc.want)
		}
	}
}

func TestContainsShellVar(t *testing.T) {
	cases := []struct {
		s    string
		want bool
	}{
		{"/tmp/${NAME}/output", true},
		{"/tmp/\\${NAME}", true},
		{"${HOME}/projects", true},
		{"$VAR", true},
		{"/usr/local/bin", false},
		{"./src/main.go", false},
		{"/tmp/myfile.txt", false},
	}
	for _, tc := range cases {
		got := containsShellVar(tc.s)
		if got != tc.want {
			t.Errorf("containsShellVar(%q) = %v, want %v", tc.s, got, tc.want)
		}
	}
}

func TestExtractPaths_NoCwd(t *testing.T) {
	// cwd should NOT appear in the extracted paths list — it is the process's
	// starting directory, not a path being read/written.
	paths := ExtractPaths("ls", []string{}, "/root")
	for _, p := range paths {
		if p == "/root" {
			t.Errorf("ExtractPaths included cwd %q as a path", p)
		}
	}
}

func TestExtractPaths_ShellVarFiltered(t *testing.T) {
	// Paths containing shell variable syntax must be excluded from the results.
	paths := ExtractPaths("bash", []string{"-c", "cp /src/file.txt /tmp/${NAME}/dest"}, "/workspace")
	for _, p := range paths {
		if containsShellVar(p) {
			t.Errorf("ExtractPaths included shell-var path %q", p)
		}
	}
	// The literal path /src/file.txt should still be extracted.
	found := false
	for _, p := range paths {
		if p == "/src/file.txt" {
			found = true
		}
	}
	if !found {
		t.Errorf("ExtractPaths did not include literal path /src/file.txt; got %v", paths)
	}
}

func TestExtractPaths_GoWildcard(t *testing.T) {
	// "go build ./..." — ./... must NOT appear in extracted paths
	paths := ExtractPaths("go", []string{"build", "./..."}, "/tmp/project")
	for _, p := range paths {
		if p == "/tmp/project/..." || p == "./..." {
			t.Errorf("ExtractPaths included Go wildcard %q", p)
		}
	}
}

func TestExtractPaths_BinaryFirst(t *testing.T) {
	// The binary should be the first extracted path.
	paths := ExtractPaths("go", []string{"build", "./src"}, "/tmp/project")
	if len(paths) == 0 {
		t.Fatal("expected at least one path (the binary)")
	}
	// First path must resolve to the go binary, not a source dir.
	first := paths[0]
	if first == "/tmp/project/src" {
		t.Errorf("first path is a source dir, expected the binary; paths=%v", paths)
	}
}

// TestExtractShellSubCommands_QuotedPipes verifies that pipes inside quoted
// strings are treated as literal characters and do NOT split the command.
func TestExtractShellSubCommands_QuotedPipes(t *testing.T) {
	// The pipe inside double quotes is a literal character; only the outer
	// pipe should be a separator.
	cmds := extractShellSubCommands(`echo "foo | bar" | grep baz`, "")
	// Should find: echo, grep (NOT a third command from splitting on the quoted pipe)
	if len(cmds) != 2 {
		t.Fatalf("expected 2 commands, got %d: %v", len(cmds), cmds)
	}
	// Commands are resolved to absolute paths.
	if !strings.HasSuffix(cmds[0], "echo") {
		t.Errorf("first command is %q, want echo", cmds[0])
	}
	if !strings.HasSuffix(cmds[1], "grep") {
		t.Errorf("second command is %q, want grep", cmds[1])
	}
}

// TestExtractShellSubCommands_EscapedPipes verifies that escaped pipes are treated
// as literal characters (the pipe becomes part of the argument, not a separator).
func TestExtractShellSubCommands_EscapedPipes(t *testing.T) {
	cmds := extractShellSubCommands(`echo \| grep`, "")
	// \| is a literal pipe argument, not a separator — only echo is the command.
	if len(cmds) != 1 {
		t.Fatalf("expected 1 command, got %d: %v", len(cmds), cmds)
	}
	if !strings.HasSuffix(cmds[0], "echo") {
		t.Errorf("command is %q, want echo", cmds[0])
	}
}

// TestExtractShellSubCommands_CommandSubstitution verifies that pipes inside
// $() are treated as literal characters.
func TestExtractShellSubCommands_CommandSubstitution(t *testing.T) {
	cmds := extractShellSubCommands(`echo $(ls | grep foo)`, "")
	// Should find: echo, ls, grep
	if len(cmds) != 3 {
		t.Fatalf("expected 3 commands, got %d: %v", len(cmds), cmds)
	}
	if !strings.HasSuffix(cmds[0], "echo") {
		t.Errorf("first command is %q, want echo", cmds[0])
	}
	if !strings.HasSuffix(cmds[1], "ls") {
		t.Errorf("second command is %q, want ls", cmds[1])
	}
	if !strings.HasSuffix(cmds[2], "grep") {
		t.Errorf("third command is %q, want grep", cmds[2])
	}
}

// TestExtractShellSubCommands_Subshells verifies that pipes inside () are treated
// as literal characters.
func TestExtractShellSubCommands_Subshells(t *testing.T) {
	cmds := extractShellSubCommands(`(echo | grep)`, "")
	// Should find: echo, grep
	if len(cmds) != 2 {
		t.Fatalf("expected 2 commands, got %d: %v", len(cmds), cmds)
	}
	if !strings.HasSuffix(cmds[0], "echo") {
		t.Errorf("first command is %q, want echo", cmds[0])
	}
	if !strings.HasSuffix(cmds[1], "grep") {
		t.Errorf("second command is %q, want grep", cmds[1])
	}
}

// TestExtractShellSubCommands_Nested verifies nested structures work.
func TestExtractShellSubCommands_Nested(t *testing.T) {
	cmds := extractShellSubCommands(`$(echo "$(echo | grep)")`, "")
	// Should find: echo, grep (echo appears twice in nesting, but deduped)
	if len(cmds) != 2 {
		t.Fatalf("expected 2 commands, got %d: %v", len(cmds), cmds)
	}
	if !strings.HasSuffix(cmds[0], "echo") {
		t.Errorf("first command is %q, want echo", cmds[0])
	}
	if !strings.HasSuffix(cmds[1], "grep") {
		t.Errorf("second command is %q, want grep", cmds[1])
	}
}

// TestExtractShellSubCommands_Mixed verifies a complex mixed script.
func TestExtractShellSubCommands_Mixed(t *testing.T) {
	cmds := extractShellSubCommands(`echo "hello" | grep "world"; ls -la`, "")
	// Should find: echo, grep, ls
	if len(cmds) != 3 {
		t.Fatalf("expected 3 commands, got %d: %v", len(cmds), cmds)
	}
	if !strings.HasSuffix(cmds[0], "echo") {
		t.Errorf("first command is %q, want echo", cmds[0])
	}
	if !strings.HasSuffix(cmds[1], "grep") {
		t.Errorf("second command is %q, want grep", cmds[1])
	}
	if !strings.HasSuffix(cmds[2], "ls") {
		t.Errorf("third command is %q, want ls", cmds[2])
	}
}

// TestExtractShellSubCommands_Heredoc verifies heredocs are handled.
func TestExtractShellSubCommands_Heredoc(t *testing.T) {
	cmds := extractShellSubCommands(`cat <<EOF
foo bar
EOF
`, "")
	// Should find: cat
	if len(cmds) != 1 {
		t.Fatalf("expected 1 command, got %d: %v", len(cmds), cmds)
	}
	if !strings.HasSuffix(cmds[0], "cat") {
		t.Errorf("command is %q, want cat", cmds[0])
	}
}

// TestExtractShellSubCommands_Fallback verifies that parse failures fall back
// to the regex-based approach.
func TestExtractShellSubCommands_Fallback(t *testing.T) {
	// A script that might fail to parse (malformed).
	// The fallback should still extract something sensible.
	cmds := extractShellSubCommands(`echo broken |`, "")
	// At minimum should find echo
	if len(cmds) == 0 {
		t.Fatalf("expected at least one command from fallback, got %v", cmds)
	}
	if !strings.HasSuffix(cmds[0], "echo") {
		t.Errorf("command is %q, want echo", cmds[0])
	}
}

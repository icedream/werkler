package tools

import (
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

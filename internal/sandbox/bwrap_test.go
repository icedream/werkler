package sandbox_test

import (
	"strings"
	"testing"

	"github.com/icedream/werkler/internal/sandbox"
)

func TestWrapCommand_NilConfig(t *testing.T) {
	cmd, args := sandbox.WrapCommand(nil, "/usr/bin/git", []string{"status"}, "/tmp/repo")
	if cmd != "/usr/bin/git" {
		t.Errorf("expected /usr/bin/git, got %q", cmd)
	}
	if len(args) != 1 || args[0] != "status" {
		t.Errorf("expected [status], got %v", args)
	}
}

func TestWrapCommand_BasicWrapping(t *testing.T) {
	cfg := &sandbox.BwrapConfig{}
	cmd, args := sandbox.WrapCommand(cfg, "/usr/bin/ls", []string{"-la"}, "/tmp/test")

	if cmd != "bwrap" {
		t.Fatalf("expected bwrap, got %q", cmd)
	}

	joined := strings.Join(args, " ")

	// Should include network isolation by default.
	if !strings.Contains(joined, "--unshare-net") {
		t.Error("expected --unshare-net in args")
	}
	// Should include PID namespace isolation.
	if !strings.Contains(joined, "--unshare-pid") {
		t.Error("expected --unshare-pid in args")
	}
	// Should bind-mount the cwd read-write.
	if !strings.Contains(joined, "--bind /tmp/test /tmp/test") {
		t.Error("expected cwd bind mount in args")
	}
	// Should chdir into cwd.
	if !strings.Contains(joined, "--chdir /tmp/test") {
		t.Error("expected --chdir in args")
	}
	// Original command and args should appear after --.
	sepIdx := indexOf(args, "--")
	if sepIdx < 0 {
		t.Fatal("expected -- separator in bwrap args")
	}
	if args[sepIdx+1] != "/usr/bin/ls" {
		t.Errorf("expected command after --, got %q", args[sepIdx+1])
	}
	if len(args) <= sepIdx+2 || args[sepIdx+2] != "-la" {
		t.Errorf("expected -la after command, got %v", args[sepIdx+2:])
	}
}

func TestWrapCommand_AllowNetwork(t *testing.T) {
	cfg := &sandbox.BwrapConfig{AllowNetwork: true}
	_, args := sandbox.WrapCommand(cfg, "echo", []string{"hi"}, "/tmp")

	for _, a := range args {
		if a == "--unshare-net" {
			t.Error("unexpected --unshare-net when AllowNetwork=true")
		}
	}
}

func TestWrapCommand_ExtraMounts(t *testing.T) {
	cfg := &sandbox.BwrapConfig{
		ExtraROBinds: []string{"/opt/tools"},
		ExtraRWBinds: []string{"/var/data"},
	}
	_, args := sandbox.WrapCommand(cfg, "echo", nil, "/tmp")
	joined := strings.Join(args, " ")

	if !strings.Contains(joined, "--ro-bind /opt/tools /opt/tools") {
		t.Error("expected ExtraROBinds in args")
	}
	if !strings.Contains(joined, "--bind /var/data /var/data") {
		t.Error("expected ExtraRWBinds in args")
	}
}

func TestWrapCommand_NonSystemBinary(t *testing.T) {
	cfg := &sandbox.BwrapConfig{}
	// A binary outside system dirs should have its parent dir RO-bound.
	cmd, args := sandbox.WrapCommand(cfg, "/home/user/mybin", []string{}, "/tmp/work")
	if cmd != "bwrap" {
		t.Fatal("expected bwrap")
	}
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--ro-bind-try /home/user /home/user") {
		t.Error("expected binary parent directory ro-bind in args")
	}
}

func indexOf(slice []string, s string) int {
	for i, v := range slice {
		if v == s {
			return i
		}
	}
	return -1
}

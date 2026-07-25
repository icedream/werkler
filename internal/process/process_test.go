package process

import (
	"strings"
	"testing"
	"time"
)

// ---- stripANSI unit tests ----

func TestStripANSI(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"plain text unchanged", "hello world", "hello world"},
		{"SGR reset", "\x1b[0mhello\x1b[0m", "hello"},
		{"bold + color", "\x1b[1;31mERROR\x1b[0m: bad", "ERROR: bad"},
		{"cursor move", "\x1b[2Jcleared", "cleared"},
		{"nested sequences", "\x1b[1m\x1b[32mok\x1b[0m", "ok"},
		{"empty input", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := string(stripANSI([]byte(tc.input)))
			if got != tc.want {
				t.Errorf("stripANSI(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

// ---- process.write / readClean unit tests ----

func TestProcessWriteReadClean(t *testing.T) {
	p := &process{done: make(chan struct{})}
	_ = p.write([]byte("hello \x1b[1;31mworld\x1b[0m\n"))

	data, newOff := p.readClean(0)
	got := string(data)
	if got != "hello world\n" {
		t.Errorf("readClean(0) = %q, want %q", got, "hello world\n")
	}
	if newOff != len("hello world\n") {
		t.Errorf("newOff = %d, want %d", newOff, len("hello world\n"))
	}

	// Reading again from same offset returns no new data.
	data2, _ := p.readClean(newOff)
	if len(data2) != 0 {
		t.Errorf("expected no new data at offset %d, got %q", newOff, data2)
	}
}

// ---- Manager integration tests ----

func TestManagerStart_EchoReadsOutput(t *testing.T) {
	mgr := New(nil)
	ctx := t.Context()

	handle, err := mgr.Start(ctx, "echo", []string{"hello process"}, "", nil, false)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}

	out, newOff, running, exitCode, err := mgr.ReadNewOutput(handle, 0, 2*time.Second)
	if err != nil {
		t.Fatalf("ReadNewOutput: %v", err)
	}
	if running {
		t.Error("echo should have exited by now")
	}
	if exitCode != 0 {
		t.Errorf("exit code = %d, want 0", exitCode)
	}
	if !strings.Contains(out, "hello process") {
		t.Errorf("output %q does not contain %q", out, "hello process")
	}
	if newOff == 0 {
		t.Error("expected non-zero new offset")
	}
}

func TestManagerStart_CatSendRead(t *testing.T) {
	mgr := New(nil)
	ctx := t.Context()

	handle, err := mgr.Start(ctx, "cat", nil, "", nil, false)
	if err != nil {
		t.Fatalf("Start cat: %v", err)
	}
	t.Cleanup(func() { _, _, _ = mgr.Stop(handle, true) })

	if err := mgr.Send(handle, "ping\n"); err != nil {
		t.Fatalf("Send: %v", err)
	}

	out, _, _, _, err := mgr.ReadNewOutput(handle, 0, 2*time.Second)
	if err != nil {
		t.Fatalf("ReadNewOutput: %v", err)
	}
	if !strings.Contains(out, "ping") {
		t.Errorf("output %q does not contain %q", out, "ping")
	}
}

func TestManagerSendKey_ValidKey(t *testing.T) {
	mgr := New(nil)
	ctx := t.Context()

	handle, err := mgr.Start(ctx, "cat", nil, "", nil, false)
	if err != nil {
		t.Fatalf("Start cat: %v", err)
	}
	t.Cleanup(func() { _, _, _ = mgr.Stop(handle, true) })

	// SendKey should succeed for any well-known key name.
	if err := mgr.SendKey(handle, "ctrl+c"); err != nil {
		t.Errorf("SendKey ctrl+c: %v", err)
	}
	if err := mgr.SendKey(handle, "Enter"); err != nil {
		t.Errorf("SendKey Enter: %v", err)
	}
}

func TestManagerSendKey_InvalidKey(t *testing.T) {
	mgr := New(nil)
	ctx := t.Context()

	handle, err := mgr.Start(ctx, "cat", nil, "", nil, false)
	if err != nil {
		t.Fatalf("Start cat: %v", err)
	}
	t.Cleanup(func() { _, _, _ = mgr.Stop(handle, true) })

	err = mgr.SendKey(handle, "notakey")
	if err == nil {
		t.Error("expected error for unknown key, got nil")
	}
}

func TestManagerStop_ForceKill(t *testing.T) {
	mgr := New(nil)
	ctx := t.Context()

	// sleep sleeps indefinitely until killed.
	handle, err := mgr.Start(ctx, "sleep", []string{"60"}, "", nil, false)
	if err != nil {
		t.Fatalf("Start sleep: %v", err)
	}

	exitCode, _, err := mgr.Stop(handle, true)
	if err != nil {
		t.Fatalf("Stop: %v", err)
	}
	// Killed by signal → non-zero exit code.
	if exitCode == 0 {
		t.Error("expected non-zero exit code after force-kill")
	}
	// After Stop, the handle should be gone.
	if _, err := mgr.get(handle); err == nil {
		t.Error("handle still present after Stop")
	}
}

func TestManagerUnknownHandle(t *testing.T) {
	mgr := New(nil)
	_, _, _, err := mgr.ReadOutput("doesnotexist", 0)
	if err == nil {
		t.Error("expected error for unknown handle, got nil")
	}
	err = mgr.Send("doesnotexist", "data")
	if err == nil {
		t.Error("expected error for unknown handle, got nil")
	}
}

func TestManagerList(t *testing.T) {
	mgr := New(nil)
	ctx := t.Context()

	handle, err := mgr.Start(ctx, "sleep", []string{"60"}, "", nil, false)
	if err != nil {
		t.Fatalf("Start sleep: %v", err)
	}
	t.Cleanup(func() { _, _, _ = mgr.Stop(handle, true) })

	infos := mgr.List()
	found := false
	for _, info := range infos {
		if info.Handle == handle {
			found = true
			if !info.Running {
				t.Error("process should be running")
			}
			if info.Command != "sleep" {
				t.Errorf("command = %q, want %q", info.Command, "sleep")
			}
		}
	}
	if !found {
		t.Errorf("handle %q not found in List()", handle)
	}
}

func TestManagerOutputNotification(t *testing.T) {
	notified := make(chan string, 16)
	mgr := New(func(handle, raw, clean string) {
		notified <- clean
	})
	ctx := t.Context()

	handle, err := mgr.Start(ctx, "echo", []string{"notify me"}, "", nil, false)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _, _, _ = mgr.Stop(handle, true) })

	// Wait for notification or timeout.
	select {
	case msg := <-notified:
		if !strings.Contains(msg, "notify me") {
			t.Errorf("notification %q does not contain %q", msg, "notify me")
		}
	case <-time.After(2 * time.Second):
		t.Error("timed out waiting for output notification")
	}
}

func TestManagerPTY_Smoke(t *testing.T) {
	mgr := New(nil)
	ctx := t.Context()

	handle, err := mgr.Start(ctx, "echo", []string{"pty test"}, "", nil, true)
	if err != nil {
		t.Skipf("PTY not available: %v", err)
	}

	out, _, _, err := mgr.ReadOutput(handle, 2*time.Second)
	if err != nil {
		t.Fatalf("ReadOutput: %v", err)
	}
	if !strings.Contains(out, "pty test") {
		t.Errorf("PTY output %q does not contain %q", out, "pty test")
	}
}

func TestManagerReadNewOutput_Incremental(t *testing.T) {
	mgr := New(nil)
	ctx := t.Context()

	handle, err := mgr.Start(ctx, "cat", nil, "", nil, false)
	if err != nil {
		t.Fatalf("Start cat: %v", err)
	}
	t.Cleanup(func() { _, _, _ = mgr.Stop(handle, true) })

	// Send first chunk.
	if err := mgr.Send(handle, "first\n"); err != nil {
		t.Fatalf("Send: %v", err)
	}
	out1, off1, _, _, err := mgr.ReadNewOutput(handle, 0, 2*time.Second)
	if err != nil {
		t.Fatalf("ReadNewOutput(0): %v", err)
	}
	if !strings.Contains(out1, "first") {
		t.Errorf("first chunk %q missing %q", out1, "first")
	}

	// Send second chunk; read from the new offset.
	if err := mgr.Send(handle, "second\n"); err != nil {
		t.Fatalf("Send second: %v", err)
	}
	out2, _, _, _, err := mgr.ReadNewOutput(handle, off1, 2*time.Second)
	if err != nil {
		t.Fatalf("ReadNewOutput(off1): %v", err)
	}
	if !strings.Contains(out2, "second") {
		t.Errorf("second chunk %q missing %q", out2, "second")
	}
	if strings.Contains(out2, "first") {
		t.Errorf("second chunk %q should not contain first chunk's data", out2)
	}
}

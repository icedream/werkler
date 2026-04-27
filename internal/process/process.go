// Package process manages long-lived subprocesses that the AI can interact with.
// Each process can run under a PTY (pseudo-terminal) for programs that require
// real terminal behaviour, or under plain pipes for clean byte-stream I/O.
// Output is buffered so callers can read incrementally via an offset cursor.
package process

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/creack/pty"
)

// keySequences maps human-readable key names to the byte sequences sent to a PTY.
var keySequences = map[string]string{
	"enter":     "\r",
	"return":    "\r",
	"newline":   "\n",
	"tab":       "\t",
	"escape":    "\x1b",
	"esc":       "\x1b",
	"backspace": "\x7f",
	"delete":    "\x1b[3~",
	"home":      "\x1b[H",
	"end":       "\x1b[F",
	"page_up":   "\x1b[5~",
	"page_down": "\x1b[6~",
	"up":        "\x1b[A",
	"down":      "\x1b[B",
	"right":     "\x1b[C",
	"left":      "\x1b[D",
	"ctrl+a":    "\x01",
	"ctrl+b":    "\x02",
	"ctrl+c":    "\x03",
	"ctrl+d":    "\x04",
	"ctrl+e":    "\x05",
	"ctrl+f":    "\x06",
	"ctrl+g":    "\x07",
	"ctrl+h":    "\x08",
	"ctrl+i":    "\x09",
	"ctrl+j":    "\x0a",
	"ctrl+k":    "\x0b",
	"ctrl+l":    "\x0c",
	"ctrl+m":    "\x0d",
	"ctrl+n":    "\x0e",
	"ctrl+o":    "\x0f",
	"ctrl+p":    "\x10",
	"ctrl+q":    "\x11",
	"ctrl+r":    "\x12",
	"ctrl+s":    "\x13",
	"ctrl+t":    "\x14",
	"ctrl+u":    "\x15",
	"ctrl+v":    "\x16",
	"ctrl+w":    "\x17",
	"ctrl+x":    "\x18",
	"ctrl+y":    "\x19",
	"ctrl+z":    "\x1a",
	"f1":        "\x1bOP",
	"f2":        "\x1bOQ",
	"f3":        "\x1bOR",
	"f4":        "\x1bOS",
	"f5":        "\x1b[15~",
	"f6":        "\x1b[17~",
	"f7":        "\x1b[18~",
	"f8":        "\x1b[19~",
	"f9":        "\x1b[20~",
	"f10":       "\x1b[21~",
	"f11":       "\x1b[23~",
	"f12":       "\x1b[24~",
}

// OutputNotification is called (non-blocking) whenever new output arrives.
// rawOutput is the unmodified PTY/pipe bytes; cleanOutput has ANSI stripped.
type OutputNotification func(handle, rawOutput, cleanOutput string)

// process holds the state of a single running subprocess.
type process struct {
	mu       sync.Mutex
	cmd      *exec.Cmd
	pty      bool
	ptmx     *os.File  // PTY master (nil when pty==false)
	stdin    io.Writer // pipe stdin (nil when pty==true)
	rawBuf   bytes.Buffer
	cleanBuf bytes.Buffer
	running  bool
	exitCode int
	done     chan struct{}
}

func (p *process) write(b []byte) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.rawBuf.Write(b)
	// Strip ANSI for the clean buffer.
	clean := stripANSI(b)
	p.cleanBuf.Write(clean)
	return nil
}

func (p *process) readClean(offset int) ([]byte, int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	all := p.cleanBuf.Bytes()
	if offset >= len(all) {
		return nil, len(all)
	}
	out := make([]byte, len(all)-offset)
	copy(out, all[offset:])
	return out, len(all)
}

// ansiEscape matches ANSI/VT100 escape sequences.
var ansiEscape = regexp.MustCompile(`\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])`)

// stripANSI removes ANSI escape sequences from b.
func stripANSI(b []byte) []byte {
	return ansiEscape.ReplaceAll(b, nil)
}

// ProcessInfo describes a running or finished process.
type ProcessInfo struct {
	Handle   string
	Command  string
	Args     []string
	Running  bool
	ExitCode int
	PTY      bool
}

// Manager maintains a set of active subprocesses keyed by handle.
type Manager struct {
	mu     sync.Mutex
	procs  map[string]*process
	infos  map[string]ProcessInfo
	notify OutputNotification
}

// New creates an empty Manager. notify (may be nil) is called whenever new
// output arrives from any process; it runs in the output-reader goroutine so
// the callback must not block.
func New(notify OutputNotification) *Manager {
	return &Manager{
		procs:  make(map[string]*process),
		infos:  make(map[string]ProcessInfo),
		notify: notify,
	}
}

// SetNotify replaces the output notification callback. Safe to call at any
// time; new output from all processes will use the updated callback.
func (m *Manager) SetNotify(fn OutputNotification) {
	m.mu.Lock()
	m.notify = fn
	m.mu.Unlock()
}

// Start launches a subprocess. command must be an absolute path or a name
// resolvable via PATH. args are the argv (without the command itself). cwd is
// the working directory; "" uses the current directory. env is merged on top
// of the current process environment. usePTY allocates a PTY.
// Returns a unique handle identifying the process.
func (m *Manager) Start(
	ctx context.Context,
	command string, args []string,
	cwd string, env map[string]string,
	usePTY bool,
) (string, error) {
	cmd := exec.CommandContext(ctx, command, args...)
	if cwd != "" {
		cmd.Dir = cwd
	}
	cmd.Env = os.Environ()
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}

	p := &process{
		cmd:     cmd,
		pty:     usePTY,
		running: true,
		done:    make(chan struct{}),
	}

	if usePTY {
		ptmx, err := pty.Start(cmd)
		if err != nil {
			return "", fmt.Errorf("starting process with PTY: %w", err)
		}
		p.ptmx = ptmx
	} else {
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			return "", fmt.Errorf("creating stdout pipe: %w", err)
		}
		cmd.Stderr = cmd.Stdout // merge stderr into stdout pipe
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return "", fmt.Errorf("creating stdin pipe: %w", err)
		}
		p.stdin = stdin
		if err := cmd.Start(); err != nil {
			return "", fmt.Errorf("starting process: %w", err)
		}
		// Read from the merged stdout/stderr pipe.
		go m.readLoop(p, stdout)
	}

	handle := newHandle()

	m.mu.Lock()
	m.procs[handle] = p
	m.infos[handle] = ProcessInfo{
		Handle:  handle,
		Command: command,
		Args:    args,
		Running: true,
		PTY:     usePTY,
	}
	m.mu.Unlock()

	if usePTY {
		go m.readLoop(p, p.ptmx)
	}
	go waitLoop(m, handle, p)

	return handle, nil
}

// readLoop drains r into the process output buffer until EOF.
func (m *Manager) readLoop(p *process, r io.Reader) {
	buf := make([]byte, 4096)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			_ = p.write(chunk)
			m.mu.Lock()
			notify := m.notify
			handle := ""
			for h, proc := range m.procs {
				if proc == p {
					handle = h
					break
				}
			}
			m.mu.Unlock()
			if notify != nil && handle != "" {
				raw := string(chunk)
				clean := string(stripANSI(chunk))
				notify(handle, raw, clean)
			}
		}
		if err != nil {
			return
		}
	}
}

// waitLoop waits for the process to exit and updates state.
func waitLoop(m *Manager, handle string, p *process) {
	err := p.cmd.Wait()
	p.mu.Lock()
	p.running = false
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			p.exitCode = exitErr.ExitCode()
		} else {
			p.exitCode = -1
		}
	}
	p.mu.Unlock()

	if p.ptmx != nil {
		_ = p.ptmx.Close()
	}

	m.mu.Lock()
	if info, ok := m.infos[handle]; ok {
		info.Running = false
		info.ExitCode = p.exitCode
		m.infos[handle] = info
	}
	m.mu.Unlock()
	close(p.done)
}

// Send writes text to the process stdin / PTY master.
func (m *Manager) Send(handle, text string) error {
	p, err := m.get(handle)
	if err != nil {
		return err
	}
	w := m.writer(p)
	if w == nil {
		return fmt.Errorf("process %s has no writable input", handle)
	}
	_, err = io.WriteString(w, text)
	return err
}

// SendKey sends a named key sequence to the process.
// The key name is case-insensitive (e.g. "ctrl+c", "Enter", "up").
func (m *Manager) SendKey(handle, key string) error {
	seq, ok := keySequences[strings.ToLower(key)]
	if !ok {
		return fmt.Errorf("unknown key %q (use names like enter, ctrl+c, up, escape)", key)
	}
	return m.Send(handle, seq)
}

// ReadOutput returns new output since offset (in raw bytes of the clean buffer).
// If the process has no new output within timeout, returns what is available.
// Returns (cleanOutput, newOffset, running, exitCode, error).
func (m *Manager) ReadOutput(handle string, timeout time.Duration) (string, bool, int, error) {
	p, err := m.get(handle)
	if err != nil {
		return "", false, -1, err
	}

	// Wait briefly for new output if the process is running.
	if timeout > 0 {
		select {
		case <-p.done:
		case <-time.After(timeout):
		}
	}

	p.mu.Lock()
	all := p.cleanBuf.Bytes()
	out := make([]byte, len(all))
	copy(out, all)
	running := p.running
	exitCode := p.exitCode
	p.mu.Unlock()

	return string(out), running, exitCode, nil
}

// ReadNewOutput returns only output that arrived after lastOffset bytes.
// Returns (newCleanOutput, newOffset, running, exitCode, error).
func (m *Manager) ReadNewOutput(handle string, lastOffset int, timeout time.Duration) (string, int, bool, int, error) {
	p, err := m.get(handle)
	if err != nil {
		return "", 0, false, -1, err
	}

	if timeout > 0 {
		deadline := time.Now().Add(timeout)
		for {
			data, newOff := p.readClean(lastOffset)
			if len(data) > 0 {
				p.mu.Lock()
				running := p.running
				exitCode := p.exitCode
				p.mu.Unlock()
				return string(data), newOff, running, exitCode, nil
			}
			p.mu.Lock()
			running := p.running
			p.mu.Unlock()
			if !running || time.Now().After(deadline) {
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
	}

	data, newOff := p.readClean(lastOffset)
	p.mu.Lock()
	running := p.running
	exitCode := p.exitCode
	p.mu.Unlock()
	return string(data), newOff, running, exitCode, nil
}

// Stop terminates a process. If force is true, SIGKILL is sent; otherwise SIGTERM.
// Returns the exit code and any final buffered output.
func (m *Manager) Stop(handle string, force bool) (int, string, error) {
	p, err := m.get(handle)
	if err != nil {
		return -1, "", err
	}

	p.mu.Lock()
	running := p.running
	p.mu.Unlock()

	if running {
		if force {
			_ = p.cmd.Process.Kill()
		} else {
			_ = p.cmd.Process.Signal(os.Interrupt)
		}
		// Give it a moment to exit gracefully before we read final output.
		select {
		case <-p.done:
		case <-time.After(3 * time.Second):
			_ = p.cmd.Process.Kill()
			<-p.done
		}
	}

	p.mu.Lock()
	exitCode := p.exitCode
	all := p.cleanBuf.Bytes()
	out := make([]byte, len(all))
	copy(out, all)
	p.mu.Unlock()

	// Remove from the active map.
	m.mu.Lock()
	delete(m.procs, handle)
	delete(m.infos, handle)
	m.mu.Unlock()

	return exitCode, string(out), nil
}

// List returns info about all known processes (running and recently stopped).
func (m *Manager) List() []ProcessInfo {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]ProcessInfo, 0, len(m.infos))
	for _, info := range m.infos {
		out = append(out, info)
	}
	return out
}

// get returns the process for handle, or an error if not found.
func (m *Manager) get(handle string) (*process, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	p, ok := m.procs[handle]
	if !ok {
		return nil, fmt.Errorf("no process with handle %q", handle)
	}
	return p, nil
}

// writer returns the appropriate write target for a process.
func (m *Manager) writer(p *process) io.Writer {
	if p.ptmx != nil {
		return p.ptmx
	}
	return p.stdin
}

// newHandle generates a short random process handle.
func newHandle() string {
	const chars = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	// Use time-based fallback; crypto/rand used via rand.Text pattern in Go 1.24+
	src := fmt.Sprintf("%016x", time.Now().UnixNano())
	for i := range b {
		idx := (int(src[i]) + i) % len(chars)
		b[i] = chars[idx]
	}
	return string(b)
}

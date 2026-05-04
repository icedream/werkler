package process

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/process"
	"github.com/icedream/werkler/internal/tools/toolutil"
)

const (
	// OutputCap is the maximum bytes captured per stream (stdout/stderr/combined).
	OutputCap = 512 * 1024
	// MaxTimeout is the maximum allowed timeout_seconds for run_command.
	MaxTimeout = 600.0
)

// Context is the narrow interface the process handler needs from the Manager.
type Context interface {
	// ProcessRegistry returns the live-process manager used by process_start etc.
	ProcessRegistry() *process.Manager
	// CheckWritePaths checks whether the executable and any path arguments in a
	// direct command invocation are approved for execution/write access.
	CheckWritePaths(ctx context.Context, command string, args []string, cwd string) []toolutil.PathAccessRequest
	// CheckShellCommandWritePaths checks paths in a shell command string,
	// skipping the shell interpreter itself (always trusted).
	CheckShellCommandWritePaths(ctx context.Context, shell, command, cwd string) []toolutil.PathAccessRequest
	// LiveOutputCh extracts a live-output channel from the context, if present.
	// Returns nil when live streaming is not enabled for this call.
	LiveOutputCh(ctx context.Context) chan<- string
}

// Handler holds the process/command tool handlers.
type Handler struct{ ctx Context }

// NewHandler creates a Handler.
func NewHandler(ctx Context) *Handler { return &Handler{ctx: ctx} }

// Tools returns Builtin definitions for all process and run_command tools.
func (h *Handler) Tools() []toolutil.Builtin {
	return []toolutil.Builtin{
		{Def: processStartDef, Handle: h.handleProcessStart},
		{Def: processSendDef, Handle: h.handleProcessSend},
		{Def: processSendKeyDef, Handle: h.handleProcessSendKey},
		{Def: processReadDef, Handle: h.handleProcessRead},
		{Def: processStopDef, Handle: h.handleProcessStop},
		{Def: runCommandDef, Handle: h.handleRunCommand},
	}
}

// ─── Schemas ─────────────────────────────────────────────────────────────────

var processStartDef = ai.ToolDefinition{
	Name: "process_start",
	Description: `Start a subprocess. Returns a handle for subsequent interaction.
Use pty=true for interactive programs (editors, REPLs, password prompts) — output will contain ANSI codes that are stripped before being returned to you.
Use pty=false (default) for non-interactive commands (builds, scripts) — output is clean plain text.

IMPORTANT: "command" is the executable only; "args" are the arguments WITHOUT the command name.
Example — to run "git status --short": command="git", args=["status", "--short"].
Do NOT put "git" (or any command name) in args.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command":         map[string]any{"type": "string", "description": "Absolute path or PATH-resolvable command name"},
			"args":            map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Command arguments only — do NOT include the command name as the first element (unlike argv[0])"},
			"title":           map[string]any{"type": "string", "description": `REQUIRED. Short human-readable phrase describing the purpose of this command (e.g. "Build project", "Run tests", "Search for TODO comments"). Shown to the user in the approval dialog and chat log.`},
			"cwd":             map[string]any{"type": "string", "description": "Working directory; empty = inherit werkler's cwd"},
			"pty":             map[string]any{"type": "boolean", "description": "Allocate a PTY (pseudo-terminal); required for interactive programs"},
			"env":             map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}, "description": "Extra environment variables merged on top of the current environment"},
			"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait for initial output before returning (0 = don't wait, default 5)"},
		},
		"required": []string{"command", "title"},
	},
}

var processSendDef = ai.ToolDefinition{
	Name:        "process_send",
	Description: "Send text to a running process's stdin / PTY.",
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"handle":          map[string]any{"type": "string", "description": "Process handle from process_start"},
			"text":            map[string]any{"type": "string", "description": "Text to write"},
			"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait for new output after sending (default 2)"},
		},
		"required": []string{"handle", "text"},
	},
}

var processSendKeyDef = ai.ToolDefinition{
	Name: "process_send_key",
	Description: `Send a named key to a running process.
Available keys: enter, tab, escape, backspace, delete, home, end, page_up, page_down,
up, down, left, right, ctrl+a through ctrl+z, f1-f12.`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"handle":          map[string]any{"type": "string", "description": "Process handle from process_start"},
			"key":             map[string]any{"type": "string", "description": `Key name, e.g. "enter", "ctrl+c", "up"`},
			"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait for new output after sending (default 2)"},
		},
		"required": []string{"handle", "key"},
	},
}

var processReadDef = ai.ToolDefinition{
	Name:        "process_read",
	Description: "Read output from a running process since the last read.",
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"handle":          map[string]any{"type": "string", "description": "Process handle from process_start"},
			"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait for new output (default 5)"},
		},
		"required": []string{"handle"},
	},
}

var processStopDef = ai.ToolDefinition{
	Name:        "process_stop",
	Description: "Stop a running process and collect its final output.",
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"handle": map[string]any{"type": "string", "description": "Process handle from process_start"},
			"force":  map[string]any{"type": "boolean", "description": "Use SIGKILL instead of SIGTERM (default false)"},
		},
		"required": []string{"handle"},
	},
}

var runCommandDef = ai.ToolDefinition{
	Name: "run_command",
	Description: `Run a command synchronously and return its output in a single call. No polling required.
Use this for one-shot non-interactive commands (builds, scripts, grep, CLI tools, etc.).
Do NOT use for interactive programs, programs that require a TTY, or long-running background processes -- use process_start for those.

IMPORTANT: "command" is the executable only; "args" are the arguments WITHOUT the command name.
Example -- to run "git status --short": command="git", args=["status", "--short"].
Do NOT put "git" (or any command name) in args.

SHELL MODE: Any time your command uses pipes (|), redirects (>, >>), logical operators (&&, ||),
subshells ($(...)), globs, or any other shell syntax, you MUST set shell=true and put the entire
command string in "command" with no "args".
Examples requiring shell=true: "ps aux | grep nginx", "cat /proc/loadavg", "free -m && uptime",
"ls *.go | wc -l", "top -bn1 | head -20".
Examples NOT requiring shell=true: "git", "grep", "uptime", "free", "ps" (with args instead).`,
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command":         map[string]any{"type": "string", "description": "Absolute path or PATH-resolvable command name. When shell=true, this is the full shell command string (e.g. \"ps aux | grep nginx\")."},
			"args":            map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Command arguments only -- do NOT include the command name. Must not be provided when shell=true."},
			"shell":           map[string]any{"type": "boolean", "description": "If true, run via bash -c. REQUIRED for any command containing |, &&, ||, >, >>, $(...), or other shell syntax."},
			"cwd":             map[string]any{"type": "string", "description": "Working directory; empty = inherit werkler's cwd"},
			"env":             map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}, "description": "Extra environment variables merged on top of the current environment"},
			"timeout_seconds": map[string]any{"type": "number", "description": "Seconds to wait before killing the process (default 30, max 600)"},
			"title":           map[string]any{"type": "string", "description": "REQUIRED. Short human-readable phrase describing what is being run. Shown in the approval dialog."},
		},
		"required": []string{"command", "title"},
	},
}

// ─── Handlers ────────────────────────────────────────────────────────────────

func jsonResult(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func (h *Handler) handleProcessStart(ctx context.Context, args map[string]any) (string, error) {
	command := toolutil.StringArg(args, "command")
	if command == "" {
		return "", fmt.Errorf("process_start: command is required")
	}
	cmdArgs := toolutil.StringSliceArg(args, "args")
	title := toolutil.StringArg(args, "title")
	if title == "" {
		base := filepath.Base(command)
		if len(cmdArgs) > 0 {
			title = base + " " + cmdArgs[0]
		} else {
			title = base
		}
		args["title"] = title
	}
	cwd := toolutil.StringArg(args, "cwd")
	usePTY := toolutil.BoolArg(args, "pty")
	env := toolutil.StringMapArg(args, "env")
	timeoutSecs := toolutil.Float64Arg(args, "timeout_seconds", 5)
	if unapproved := h.ctx.CheckWritePaths(ctx, command, cmdArgs, cwd); len(unapproved) > 0 {
		return "", &toolutil.UnapprovedPathsError{Requests: unapproved}
	}
	reg := h.ctx.ProcessRegistry()
	handle, err := reg.Start(ctx, command, cmdArgs, cwd, env, usePTY)
	if err != nil {
		return "", fmt.Errorf("process_start: %w", err)
	}
	timeout := time.Duration(timeoutSecs * float64(time.Second))
	output, _, _, err := reg.ReadOutput(handle, timeout)
	if err != nil {
		output = ""
	}
	return jsonResult(map[string]any{"handle": handle, "output": output}), nil
}

func (h *Handler) handleProcessSend(_ context.Context, args map[string]any) (string, error) {
	reg := h.ctx.ProcessRegistry()
	handle := toolutil.StringArg(args, "handle")
	text := toolutil.StringArg(args, "text")
	timeoutSecs := toolutil.Float64Arg(args, "timeout_seconds", 2)
	if err := reg.Send(handle, text); err != nil {
		return "", err
	}
	timeout := time.Duration(timeoutSecs * float64(time.Second))
	output, running, exitCode, err := reg.ReadOutput(handle, timeout)
	if err != nil {
		return "", err
	}
	return jsonResult(map[string]any{"output": output, "running": running, "exit_code": exitCode}), nil
}

func (h *Handler) handleProcessSendKey(_ context.Context, args map[string]any) (string, error) {
	reg := h.ctx.ProcessRegistry()
	handle := toolutil.StringArg(args, "handle")
	key := toolutil.StringArg(args, "key")
	timeoutSecs := toolutil.Float64Arg(args, "timeout_seconds", 2)
	if err := reg.SendKey(handle, key); err != nil {
		return "", err
	}
	timeout := time.Duration(timeoutSecs * float64(time.Second))
	output, running, exitCode, err := reg.ReadOutput(handle, timeout)
	if err != nil {
		return "", err
	}
	return jsonResult(map[string]any{"output": output, "running": running, "exit_code": exitCode}), nil
}

func (h *Handler) handleProcessRead(_ context.Context, args map[string]any) (string, error) {
	reg := h.ctx.ProcessRegistry()
	handle := toolutil.StringArg(args, "handle")
	timeoutSecs := toolutil.Float64Arg(args, "timeout_seconds", 5)
	timeout := time.Duration(timeoutSecs * float64(time.Second))
	output, running, exitCode, err := reg.ReadOutput(handle, timeout)
	if err != nil {
		return "", err
	}
	return jsonResult(map[string]any{"output": output, "running": running, "exit_code": exitCode}), nil
}

func (h *Handler) handleProcessStop(_ context.Context, args map[string]any) (string, error) {
	reg := h.ctx.ProcessRegistry()
	handle := toolutil.StringArg(args, "handle")
	force := toolutil.BoolArg(args, "force")
	exitCode, output, err := reg.Stop(handle, force)
	if err != nil {
		return "", err
	}
	return jsonResult(map[string]any{"exit_code": exitCode, "output": output}), nil
}

func (h *Handler) handleRunCommand(ctx context.Context, args map[string]any) (string, error) {
	command := toolutil.StringArg(args, "command")
	if command == "" {
		return jsonResult(map[string]any{"error": "run_command: command is required"}), nil
	}
	title := toolutil.StringArg(args, "title")
	if title == "" {
		title = command
		args["title"] = title
	}
	useShell := toolutil.BoolArg(args, "shell")
	cmdArgs := toolutil.StringSliceArg(args, "args")
	if useShell && len(cmdArgs) > 0 {
		cmdArgs = nil
	}
	if !useShell && strings.ContainsRune(command, ' ') {
		useShell = true
		cmdArgs = nil
	}
	cwd := toolutil.StringArg(args, "cwd")
	if cwd != "" {
		info, err := os.Stat(cwd)
		if err != nil {
			return jsonResult(map[string]any{"error": fmt.Sprintf("run_command: cwd does not exist: %s", cwd)}), nil
		}
		if !info.IsDir() {
			return jsonResult(map[string]any{"error": fmt.Sprintf("run_command: cwd is not a directory: %s", cwd)}), nil
		}
	}
	timeoutSecs := toolutil.Float64Arg(args, "timeout_seconds", 30)
	if timeoutSecs <= 0 {
		timeoutSecs = 30
	}
	if timeoutSecs > MaxTimeout {
		timeoutSecs = MaxTimeout
	}
	// Build final env.
	baseEnv := os.Environ()
	var finalEnv []string
	if raw, ok := args["env"].(map[string]any); ok && len(raw) > 0 {
		envMap := make(map[string]string, len(baseEnv))
		for _, kv := range baseEnv {
			if idx := strings.IndexByte(kv, '='); idx >= 0 {
				envMap[kv[:idx]] = kv[idx+1:]
			}
		}
		for k, v := range raw {
			if v == nil {
				delete(envMap, k)
			} else if s, ok := v.(string); ok {
				envMap[k] = s
			}
		}
		finalEnv = make([]string, 0, len(envMap))
		for k, v := range envMap {
			finalEnv = append(finalEnv, k+"="+v)
		}
	} else {
		finalEnv = baseEnv
	}
	// Build exec.Cmd.
	var cmd *exec.Cmd
	if useShell {
		cmd = exec.Command("bash", "-c", command)
	} else {
		cmd = exec.Command(command, cmdArgs...)
	}
	cmd.Env = finalEnv
	if cwd != "" {
		cmd.Dir = cwd
	}
	devNull, err := os.Open(os.DevNull)
	if err == nil {
		cmd.Stdin = devNull
		defer func() { _ = devNull.Close() }()
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	// Path approval.
	if useShell {
		if unapproved := h.ctx.CheckShellCommandWritePaths(ctx, "bash", command, cwd); len(unapproved) > 0 {
			return "", &toolutil.UnapprovedPathsError{Requests: unapproved}
		}
	} else {
		if unapproved := h.ctx.CheckWritePaths(ctx, command, cmdArgs, cwd); len(unapproved) > 0 {
			return "", &toolutil.UnapprovedPathsError{Requests: unapproved}
		}
	}
	// Output capture with optional live streaming.
	stdoutBuf := newCapBuffer(OutputCap)
	stderrBuf := newCapBuffer(OutputCap)
	combinedBuf := newCapBuffer(OutputCap)
	combinedMu := &sync.Mutex{}
	var stdoutLive, stderrLive *liveLineWriter
	if liveCh := h.ctx.LiveOutputCh(ctx); liveCh != nil {
		stdoutLive = &liveLineWriter{inner: &combinedWriter{stream: stdoutBuf, combined: combinedBuf, mu: combinedMu}, liveCh: liveCh}
		stderrLive = &liveLineWriter{inner: &combinedWriter{stream: stderrBuf, combined: combinedBuf, mu: combinedMu}, liveCh: liveCh}
		cmd.Stdout = stdoutLive
		cmd.Stderr = stderrLive
	} else {
		cmd.Stdout = &combinedWriter{stream: stdoutBuf, combined: combinedBuf, mu: combinedMu}
		cmd.Stderr = &combinedWriter{stream: stderrBuf, combined: combinedBuf, mu: combinedMu}
	}
	if startErr := cmd.Start(); startErr != nil {
		return jsonResult(map[string]any{"error": fmt.Sprintf("run_command: %s", startErr.Error())}), nil
	}
	type waitResult struct{ exitCode int }
	done := make(chan waitResult, 1)
	go func() {
		err := cmd.Wait()
		code := 0
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				code = exitErr.ExitCode()
			} else {
				code = -1
			}
		}
		done <- waitResult{code}
	}()
	timeout := time.Duration(timeoutSecs * float64(time.Second))
	timedOut := false
	var exitCode int
	select {
	case result := <-done:
		exitCode = result.exitCode
		if stdoutLive != nil {
			stdoutLive.flush()
		}
		if stderrLive != nil {
			stderrLive.flush()
		}
	case <-time.After(timeout):
		timedOut = true
		if cmd.Process != nil {
			pgid := -cmd.Process.Pid
			_ = syscall.Kill(pgid, syscall.SIGTERM)
			select {
			case result := <-done:
				exitCode = result.exitCode
			case <-time.After(2 * time.Second):
				_ = syscall.Kill(pgid, syscall.SIGKILL)
				result := <-done
				exitCode = result.exitCode
			}
		}
	}
	return jsonResult(map[string]any{
		"exit_code":       exitCode,
		"stdout":          stdoutBuf.String(),
		"stderr":          stderrBuf.String(),
		"combined_output": combinedBuf.String(),
		"timed_out":       timedOut,
	}), nil
}

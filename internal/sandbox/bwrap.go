package sandbox

import (
	"os/exec"
)

// BwrapAvailable returns true if bubblewrap (bwrap) is available on PATH.
// Returns false on non-Linux platforms or if bwrap is not installed.
func BwrapAvailable() bool {
	_, err := exec.LookPath("bwrap")
	return err == nil
}

// BwrapConfig holds options for wrapping a process with bubblewrap.
type BwrapConfig struct {
	// ReadOnlyBinds are host paths bind-mounted read-only at the same location
	// inside the sandbox (e.g. "/home/user/project" → "/home/user/project").
	ReadOnlyBinds []string
	// ReadWriteBinds are host paths bind-mounted read-write at the same location.
	ReadWriteBinds []string
	// UnshareNet disables network access inside the sandbox.
	UnshareNet bool
	// UnshareIPC isolates the IPC namespace.
	UnshareIPC bool
	// Hostname sets a custom hostname inside the sandbox.
	// When non-empty the UTS namespace is also unshared.
	Hostname string
}

// DefaultBwrapConfig returns a conservative starting config: network and IPC
// are both isolated. Callers add bind mounts for paths they need.
func DefaultBwrapConfig() BwrapConfig {
	return BwrapConfig{
		UnshareNet: true,
		UnshareIPC: true,
	}
}

// WrapCommand converts (originalCmd, originalArgs) into an equivalent
// (bwrap, bwrapArgs) invocation that runs the original command inside a
// bubblewrap sandbox.
//
// The sandbox always provides:
//   - /usr, /bin (and /lib*, /sbin if present) from the host — read-only
//   - Minimal /etc stubs (resolv.conf, passwd, group) — read-only
//   - Fresh /tmp (tmpfs), /proc, and /dev
//   - --die-with-parent so the sandbox exits if Werkler crashes
//
// Additional read-only or read-write mounts are applied from cfg.
// Paths are bound at their original host locations so tool output (error
// messages, file paths) is identical to running without the sandbox.
func WrapCommand(originalCmd string, originalArgs []string, cfg BwrapConfig) (cmd string, args []string) {
	bwrapArgs := []string{
		// Essential system paths — read-only.
		"--ro-bind", "/usr", "/usr",
		"--ro-bind-try", "/lib", "/lib",
		"--ro-bind-try", "/lib64", "/lib64",
		"--ro-bind-try", "/lib32", "/lib32",
		"--ro-bind", "/bin", "/bin",
		"--ro-bind-try", "/sbin", "/sbin",

		// Minimal /etc stubs needed by common tools.
		"--ro-bind-try", "/etc/resolv.conf", "/etc/resolv.conf",
		"--ro-bind-try", "/etc/passwd", "/etc/passwd",
		"--ro-bind-try", "/etc/group", "/etc/group",
		"--ro-bind-try", "/etc/hosts", "/etc/hosts",
		"--ro-bind-try", "/etc/ssl/certs", "/etc/ssl/certs",

		// Fresh namespaced filesystems.
		"--tmpfs", "/tmp",
		"--proc", "/proc",
		"--dev", "/dev",

		// Orphan prevention.
		"--die-with-parent",
	}

	if cfg.UnshareNet {
		bwrapArgs = append(bwrapArgs, "--unshare-net")
	}
	if cfg.UnshareIPC {
		bwrapArgs = append(bwrapArgs, "--unshare-ipc")
	}
	if cfg.Hostname != "" {
		bwrapArgs = append(bwrapArgs, "--unshare-uts", "--hostname", cfg.Hostname)
	}

	for _, p := range cfg.ReadOnlyBinds {
		bwrapArgs = append(bwrapArgs, "--ro-bind", p, p)
	}
	for _, p := range cfg.ReadWriteBinds {
		bwrapArgs = append(bwrapArgs, "--bind", p, p)
	}

	// -- separates bwrap options from the sandboxed command.
	bwrapArgs = append(bwrapArgs, "--")
	bwrapArgs = append(bwrapArgs, originalCmd)
	bwrapArgs = append(bwrapArgs, originalArgs...)

	return "bwrap", bwrapArgs
}

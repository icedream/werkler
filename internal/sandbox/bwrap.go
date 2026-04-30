package sandbox

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// BwrapConfig controls process-level sandboxing via bubblewrap (bwrap).
// A nil *BwrapConfig means sandboxing is disabled.
type BwrapConfig struct {
	// AllowNetwork permits outbound network access inside the sandbox.
	// When false (default), --unshare-net is passed to bwrap.
	AllowNetwork bool

	// ExtraROBinds are additional host paths bind-mounted read-only at the
	// same location inside the sandbox.
	ExtraROBinds []string

	// ExtraRWBinds are additional host paths bind-mounted read-write at the
	// same location inside the sandbox.
	ExtraRWBinds []string
}

// BwrapAvailable returns true if bubblewrap (bwrap) is available on PATH.
func BwrapAvailable() bool {
	_, err := exec.LookPath("bwrap")
	return err == nil
}

// systemPrefixes are root tree prefixes that WrapCommand mounts from the host
// read-only. Binaries and libraries that live under these prefixes are already
// reachable without extra bind mounts.
var systemPrefixes = []string{
	"/usr", "/bin", "/lib", "/lib64", "/lib32", "/sbin",
	"/etc", "/dev", "/proc", "/sys", "/tmp",
}

// isSystemPath reports whether p lives inside a standard system tree.
func isSystemPath(p string) bool {
	for _, prefix := range systemPrefixes {
		if p == prefix || strings.HasPrefix(p, prefix+"/") {
			return true
		}
	}
	return false
}

// WrapCommand converts (originalCmd, originalArgs, cwd) into an equivalent
// bwrap invocation that runs the original command inside a minimal sandbox.
//
// When cfg is nil the original cmd and args are returned unchanged.
//
// The sandbox always provides:
//   - /usr, /bin, /lib*, /sbin, /etc stubs from the host — read-only
//   - Fresh /tmp (tmpfs), /proc, /dev
//   - cwd bind-mounted read-write so tool output lands on the host
//   - --chdir to cwd so relative paths behave identically
//   - --die-with-parent so the sandbox exits if Werkler crashes
//
// If originalCmd is an absolute path outside standard system directories its
// parent directory is also bind-mounted read-only (e.g. a Go binary in
// /home/user/go/bin/).
func WrapCommand(cfg *BwrapConfig, originalCmd string, originalArgs []string, cwd string) (cmd string, args []string) {
	if cfg == nil {
		return originalCmd, originalArgs
	}

	if cwd == "" {
		cwd, _ = os.Getwd()
	}

	bwrapArgs := []string{
		// Namespace isolation.
		"--unshare-pid",
		"--unshare-ipc",
		"--unshare-uts",
		"--unshare-cgroup-try",
	}

	if !cfg.AllowNetwork {
		bwrapArgs = append(bwrapArgs, "--unshare-net")
	}

	// Read-only system trees — use --ro-bind-try so missing paths are silently skipped.
	for _, d := range []string{"/usr", "/bin", "/lib", "/lib64", "/lib32", "/sbin"} {
		bwrapArgs = append(bwrapArgs, "--ro-bind-try", d, d)
	}
	// Minimal /etc stubs.
	for _, f := range []string{
		"/etc/resolv.conf", "/etc/passwd", "/etc/group",
		"/etc/hosts", "/etc/ssl/certs", "/etc/ld.so.cache",
		"/etc/ld.so.conf", "/etc/ld.so.conf.d",
	} {
		bwrapArgs = append(bwrapArgs, "--ro-bind-try", f, f)
	}

	// Pseudo-filesystems.
	bwrapArgs = append(bwrapArgs,
		"--proc", "/proc",
		"--dev", "/dev",
		"--tmpfs", "/tmp",
	)

	// Working directory — read-write so the process can create files there.
	bwrapArgs = append(bwrapArgs, "--bind", cwd, cwd)

	// If the binary is an absolute path outside the system tree, ensure its
	// directory is visible. Skip if it's already covered by the cwd bind.
	if filepath.IsAbs(originalCmd) && !isSystemPath(originalCmd) {
		binDir := filepath.Dir(originalCmd)
		if binDir != cwd && !strings.HasPrefix(binDir, cwd+"/") {
			bwrapArgs = append(bwrapArgs, "--ro-bind-try", binDir, binDir)
		}
	}

	// Caller-supplied extra mounts.
	for _, p := range cfg.ExtraROBinds {
		bwrapArgs = append(bwrapArgs, "--ro-bind", p, p)
	}
	for _, p := range cfg.ExtraRWBinds {
		bwrapArgs = append(bwrapArgs, "--bind", p, p)
	}

	// Set working directory inside the sandbox.
	bwrapArgs = append(bwrapArgs, "--chdir", cwd)

	// Orphan prevention.
	bwrapArgs = append(bwrapArgs, "--die-with-parent")

	// Separator + original command.
	bwrapArgs = append(bwrapArgs, "--")
	bwrapArgs = append(bwrapArgs, originalCmd)
	bwrapArgs = append(bwrapArgs, originalArgs...)

	return "bwrap", bwrapArgs
}

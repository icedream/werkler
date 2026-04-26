// Package memorystore persists per-project notes across sessions.
//
// Notes are stored as a markdown file at:
//
//	~/.config/werkler/memory/<sha256-of-abs-cwd>.md
//
// The file is private to the user (dir 0700, file 0600) and is never
// written inside the project repository.
package memorystore

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// MaxBytes is the maximum number of bytes allowed in a memory file.
// Writes that would exceed this limit are rejected.
const MaxBytes = 8 * 1024 // 8 KB

// MemoryStore manages the per-CWD markdown memory file.
type MemoryStore struct {
	path string

	mu     sync.RWMutex
	cached string // in-memory mirror, updated on every successful Write
	loaded bool   // true once Read has been called (or Write has succeeded)
}

// New returns a MemoryStore for the given working directory.
// The storage directory is created if it does not exist.
func New(cwd string) (*MemoryStore, error) {
	abs, err := filepath.Abs(cwd)
	if err != nil {
		return nil, fmt.Errorf("memorystore: resolving cwd: %w", err)
	}

	dir, err := defaultDir()
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, fmt.Errorf("memorystore: creating storage dir: %w", err)
	}

	hash := sha256.Sum256([]byte(abs))
	filename := fmt.Sprintf("%x.md", hash)

	s := &MemoryStore{path: filepath.Join(dir, filename)}
	// Pre-load so Cached() is correct from the start.
	_, _ = s.Read()
	return s, nil
}

// Path returns the full filesystem path of the memory file.
func (s *MemoryStore) Path() string { return s.path }

// Exists reports whether the memory file exists and has non-empty content.
func (s *MemoryStore) Exists() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.loaded && s.cached != ""
}

// Cached returns the current in-memory content without reading the file again.
// Returns an empty string when no memory has been loaded or written yet.
func (s *MemoryStore) Cached() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cached
}

// Read returns the current memory content, reading from disk if not already
// loaded. Returns "" (no error) when the file does not exist.
func (s *MemoryStore) Read() (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.loaded {
		return s.cached, nil
	}
	data, err := os.ReadFile(s.path)
	if os.IsNotExist(err) {
		s.loaded = true
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("memorystore: reading file: %w", err)
	}
	s.cached = string(data)
	s.loaded = true
	return s.cached, nil
}

// Write atomically replaces the memory file with content.
// Returns an error if content exceeds MaxBytes.
// Writes use file mode 0600; parent directory is 0700.
func (s *MemoryStore) Write(content string) error {
	content = strings.TrimSpace(content)
	if len(content) > MaxBytes {
		return fmt.Errorf("memorystore: content too large (%d bytes, max %d) — summarise before writing", len(content), MaxBytes)
	}

	dir := filepath.Dir(s.path)
	tmp, err := os.CreateTemp(dir, ".mem-*.tmp")
	if err != nil {
		return fmt.Errorf("memorystore: creating temp file: %w", err)
	}
	tmpPath := tmp.Name()
	defer func() {
		// Clean up temp file on error (ignored if already renamed).
		_ = os.Remove(tmpPath)
	}()

	if _, err := fmt.Fprint(tmp, content); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("memorystore: writing temp file: %w", err)
	}
	if err := tmp.Chmod(0600); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("memorystore: setting file permissions: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("memorystore: closing temp file: %w", err)
	}
	if err := os.Rename(tmpPath, s.path); err != nil {
		return fmt.Errorf("memorystore: atomic rename: %w", err)
	}

	s.mu.Lock()
	s.cached = content
	s.loaded = true
	s.mu.Unlock()
	return nil
}

// defaultDir returns the platform-specific storage directory.
func defaultDir() (string, error) {
	cfgDir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("memorystore: resolving config dir: %w", err)
	}
	return filepath.Join(cfgDir, "werkler", "memory"), nil
}

// Package sandbox provides utilities for sandboxed AI tool execution.
// It includes a staging store for deferred file writes and bwrap-based
// process isolation.
package sandbox

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// OpKind describes what kind of staged file operation this entry represents.
type OpKind int

const (
	// OpWrite stages a file creation or overwrite.
	OpWrite OpKind = iota
	// OpDelete stages a file removal.
	OpDelete
)

// StagedOp is a single pending file operation held by a Store.
type StagedOp struct {
	Kind     OpKind
	Path     string    // absolute, cleaned path
	Content  []byte    // valid only for OpWrite
	StagedAt time.Time // when this op was recorded
}

// Store holds pending file operations that have not yet been written to disk.
// file_read calls go through the store so that subsequent tool calls see the
// staged (not yet committed) state rather than the on-disk state.
//
// All methods are safe for concurrent use.
type Store struct {
	mu  sync.RWMutex
	ops map[string]*StagedOp // keyed by absolute path
}

// NewStore allocates an empty staging Store.
func NewStore() *Store {
	return &Store{ops: make(map[string]*StagedOp)}
}

// StageWrite records a write (create or overwrite) for absPath.
// Calling StageWrite again for the same path replaces the earlier entry.
func (s *Store) StageWrite(absPath string, content []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ops[absPath] = &StagedOp{
		Kind:     OpWrite,
		Path:     absPath,
		Content:  content,
		StagedAt: time.Now(),
	}
}

// StageDelete records a deletion for absPath.
func (s *Store) StageDelete(absPath string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ops[absPath] = &StagedOp{
		Kind:     OpDelete,
		Path:     absPath,
		StagedAt: time.Now(),
	}
}

// Read returns the staged content for absPath if a staged entry exists.
//
//   - (content, true)  — path has a staged write; content is the new bytes.
//   - (nil, true)      — path is staged for deletion.
//   - (nil, false)     — no staged entry; caller should fall through to disk.
func (s *Store) Read(absPath string) (content []byte, found bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	op, ok := s.ops[absPath]
	if !ok {
		return nil, false
	}
	if op.Kind == OpDelete {
		return nil, true
	}
	return op.Content, true
}

// Len returns the number of pending operations.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.ops)
}

// Pending returns a stable (path-sorted) snapshot of all pending operations.
func (s *Store) Pending() []*StagedOp {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*StagedOp, 0, len(s.ops))
	for _, op := range s.ops {
		out = append(out, op)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

// Summary returns a human-readable list of pending operations for display.
func (s *Store) Summary() string {
	ops := s.Pending()
	if len(ops) == 0 {
		return "No staged changes."
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "%d staged change(s):\n", len(ops))
	for _, op := range ops {
		switch op.Kind {
		case OpWrite:
			fmt.Fprintf(&sb, "  WRITE   %s  (%d bytes)\n", op.Path, len(op.Content))
		case OpDelete:
			fmt.Fprintf(&sb, "  DELETE  %s\n", op.Path)
		}
	}
	return sb.String()
}

// CommitResult is returned by Commit.
type CommitResult struct {
	Written int
	Deleted int
	Failed  int
	Errors  []error
}

func (r CommitResult) String() string {
	s := fmt.Sprintf("Committed: %d written, %d deleted", r.Written, r.Deleted)
	if r.Failed > 0 {
		s += fmt.Sprintf(", %d failed", r.Failed)
	}
	return s
}

// Commit writes all staged operations to disk and clears the store.
// It is best-effort per file: if one write fails the others still proceed.
// The store is always cleared after Commit so callers can safely call it
// even when some operations fail.
func (s *Store) Commit() CommitResult {
	s.mu.Lock()
	defer s.mu.Unlock()

	var res CommitResult
	for path, op := range s.ops {
		switch op.Kind {
		case OpWrite:
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				res.Errors = append(res.Errors, fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err))
				res.Failed++
				continue
			}
			if err := os.WriteFile(path, op.Content, 0o644); err != nil {
				res.Errors = append(res.Errors, fmt.Errorf("write %s: %w", path, err))
				res.Failed++
				continue
			}
			res.Written++
		case OpDelete:
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				res.Errors = append(res.Errors, fmt.Errorf("delete %s: %w", path, err))
				res.Failed++
				continue
			}
			res.Deleted++
		}
	}
	s.ops = make(map[string]*StagedOp)
	return res
}

// Discard clears all staged operations without touching disk.
// Returns the number of operations discarded.
func (s *Store) Discard() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := len(s.ops)
	s.ops = make(map[string]*StagedOp)
	return n
}

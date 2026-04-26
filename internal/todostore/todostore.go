// Package todostore provides an in-memory, thread-safe todo list shared between
// the AI tool layer and the TUI.
package todostore

import (
	"fmt"
	"sync"
	"time"
)

// Status values for a Todo.
const (
	StatusPending    = "pending"
	StatusInProgress = "in_progress"
	StatusDone       = "done"
	StatusBlocked    = "blocked"
)

var validStatuses = map[string]bool{
	StatusPending:    true,
	StatusInProgress: true,
	StatusDone:       true,
	StatusBlocked:    true,
}

// Todo is a single task item managed by the AI.
type Todo struct {
	ID          string
	Title       string
	Description string
	Status      string
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// Store holds an ordered list of todos and notifies a listener on every mutation.
type Store struct {
	mu     sync.Mutex
	todos  []Todo
	nextID int
	notify func() // called after unlock; may be nil
}

// New returns an empty Store.
func New() *Store { return &Store{} }

// SetNotify registers fn to be called (outside the lock) after any mutation.
// Replaces any previously registered callback. Safe to call before use.
func (s *Store) SetNotify(fn func()) {
	s.mu.Lock()
	s.notify = fn
	s.mu.Unlock()
}

// Add creates a new todo with StatusPending and returns its ID.
// If id is non-empty it is used as-is (caller must ensure uniqueness);
// otherwise a numeric fallback id ("t1", "t2", …) is generated.
func (s *Store) Add(id, title, description string) string {
	s.mu.Lock()
	if id == "" {
		s.nextID++
		id = fmt.Sprintf("t%d", s.nextID)
	}
	now := time.Now()
	s.todos = append(s.todos, Todo{
		ID:          id,
		Title:       title,
		Description: description,
		Status:      StatusPending,
		CreatedAt:   now,
		UpdatedAt:   now,
	})
	notify := s.notify
	s.mu.Unlock()
	if notify != nil {
		notify()
	}
	return id
}

// UpdateFields carries optional fields for Update. Nil pointer = no change.
type UpdateFields struct {
	Title       *string
	Description *string
	Status      *string
}

// Update modifies an existing todo. Returns an error if the ID is not found or
// the requested status is invalid.
func (s *Store) Update(id string, f UpdateFields) error {
	if f.Status != nil && !validStatuses[*f.Status] {
		return fmt.Errorf("invalid status %q; must be one of: pending, in_progress, done, blocked", *f.Status)
	}

	s.mu.Lock()
	idx := -1
	for i := range s.todos {
		if s.todos[i].ID == id {
			idx = i
			break
		}
	}
	if idx < 0 {
		s.mu.Unlock()
		return fmt.Errorf("todo %q not found", id)
	}

	t := &s.todos[idx]
	if f.Title != nil {
		t.Title = *f.Title
	}
	if f.Description != nil {
		t.Description = *f.Description
	}
	if f.Status != nil {
		t.Status = *f.Status
	}
	t.UpdatedAt = time.Now()
	notify := s.notify
	s.mu.Unlock()

	if notify != nil {
		notify()
	}
	return nil
}

// List returns a snapshot copy of all todos.
func (s *Store) List() []Todo {
	s.mu.Lock()
	out := make([]Todo, len(s.todos))
	copy(out, s.todos)
	s.mu.Unlock()
	return out
}

// Restore replaces the store contents with the given todos (e.g. on session resume).
// The ID counter is set to the highest numeric ID found so new todos don't collide.
// Fires the notify callback if one is registered.
func (s *Store) Restore(todos []Todo) {
	s.mu.Lock()
	s.todos = make([]Todo, len(todos))
	copy(s.todos, todos)
	s.nextID = 0
	for _, t := range s.todos {
		// IDs are "t<n>"; parse the number to find the high-water mark.
		var n int
		if _, err := fmt.Sscanf(t.ID, "t%d", &n); err == nil && n > s.nextID {
			s.nextID = n
		}
	}
	notify := s.notify
	s.mu.Unlock()
	if notify != nil {
		notify()
	}
}

// Clear removes all todos and resets the ID counter. Used when starting a new session.
func (s *Store) Clear() {
	s.mu.Lock()
	s.todos = s.todos[:0]
	s.nextID = 0
	notify := s.notify
	s.mu.Unlock()
	if notify != nil {
		notify()
	}
}

// Counts returns the number of todos in each status bucket.
func (s *Store) Counts() (pending, inProgress, done, blocked int) {
	s.mu.Lock()
	for _, t := range s.todos {
		switch t.Status {
		case StatusPending:
			pending++
		case StatusInProgress:
			inProgress++
		case StatusDone:
			done++
		case StatusBlocked:
			blocked++
		}
	}
	s.mu.Unlock()
	return
}

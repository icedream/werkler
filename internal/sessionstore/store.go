// Package sessionstore persists and retrieves werkler chat sessions.
// Sessions are stored as JSON files named {id}.json in a per-user directory.
package sessionstore

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/icedream/werkler/internal/ai"
)

// Session is the full persisted state of a chat conversation.
type Session struct {
	ID        string       `json:"id"`
	Title     string       `json:"title"`
	CreatedAt time.Time    `json:"created_at"`
	UpdatedAt time.Time    `json:"updated_at"`
	CWD       string       `json:"cwd"`
	Model     string       `json:"model"`
	Messages  []ai.Message `json:"messages"`
	// ApprovedTools persists session-level tool approvals so they carry over on resume.
	ApprovedTools []string `json:"approved_tools,omitempty"`
}

// Store reads and writes sessions to a directory on disk.
type Store struct {
	dir string
}

// New creates a Store backed by dir.
func New(dir string) *Store {
	return &Store{dir: dir}
}

// DefaultDir returns the platform-appropriate directory for session files.
// On Linux/BSD: $XDG_CONFIG_HOME/werkler/sessions (usually ~/.config/werkler/sessions).
// On macOS/Windows: follows os.UserConfigDir conventions.
func DefaultDir() string {
	cfgDir, err := os.UserConfigDir()
	if err != nil {
		home, _ := os.UserHomeDir()
		cfgDir = filepath.Join(home, ".config")
	}
	return filepath.Join(cfgDir, "werkler", "sessions")
}

// Save writes sess to disk, updating UpdatedAt. If the file already exists it
// is overwritten atomically via a temp file + rename.
func (s *Store) Save(sess *Session) error {
	if err := os.MkdirAll(s.dir, 0o700); err != nil {
		return fmt.Errorf("sessionstore: creating directory: %w", err)
	}
	sess.UpdatedAt = time.Now()
	data, err := json.MarshalIndent(sess, "", "  ")
	if err != nil {
		return fmt.Errorf("sessionstore: marshalling %s: %w", sess.ID, err)
	}
	// Write to a temp file then rename for atomicity.
	tmp, err := os.CreateTemp(s.dir, ".tmp-")
	if err != nil {
		return fmt.Errorf("sessionstore: creating temp file: %w", err)
	}
	if _, err = tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("sessionstore: writing %s: %w", sess.ID, err)
	}
	if err = tmp.Close(); err != nil {
		_ = os.Remove(tmp.Name())
		return fmt.Errorf("sessionstore: closing temp file: %w", err)
	}
	dest := filepath.Join(s.dir, sess.ID+".json")
	return os.Rename(tmp.Name(), dest)
}

// Load reads the session with the given ID.
func (s *Store) Load(id string) (*Session, error) {
	path := filepath.Join(s.dir, id+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("sessionstore: reading %s: %w", id, err)
	}
	var sess Session
	if err := json.Unmarshal(data, &sess); err != nil {
		return nil, fmt.Errorf("sessionstore: parsing %s: %w", id, err)
	}
	return &sess, nil
}

// LoadByPrefix loads a session whose ID starts with prefix.
// Returns an error if zero or more than one session matches.
func (s *Store) LoadByPrefix(prefix string) (*Session, error) {
	sessions, err := s.List()
	if err != nil {
		return nil, err
	}
	var matches []Session
	for _, sess := range sessions {
		if strings.HasPrefix(sess.ID, prefix) {
			matches = append(matches, sess)
		}
	}
	switch len(matches) {
	case 0:
		return nil, fmt.Errorf("no session found with prefix %q", prefix)
	case 1:
		return &matches[0], nil
	default:
		return nil, fmt.Errorf("prefix %q is ambiguous (%d sessions match)", prefix, len(matches))
	}
}

// List returns all sessions sorted by UpdatedAt descending (most recent first).
// Corrupt or unreadable files are silently skipped.
func (s *Store) List() ([]Session, error) {
	entries, err := os.ReadDir(s.dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("sessionstore: reading directory: %w", err)
	}
	var sessions []Session
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		id := strings.TrimSuffix(e.Name(), ".json")
		sess, err := s.Load(id)
		if err != nil {
			continue // skip corrupt files
		}
		sessions = append(sessions, *sess)
	}
	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].UpdatedAt.After(sessions[j].UpdatedAt)
	})
	return sessions, nil
}

// Delete removes the session with the given ID from disk.
func (s *Store) Delete(id string) error {
	path := filepath.Join(s.dir, id+".json")
	if err := os.Remove(path); err != nil {
		return fmt.Errorf("sessionstore: deleting %s: %w", id, err)
	}
	return nil
}

// LoadLatestForCWD returns the most recently updated session whose CWD
// matches cwd, or nil if none exists.
func (s *Store) LoadLatestForCWD(cwd string) (*Session, error) {
	sessions, err := s.List()
	if err != nil {
		return nil, err
	}
	for i := range sessions { // already sorted by UpdatedAt desc
		if sessions[i].CWD == cwd {
			return &sessions[i], nil
		}
	}
	return nil, nil
}

// NewID generates a new unique session ID based on the current time plus random bytes.
// IDs are lexicographically sortable by creation time.
func NewID() string {
	var buf [4]byte
	_, _ = rand.Read(buf[:])
	return time.Now().Format("20060102-150405") + "-" + hex.EncodeToString(buf[:])
}

// GenerateTitle creates a short human-readable title from the conversation's
// first user message. Falls back to "New session" if no user message exists.
func GenerateTitle(messages []ai.Message) string {
	for _, msg := range messages {
		if msg.Role == "user" && msg.Content != "" {
			title := strings.ReplaceAll(msg.Content, "\n", " ")
			title = strings.Join(strings.Fields(title), " ") // collapse whitespace
			if len([]rune(title)) > 60 {
				title = string([]rune(title)[:57]) + "…"
			}
			return title
		}
	}
	return "New session"
}

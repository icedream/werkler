package process

import (
	"bytes"
	"fmt"
	"sync"
	"unicode/utf8"
)

// capBuffer is a size-limited write buffer.  Writes beyond the cap are
// silently discarded; Truncated records the total bytes dropped.
type capBuffer struct {
	mu        sync.Mutex
	buf       bytes.Buffer
	cap       int
	written   int
	Truncated int
}

func newCapBuffer(cap int) *capBuffer { return &capBuffer{cap: cap} }

func (b *capBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	remaining := b.cap - b.written
	if remaining <= 0 {
		b.Truncated += len(p)
		return len(p), nil
	}
	if len(p) > remaining {
		b.buf.Write(p[:remaining])
		b.written += remaining
		b.Truncated += len(p) - remaining
		return len(p), nil
	}
	b.buf.Write(p)
	b.written += len(p)
	return len(p), nil
}

func (b *capBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	raw := b.buf.Bytes()
	for len(raw) > 0 && !utf8.Valid(raw) {
		raw = raw[:len(raw)-1]
	}
	s := string(raw)
	if b.Truncated > 0 {
		s += fmt.Sprintf("\n[truncated: %d bytes omitted]", b.Truncated)
	}
	return s
}

// combinedWriter multiplexes writes to a stream-specific buffer and a shared
// combined buffer.  The combined buffer is serialised by a shared mutex so
// concurrent stdout+stderr writes appear in order.
type combinedWriter struct {
	stream   *capBuffer
	combined *capBuffer
	mu       *sync.Mutex
}

func (w *combinedWriter) Write(p []byte) (int, error) {
	_, _ = w.stream.Write(p)
	w.mu.Lock()
	_, _ = w.combined.Write(p)
	w.mu.Unlock()
	return len(p), nil
}

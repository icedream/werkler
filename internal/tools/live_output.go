package tools

import (
	"bytes"
	"context"
	"io"
)

type liveOutputKeyType struct{}

var liveOutputKey = liveOutputKeyType{}

// WithLiveOutput returns a context that carries ch as the destination for
// live output lines from tools that support streaming (e.g. run_command).
// Each complete stdout/stderr line is sent to ch as it arrives.
func WithLiveOutput(ctx context.Context, ch chan<- string) context.Context {
	return context.WithValue(ctx, liveOutputKey, ch)
}

func liveOutputFromCtx(ctx context.Context) chan<- string {
	ch, _ := ctx.Value(liveOutputKey).(chan<- string)
	return ch
}

// liveLineWriter wraps an existing io.Writer and forwards each complete line
// to liveCh as it arrives.  Incomplete final lines (no trailing newline) are
// flushed by calling flush() after the process exits.
//
// NOT goroutine-safe on its own — callers must ensure serialised calls to
// Write/flush (the existing combinedWriter mutex satisfies this).
type liveLineWriter struct {
	inner  io.Writer     // existing combinedWriter or capBuffer
	liveCh chan<- string // receives one string per complete line
	buf    bytes.Buffer  // incomplete-line accumulator
}

func (w *liveLineWriter) Write(p []byte) (int, error) {
	n, err := w.inner.Write(p)
	w.buf.Write(p[:n])
	for {
		b := w.buf.Bytes()
		idx := bytes.IndexByte(b, '\n')
		if idx < 0 {
			break
		}
		line := b[:idx]
		// Strip trailing CR so Windows-style CRLF renders cleanly.
		if len(line) > 0 && line[len(line)-1] == '\r' {
			line = line[:len(line)-1]
		}
		select {
		case w.liveCh <- string(line):
		default: // non-blocking: drop if consumer is too slow
		}
		w.buf.Next(idx + 1)
	}
	return n, err
}

// flush sends any remaining buffered bytes as a final (unterminated) line.
func (w *liveLineWriter) flush() {
	if w.buf.Len() > 0 {
		select {
		case w.liveCh <- w.buf.String():
		default:
		}
		w.buf.Reset()
	}
}

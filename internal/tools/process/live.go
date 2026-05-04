package process

import (
	"bytes"
	"io"
)

// liveLineWriter wraps an existing io.Writer and forwards each complete line
// to liveCh as it arrives.  Incomplete final lines are flushed by calling
// flush() after the process exits.
type liveLineWriter struct {
	inner  io.Writer
	liveCh chan<- string
	buf    bytes.Buffer
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
		if len(line) > 0 && line[len(line)-1] == '\r' {
			line = line[:len(line)-1]
		}
		select {
		case w.liveCh <- string(line):
		default:
		}
		w.buf.Next(idx + 1)
	}
	return n, err
}

func (w *liveLineWriter) flush() {
	if w.buf.Len() > 0 {
		select {
		case w.liveCh <- w.buf.String():
		default:
		}
		w.buf.Reset()
	}
}

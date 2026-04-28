package ai

import (
	"bytes"
	"io"
	"net/http"
	"strings"
)

// NewReasoningAliasTransport returns an [http.RoundTripper] that renames
// provider-specific thinking-token fields to "reasoning_content" in SSE delta
// JSON objects so that go-openai's delta.ReasoningContent field is populated.
//
// Two aliases are applied:
//   - "reasoning":  → "reasoning_content":  (Ollama / DeepSeek thinking models)
//   - "thinking":   → "reasoning_content":  (GitHub Copilot / Claude models)
//
// The transform is a no-op for providers that already use "reasoning_content"
// (OpenAI o-series) or that send no thinking tokens at all (most models).
// If inner is nil, http.DefaultTransport is used.
func NewReasoningAliasTransport(inner http.RoundTripper) http.RoundTripper {
	if inner == nil {
		inner = http.DefaultTransport
	}
	return &reasoningAliasTransport{inner: inner}
}

type reasoningAliasTransport struct{ inner http.RoundTripper }

func (t *reasoningAliasTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.inner.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	ct := resp.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "text/event-stream") {
		resp.Body = &reasoningAliasBody{inner: resp.Body}
	}
	return resp, nil
}

// reasoningAliasBody wraps an SSE response body and renames provider-specific
// thinking-token fields ("reasoning": or "thinking":) to "reasoning_content":
// in data: lines so go-openai parses thinking tokens correctly.
type reasoningAliasBody struct {
	inner   io.ReadCloser
	outBuf  []byte
	inBuf   []byte
	done    bool
	pendErr error // non-EOF error deferred until outBuf is drained
}

var (
	sseDataPrefix   = []byte("data:")
	reasoningKeyOld = []byte(`"reasoning":`)
	thinkingKeyOld  = []byte(`"thinking":`)
	reasoningKeyNew = []byte(`"reasoning_content":`)
)

func (b *reasoningAliasBody) Read(p []byte) (int, error) {
	for len(b.outBuf) == 0 {
		if b.done {
			if b.pendErr != nil {
				err := b.pendErr
				b.pendErr = nil
				return 0, err
			}
			return 0, io.EOF
		}

		// Read a fresh chunk from the upstream body.
		tmp := make([]byte, 4096)
		n, readErr := b.inner.Read(tmp)
		if n > 0 {
			b.inBuf = append(b.inBuf, tmp[:n]...)
		}
		if readErr != nil {
			if readErr == io.EOF {
				b.done = true
			} else {
				b.pendErr = readErr
				b.done = true
			}
		}

		// Process all complete newline-terminated lines available.
		for {
			idx := bytes.IndexByte(b.inBuf, '\n')
			if idx < 0 {
				if b.done && len(b.inBuf) > 0 {
					// Unterminated trailing line — flush as-is with transform.
					b.outBuf = append(b.outBuf, transformSSELine(b.inBuf)...)
					b.inBuf = nil
				}
				break
			}
			b.outBuf = append(b.outBuf, transformSSELine(b.inBuf[:idx+1])...)
			b.inBuf = b.inBuf[idx+1:]
		}
	}

	n := copy(p, b.outBuf)
	b.outBuf = b.outBuf[n:]
	return n, nil
}

func (b *reasoningAliasBody) Close() error { return b.inner.Close() }

// transformSSELine renames provider-specific thinking-token keys to
// "reasoning_content": in SSE data lines so go-openai can parse them.
// It handles two aliases:
//   - "reasoning":  (Ollama / DeepSeek)
//   - "thinking":   (GitHub Copilot / Claude)
//
// Non-data lines (comments, event:, blank) are returned unchanged.
func transformSSELine(line []byte) []byte {
	// Find the "data:" prefix, allowing optional whitespace after it.
	if !bytes.HasPrefix(line, sseDataPrefix) {
		return line
	}
	payload := bytes.TrimLeft(line[len(sseDataPrefix):], " \t")
	if len(payload) == 0 || payload[0] != '{' {
		return line
	}
	// Skip if already uses reasoning_content (avoid double-replace).
	if bytes.Contains(payload, reasoningKeyNew) {
		return line
	}

	var oldKey []byte
	switch {
	case bytes.Contains(payload, reasoningKeyOld):
		oldKey = reasoningKeyOld
	case bytes.Contains(payload, thinkingKeyOld):
		oldKey = thinkingKeyOld
	default:
		return line
	}

	// Rebuild: keep the original "data:" prefix bytes (preserving original
	// spacing), then replace the matched key in the JSON payload.
	prefixLen := len(line) - len(payload)
	transformed := bytes.Replace(payload, oldKey, reasoningKeyNew, 1)
	out := make([]byte, prefixLen+len(transformed))
	copy(out, line[:prefixLen])
	copy(out[prefixLen:], transformed)
	return out
}

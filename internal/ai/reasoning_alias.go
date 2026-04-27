package ai

import (
	"bytes"
	"io"
	"net/http"
	"strings"
)

// NewReasoningAliasTransport returns an [http.RoundTripper] that renames the
// "reasoning" field to "reasoning_content" in SSE delta JSON objects.
//
// Ollama's OpenAI-compatible endpoint emits thinking tokens as
// delta.reasoning (json:"reasoning"), but go-openai only maps
// delta.reasoning_content. Applying this transport makes thinking tokens from
// Ollama-hosted models visible to the rest of the application.
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

// reasoningAliasBody wraps an SSE response body and renames "reasoning": to
// "reasoning_content": in data: lines so go-openai parses Ollama's thinking
// tokens correctly.
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

// transformSSELine renames "reasoning": to "reasoning_content": in SSE data
// lines. Non-data lines (comments, event:, blank) are returned unchanged.
func transformSSELine(line []byte) []byte {
	// Find the "data:" prefix, allowing optional whitespace after it.
	rest := line
	if !bytes.HasPrefix(rest, sseDataPrefix) {
		return line
	}
	payload := bytes.TrimLeft(rest[len(sseDataPrefix):], " \t")
	if len(payload) == 0 || payload[0] != '{' {
		return line
	}
	if !bytes.Contains(payload, reasoningKeyOld) {
		return line
	}
	// Rebuild: keep the original "data:" prefix bytes (preserving original
	// spacing), then replace reasoning key in the JSON payload.
	prefixLen := len(line) - len(payload)
	transformed := bytes.Replace(payload, reasoningKeyOld, reasoningKeyNew, 1)
	out := make([]byte, prefixLen+len(transformed))
	copy(out, line[:prefixLen])
	copy(out[prefixLen:], transformed)
	return out
}

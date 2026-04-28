package ai

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"testing"
)

// roundTripperFunc is a convenience adapter for http.RoundTripper.
type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func TestTransformSSELine(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		// Standard spacing
		{`data: {"choices":[{"delta":{"reasoning":"think"}}]}`, `data: {"choices":[{"delta":{"reasoning_content":"think"}}]}`},
		// No space after colon
		{`data:{"choices":[{"delta":{"reasoning":"x"}}]}`, `data:{"choices":[{"delta":{"reasoning_content":"x"}}]}`},
		// Tab after colon
		{"data:\t{\"choices\":[{\"delta\":{\"reasoning\":\"x\"}}]}", "data:\t{\"choices\":[{\"delta\":{\"reasoning_content\":\"x\"}}]}"},
		// Already uses reasoning_content — must NOT double-replace
		{`data: {"choices":[{"delta":{"reasoning_content":"already"}}]}`, `data: {"choices":[{"delta":{"reasoning_content":"already"}}]}`},
		// thinking: alias (GitHub Copilot / Claude)
		{`data: {"choices":[{"delta":{"thinking":"think"}}]}`, `data: {"choices":[{"delta":{"reasoning_content":"think"}}]}`},
		{`data:{"choices":[{"delta":{"thinking":"x"}}]}`, `data:{"choices":[{"delta":{"reasoning_content":"x"}}]}`},
		// thinking: must NOT fire when reasoning_content already present
		{`data: {"choices":[{"delta":{"reasoning_content":"already","thinking":"x"}}]}`, `data: {"choices":[{"delta":{"reasoning_content":"already","thinking":"x"}}]}`},
		// Non-data lines untouched
		{`event: ping`, `event: ping`},
		{``, ``},
		{`: comment`, `: comment`},
		{`data: [DONE]`, `data: [DONE]`},
		// data: with non-JSON payload
		{`data: hello`, `data: hello`},
		// reasoning_content should NOT be double-replaced
		{`data: {"reasoning_content":"x"}`, `data: {"reasoning_content":"x"}`},
	}
	for _, tt := range tests {
		got := string(transformSSELine([]byte(tt.in)))
		if got != tt.want {
			t.Errorf("transformSSELine(%q)\n  got  %q\n  want %q", tt.in, got, tt.want)
		}
	}
}

func TestReasoningAliasBody_Basic(t *testing.T) {
	input := "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think\"}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n" +
		"data: [DONE]\n"

	want := "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n" +
		"data: [DONE]\n"

	body := &reasoningAliasBody{inner: io.NopCloser(bytes.NewBufferString(input))}
	got, err := io.ReadAll(body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if string(got) != want {
		t.Errorf("got:\n%s\nwant:\n%s", got, want)
	}
}

func TestReasoningAliasBody_SmallReads(t *testing.T) {
	// Simulate a slow upstream that returns 1 byte at a time.
	input := "data: {\"reasoning\":\"x\"}\ndata: [DONE]\n"
	want := "data: {\"reasoning_content\":\"x\"}\ndata: [DONE]\n"

	body := &reasoningAliasBody{inner: io.NopCloser(&slowReader{data: []byte(input)})}
	got, err := io.ReadAll(body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if string(got) != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// slowReader returns one byte per Read call.
type slowReader struct {
	data []byte
	pos  int
}

func (r *slowReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	p[0] = r.data[r.pos]
	r.pos++
	return 1, nil
}

func TestReasoningAliasBody_NonEOFError(t *testing.T) {
	sentinel := errors.New("network error")
	inner := &errorAfterReader{
		data: []byte("data: {\"reasoning\":\"x\"}\n"),
		err:  sentinel,
	}
	body := &reasoningAliasBody{inner: io.NopCloser(inner)}
	got, err := io.ReadAll(body)
	// Should get the transformed data before the error.
	wantData := "data: {\"reasoning_content\":\"x\"}\n"
	if string(got) != wantData {
		t.Errorf("got data %q, want %q", got, wantData)
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("got error %v, want %v", err, sentinel)
	}
}

// errorAfterReader returns all data then a non-EOF error.
type errorAfterReader struct {
	data []byte
	pos  int
	err  error
}

func (r *errorAfterReader) Read(p []byte) (int, error) {
	if r.pos < len(r.data) {
		n := copy(p, r.data[r.pos:])
		r.pos += n
		return n, nil
	}
	return 0, r.err
}

func TestReasoningAliasTransport_NonSSE(t *testing.T) {
	// Non-SSE responses must be passed through unchanged.
	body := `{"choices":[{"message":{"reasoning":"raw"}}]}`
	inner := roundTripperFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(bytes.NewBufferString(body)),
		}, nil
	})
	rt := NewReasoningAliasTransport(inner)
	resp, err := rt.RoundTrip(&http.Request{})
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	got, _ := io.ReadAll(resp.Body)
	if string(got) != body {
		t.Errorf("non-SSE body should be unchanged, got %q", got)
	}
}

func TestReasoningAliasTransport_SSEWithCharset(t *testing.T) {
	// Content-Type with charset should still be transformed.
	input := "data: {\"reasoning\":\"x\"}\n"
	inner := roundTripperFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"text/event-stream; charset=utf-8"}},
			Body:       io.NopCloser(bytes.NewBufferString(input)),
		}, nil
	})
	rt := NewReasoningAliasTransport(inner)
	resp, err := rt.RoundTrip(&http.Request{})
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	got, _ := io.ReadAll(resp.Body)
	want := "data: {\"reasoning_content\":\"x\"}\n"
	if string(got) != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

package tools

import (
	"context"
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

package file

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/icedream/werkler/internal/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// mockContext satisfies the file.Context interface.
type mockContext struct {
	mock.Mock
}

func (m *mockContext) ActiveApprover(ctx context.Context) PathApprover {
	args := m.Called(ctx)
	if args.Get(0) == nil {
		return nil
	}
	return args.Get(0).(PathApprover)
}

func (m *mockContext) CheckRedundantRead(path string, ranges []chat.Range) (bool, string) {
	args := m.Called(path, ranges)
	return args.Bool(0), args.String(1)
}

func (m *mockContext) RecordRecentRead(path, rawPath string, ranges []chat.Range) {
	m.Called(path, rawPath, ranges)
}

// dummyApprover is a simple PathApprover that approves everything.
type dummyApprover struct{}

func (dummyApprover) IsPathReadApproved(_ string) bool  { return true }
func (dummyApprover) IsPathWriteApproved(_ string) bool { return true }

func TestHandleFileReadMulti_RedundancyDetection(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "test.txt")
	content := "line 1\nline 2\nline 3\nline 4\nline 5\n"
	require.NoError(t, os.WriteFile(f, []byte(content), 0o644))

	ctx := &mockContext{}
	h := NewHandler(ctx)

	// 1. Test Redundant Read Detection
	args := map[string]any{
		"regions": []any{
			map[string]any{
				"path":       f,
				"start_line": 1.0,
				"end_line":   2.0,
			},
		},
	}

	// First call: Not redundant
	ctx.On("ActiveApprover", mock.Anything).Return(dummyApprover{}).Once()
	ctx.On("CheckRedundantRead", f, []chat.Range{{StartLine: 1, EndLine: 2}}).Return(false, "").Once()
	ctx.On("RecordRecentRead", f, f, []chat.Range{{StartLine: 1, EndLine: 2}}).Return().Once()

	result, err := h.handleFileReadMulti(t.Context(), args)
	require.NoError(t, err)
	assert.Contains(t, result, "line 1")
	assert.Contains(t, result, "line 2")

	// Second call: Redundant
	ctx.On("ActiveApprover", mock.Anything).Return(dummyApprover{}).Once()
	ctx.On("CheckRedundantRead", f, []chat.Range{{StartLine: 1, EndLine: 2}}).Return(true, f).Once()

	result, err = h.handleFileReadMulti(t.Context(), args)
	require.NoError(t, err)
	assert.Contains(t, result, "Warning: The requested range [1-2] for "+f+" has already been read in this session and is currently in your context.")
	ctx.AssertExpectations(t)
}

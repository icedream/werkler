package chat_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
)

// --- Mock ToolManager ---

type mockToolManagerRedundancy struct {
	mock.Mock
}

func (m *mockToolManagerRedundancy) Tools(ctx context.Context) ([]ai.ToolDefinition, error) {
	args := m.Called(ctx)
	return args.Get(0).([]ai.ToolDefinition), args.Error(1)
}

func (m *mockToolManagerRedundancy) CallTool(ctx context.Context, name string, toolArgs map[string]any) (string, error) {
	args := m.Called(ctx, name, toolArgs)
	return args.String(0), args.Error(1)
}

func TestSession_RedundancyDetection(t *testing.T) {
	tm := &mockToolManagerRedundancy{}
	s := chat.NewSession(tm, nil, nil)

	path := "/project/src/main.go"
	rawPath := "./src/main.go"
	ranges := []chat.Range{{StartLine: 1, EndLine: 10}}

	// 1. Test RecordRecentRead and CheckRedundantRead (Exact Match)
	s.RecordRecentRead(path, rawPath, ranges)
	isRedundant, returnedRawPath := s.CheckRedundantRead(path, ranges)
	assert.True(t, isRedundant)
	assert.Equal(t, rawPath, returnedRawPath)

	// 2. Test Sub-range Match
	subRange := []chat.Range{{StartLine: 3, EndLine: 5}}
	isRedundant, returnedRawPath = s.CheckRedundantRead(path, subRange)
	assert.True(t, isRedundant)
	assert.Equal(t, rawPath, returnedRawPath)

	// 3. Test No Match (Different range)
	differentRange := []chat.Range{{StartLine: 11, EndLine: 20}}
	isRedundant, _ = s.CheckRedundantRead(path, differentRange)
	assert.False(t, isRedundant)

	// 4. Test No Match (Different file)
	otherPath := "/project/src/other.go"
	isRedundant, _ = s.CheckRedundantRead(otherPath, ranges)
	assert.False(t, isRedundant)

	// 5. Test Multiple ranges in one read
	multiRanges := []chat.Range{{StartLine: 1, EndLine: 5}, {StartLine: 10, EndLine: 15}}
	s.RecordRecentRead(path, rawPath, multiRanges)

	// Sub-range of one of the ranges
	isRedundant, _ = s.CheckRedundantRead(path, []chat.Range{{StartLine: 2, EndLine: 4}})
	assert.True(t, isRedundant)
	isRedundant, _ = s.CheckRedundantRead(path, []chat.Range{{StartLine: 11, EndLine: 14}})
	assert.True(t, isRedundant)

	// Range that spans across two ranges but doesn't cover them?
	// Wait, the implementation says:
	// "Check if all requested ranges are covered by previously read ranges for this file"
	// If I request [1, 15], it is NOT covered by {[1, 5], [10, 15]} because [6, 9] is missing.
	isRedundant, _ = s.CheckRedundantRead(path, []chat.Range{{StartLine: 1, EndLine: 15}})
	assert.False(t, isRedundant)
}

func TestSession_RedundancyPruning(t *testing.T) {
	tm := &mockToolManagerRedundancy{}
	s := chat.NewSession(tm, nil, nil)

	// Add 50 reads
	for i := 0; i < 50; i++ {
		s.RecordRecentRead(fmt.Sprintf("/path/%d", i), fmt.Sprintf("./path/%d", i), []chat.Range{{StartLine: 1, EndLine: 1}})
	}

	// Add the 51st read
	s.RecordRecentRead("/path/50", "./path/50", []chat.Range{{StartLine: 1, EndLine: 1}})

	// Check if /path/0 was pruned
	isRedundant, _ := s.CheckRedundantRead("/path/0", []chat.Range{{StartLine: 1, EndLine: 1}})
	assert.False(t, isRedundant, "The first read should have been pruned")

	// Check if /path/50 is present
	isRedundant, _ = s.CheckRedundantRead("/path/50", []chat.Range{{StartLine: 1, EndLine: 1}})
	assert.True(t, isRedundant, "The latest read should be present")
}

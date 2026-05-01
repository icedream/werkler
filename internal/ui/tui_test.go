package ui

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
)

// --- Mock StreamCompleter ---

type mockStreamCompleter struct {
	mock.Mock
}

func (m *mockStreamCompleter) CompleteStream(
	ctx context.Context,
	messages []ai.Message,
	tools []ai.ToolDefinition,
) <-chan ai.StreamChunk {
	args := m.Called(ctx, messages, tools)
	return args.Get(0).(<-chan ai.StreamChunk)
}

// chanOf returns a read-only channel pre-loaded with the given chunks.
func chanOf(chunks ...ai.StreamChunk) <-chan ai.StreamChunk {
	ch := make(chan ai.StreamChunk, len(chunks))
	for _, c := range chunks {
		ch <- c
	}
	close(ch)
	return ch
}

// --- Mock ToolManager ---

type mockToolManager struct {
	mock.Mock
}

func (m *mockToolManager) Tools(ctx context.Context) ([]ai.ToolDefinition, error) {
	args := m.Called(ctx)
	return args.Get(0).([]ai.ToolDefinition), args.Error(1)
}

func (m *mockToolManager) CallTool(ctx context.Context, name string, toolArgs map[string]any) (string, error) {
	args := m.Called(ctx, name, toolArgs)
	return args.String(0), args.Error(1)
}

// baseModel builds a minimal, ready-to-use Model for tests.
// It uses a nil client; tests that exercise streaming inject their own.
func baseModel() Model {
	tm := &mockToolManager{}
	session := chat.NewSession(tm, nil, nil)
	return initialModel(
		context.Background(),
		nil, // client not needed for most state-machine tests
		session,
		nil,
		"test-model",
		nil,
		"dark",
		nil, // send not needed for state-machine tests
	)
}

// update is a convenience wrapper that calls Update and asserts no panic.
func update(t *testing.T, m Model, msg tea.Msg) (Model, tea.Cmd) {
	t.Helper()
	next, cmd := (&m).Update(msg)
	switch v := next.(type) {
	case *Model:
		return *v, cmd
	default:
		return Model{}, cmd
	}
}

// --- formatArgsCompact ---

func TestFormatArgsCompact_Empty(t *testing.T) {
	assert.Equal(t, "", formatArgsCompact(nil))
	assert.Equal(t, "", formatArgsCompact(map[string]any{}))
}

func TestFormatArgsCompact_Simple(t *testing.T) {
	out := formatArgsCompact(map[string]any{"k": "v"})
	assert.Contains(t, out, `"k"`)
	assert.Contains(t, out, `"v"`)
}

func TestFormatArgsCompact_Truncation(t *testing.T) {
	// Build a value that produces JSON > 120 chars.
	long := map[string]any{"key": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}
	out := formatArgsCompact(long)
	// s[:120] + "…" where "…" is 3 bytes → max 123 bytes.
	assert.LessOrEqual(t, len(out), 123)
	assert.True(t, len(out) > 120, "should contain truncated content")
}

// --- streamChunkMsg: first delta ---

func TestUpdate_StreamDelta_FirstToken_CreatesAssistantItem(t *testing.T) {
	m := baseModel()
	m.state = stateThinking

	ch := chanOf() // not used; passed for completeness
	m, _ = update(t, m, streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Delta: "Hello"}})

	assert.Equal(t, stateStreaming, m.state)
	// First delta reserves two slots: reasoning (empty) at [0], assistant at [1].
	require.Len(t, m.items, 2)
	assert.Equal(t, itemReasoning, m.items[0].kind)
	assert.Equal(t, "", m.items[0].content) // no reasoning emitted
	assert.Equal(t, itemAssistant, m.items[1].kind)
	assert.Equal(t, "Hello", m.items[1].content)
	assert.Equal(t, 0, m.reasoningItemIdx)
	assert.Equal(t, 1, m.streamingItemIdx)
}

func TestUpdate_StreamDelta_SubsequentTokens_AppendContent(t *testing.T) {
	m := baseModel()
	m.state = stateStreaming
	m.items = append(m.items, displayItem{kind: itemAssistant, content: "Hel"})
	m.streamingItemIdx = 0

	ch := chanOf()
	m, _ = update(t, m, streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Delta: "lo!"}})

	assert.Equal(t, stateStreaming, m.state)
	assert.Equal(t, "Hello!", m.items[0].content)
}

// --- streamChunkMsg: Done (no tool calls) ---

func TestUpdate_StreamDone_NoToolCalls_GoesIdle(t *testing.T) {
	m := baseModel()
	m.state = stateStreaming
	m.items = append(m.items, displayItem{kind: itemAssistant, content: "Hi"})
	m.streamingItemIdx = 0

	finalMsg := ai.Message{Role: "assistant", Content: "Hi"}
	ch := chanOf()
	m, _ = update(t, m, streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Done: true, Msg: finalMsg}})

	assert.Equal(t, stateIdle, m.state)
	assert.Equal(t, -1, m.streamingItemIdx)
	require.Len(t, m.messages, 2) // system + assistant
	assert.Equal(t, "assistant", m.messages[1].Role)
}

// --- streamChunkMsg: Done (with tool calls, unapproved) ---

func TestUpdate_StreamDone_WithToolCalls_AwaitingApproval(t *testing.T) {
	tm := &mockToolManager{}
	session := chat.NewSession(tm, nil, nil) // no auto-approve
	m := initialModel(context.Background(), nil, session, nil, "m", nil, "dark", nil)
	m.state = stateStreaming

	tc := ai.ToolCall{ID: "c1", Name: "my_tool", Arguments: map[string]any{"x": 1}}
	finalMsg := ai.Message{Role: "assistant", ToolCalls: []ai.ToolCall{tc}}
	ch := chanOf()
	m, _ = update(t, m, streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Done: true, Msg: finalMsg}})

	assert.Equal(t, stateAwaitingApproval, m.state)
	require.NotNil(t, m.currentCall)
	assert.Equal(t, "my_tool", m.currentCall.Name)
	// A tool-call display item should have been added.
	require.Len(t, m.items, 1)
	assert.Equal(t, itemToolCall, m.items[0].kind)
}

// --- streamChunkMsg: Done (with tool calls, pre-approved) ---

func TestUpdate_StreamDone_WithApprovedToolCall_StartsCallingTool(t *testing.T) {
	tm := &mockToolManager{}
	session := chat.NewSession(tm, []string{"approved_tool"}, nil)
	m := initialModel(context.Background(), nil, session, nil, "m", nil, "dark", nil)
	m.state = stateStreaming

	tc := ai.ToolCall{ID: "c1", Name: "approved_tool", Arguments: nil}
	finalMsg := ai.Message{Role: "assistant", ToolCalls: []ai.ToolCall{tc}}
	ch := chanOf()
	m, cmd := update(t, m, streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Done: true, Msg: finalMsg}})

	assert.Equal(t, stateCallingTool, m.state)
	assert.Nil(t, m.currentCall)
	assert.Equal(t, "approved_tool", m.callingToolName)
	assert.NotNil(t, cmd) // doCallTool command returned
}

// --- streamChunkMsg: Err ---

func TestUpdate_StreamError_GoesIdleWithErrorItem(t *testing.T) {
	m := baseModel()
	m.state = stateThinking

	ch := chanOf()
	m, _ = update(t, m, streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Err: assert.AnError}})

	assert.Equal(t, stateIdle, m.state)
	assert.Equal(t, -1, m.streamingItemIdx)
	require.Len(t, m.items, 1)
	assert.Equal(t, itemError, m.items[0].kind)
}

func TestUpdate_StreamError_DuringStreaming_EmbedsInExistingItem(t *testing.T) {
	m := baseModel()
	m.state = stateStreaming
	m.items = append(m.items, displayItem{kind: itemAssistant, content: "partial"})
	m.streamingItemIdx = 0

	ch := chanOf()
	m, _ = update(t, m, streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Err: assert.AnError}})

	assert.Equal(t, stateIdle, m.state)
	assert.Equal(t, -1, m.streamingItemIdx)
	assert.Contains(t, m.items[0].content, "stream error")
	require.Len(t, m.items, 1) // no new item added; error embedded
}

// --- Tool approval key handler ---

func TestUpdate_ToolApproval_Y_StartsCallingTool(t *testing.T) {
	tm := &mockToolManager{}
	session := chat.NewSession(tm, nil, nil)
	m := initialModel(context.Background(), nil, session, nil, "m", nil, "dark", nil)
	tc := ai.ToolCall{ID: "c1", Name: "some_tool"}
	m.state = stateAwaitingApproval
	m.currentCall = &tc
	m.toolCallIdx[tc.ID] = 0
	m.items = append(m.items, displayItem{kind: itemToolCall, toolStatus: toolStatusPending})

	// First key press stages the choice.
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("y")})
	assert.Equal(t, "y", m.pendingApprovalChoice)
	assert.Equal(t, stateAwaitingApproval, m.state)

	// Enter confirms.
	m, cmd := update(t, m, tea.KeyMsg{Type: tea.KeyEnter})

	assert.Equal(t, stateCallingTool, m.state)
	assert.Nil(t, m.currentCall)
	assert.Equal(t, toolStatusRunning, m.items[0].toolStatus)
	assert.NotNil(t, cmd)
}

func TestUpdate_ToolApproval_A_ApprovesForSessionAndCalls(t *testing.T) {
	tm := &mockToolManager{}
	session := chat.NewSession(tm, nil, nil)
	m := initialModel(context.Background(), nil, session, nil, "m", nil, "dark", nil)
	tc := ai.ToolCall{ID: "c1", Name: "some_tool"}
	m.state = stateAwaitingApproval
	m.currentCall = &tc
	m.toolCallIdx[tc.ID] = 0
	m.items = append(m.items, displayItem{kind: itemToolCall, toolStatus: toolStatusPending})

	// Stage "a" then confirm with Enter.
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("a")})
	assert.Equal(t, "a", m.pendingApprovalChoice)
	m, cmd := update(t, m, tea.KeyMsg{Type: tea.KeyEnter})

	assert.Equal(t, stateCallingTool, m.state)
	assert.True(t, session.IsApproved("some_tool"), "tool should now be session-approved")
	assert.NotNil(t, cmd)
}

func TestUpdate_ToolApproval_N_DeniesAndProcessesNext(t *testing.T) {
	sc := &mockStreamCompleter{}
	// processNextCall with no more pending calls → doStartStream → needs a client
	sc.On("CompleteStream", mock.Anything, mock.Anything, mock.Anything).
		Return(chanOf(ai.StreamChunk{Done: true, Msg: ai.Message{Role: "assistant", Content: "ok"}}))

	tm := &mockToolManager{}
	session := chat.NewSession(tm, nil, nil)
	m := initialModel(context.Background(), sc, session, nil, "m", nil, "dark", nil)
	tc := ai.ToolCall{ID: "c1", Name: "some_tool"}
	m.state = stateAwaitingApproval
	m.currentCall = &tc
	m.toolCallIdx[tc.ID] = 0
	m.items = append(m.items, displayItem{kind: itemToolCall, toolStatus: toolStatusPending})

	// Stage "n" then confirm.
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("n")})
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyEnter})

	assert.Equal(t, toolStatusDenied, m.items[0].toolStatus)
	assert.Nil(t, m.currentCall)
	// After denial with no more pending, processNextCall starts a new stream → stateThinking.
	assert.Equal(t, stateThinking, m.state)
	// Denial message should be in the conversation history.
	require.GreaterOrEqual(t, len(m.messages), 2)
	last := m.messages[len(m.messages)-1]
	assert.Equal(t, "tool", last.Role)
	assert.Equal(t, "c1", last.ToolCallID)
}

// --- toolResultMsg ---

func TestUpdate_ToolResult_Success_ProcessesNext(t *testing.T) {
	sc := &mockStreamCompleter{}
	sc.On("CompleteStream", mock.Anything, mock.Anything, mock.Anything).
		Return(chanOf(ai.StreamChunk{Done: true, Msg: ai.Message{Role: "assistant", Content: "done"}}))

	tm := &mockToolManager{}
	session := chat.NewSession(tm, nil, nil)
	m := initialModel(context.Background(), sc, session, nil, "m", nil, "dark", nil)
	m.state = stateCallingTool
	m.callingToolName = "my_tool"
	m.toolCallIdx["c1"] = 0
	m.items = append(m.items, displayItem{kind: itemToolCall, toolStatus: toolStatusRunning})

	m, _ = update(t, m, toolResultMsg{callID: "c1", toolName: "my_tool", result: "result text"})

	assert.Equal(t, toolStatusDone, m.items[0].toolStatus)
	assert.Equal(t, "", m.callingToolName)
	// Tool result appended to messages.
	last := m.messages[len(m.messages)-1]
	assert.Equal(t, "tool", last.Role)
	assert.Equal(t, "result text", last.Content)
}

func TestUpdate_ToolResult_Failure_InformsAI(t *testing.T) {
	sc := &mockStreamCompleter{}
	sc.On("CompleteStream", mock.Anything, mock.Anything, mock.Anything).
		Return(chanOf(ai.StreamChunk{Done: true, Msg: ai.Message{Role: "assistant", Content: "ok"}}))

	tm := &mockToolManager{}
	session := chat.NewSession(tm, nil, nil)
	m := initialModel(context.Background(), sc, session, nil, "m", nil, "dark", nil)
	m.state = stateCallingTool
	m.callingToolName = "bad_tool"
	m.toolCallIdx["c1"] = 0
	m.items = append(m.items, displayItem{kind: itemToolCall, toolStatus: toolStatusRunning})
	m.queuedPrompts = []queuedPrompt{{text: "follow-up"}}
	// Pre-seed an assistant message with a tool call so history is valid.
	m.messages = append(m.messages, ai.Message{
		Role:      "assistant",
		ToolCalls: []ai.ToolCall{{ID: "c1", Name: "bad_tool"}},
	})

	m, _ = update(t, m, toolResultMsg{callID: "c1", toolName: "bad_tool", err: assert.AnError})

	// Tool item marked failed in display.
	assert.Equal(t, toolStatusFailed, m.items[0].toolStatus)
	// Error forwarded to AI as a tool result message (now second-to-last; the
	// queued prompt was injected as a new user turn immediately after).
	toolMsg := m.messages[len(m.messages)-2]
	assert.Equal(t, "tool", toolMsg.Role)
	assert.Equal(t, "c1", toolMsg.ToolCallID)
	assert.Contains(t, toolMsg.Content, "Error:")
	// Queued prompt injected as new turn (interjection behaviour).
	userMsg := m.messages[len(m.messages)-1]
	assert.Equal(t, "user", userMsg.Role)
	assert.Equal(t, "follow-up", userMsg.Content)
	// New stream started for the injected turn.
	assert.Equal(t, stateThinking, m.state)
	// Queue was fully drained by interjection.
	assert.Empty(t, m.queuedPrompts)
	assert.Nil(t, m.pendingCalls)
	assert.Nil(t, m.currentCall)
}

// --- CtrlC always quits ---

func TestUpdate_CtrlC_Quits(t *testing.T) {
	m := baseModel()
	_, cmd := update(t, m, tea.KeyMsg{Type: tea.KeyCtrlC})
	require.NotNil(t, cmd)
	// Execute the cmd: bubbletea quit commands return tea.QuitMsg.
	msg := cmd()
	assert.IsType(t, tea.QuitMsg{}, msg)
}

// --- Enter in idle state submits message ---

func TestUpdate_Enter_InIdle_SendsMessage(t *testing.T) {
	sc := &mockStreamCompleter{}
	sc.On("CompleteStream", mock.Anything, mock.Anything, mock.Anything).
		Return(chanOf(ai.StreamChunk{Done: true, Msg: ai.Message{Role: "assistant", Content: "hi"}}))

	tm := &mockToolManager{}
	session := chat.NewSession(tm, nil, nil)
	m := initialModel(context.Background(), sc, session, nil, "m", nil, "dark", nil)
	m.state = stateIdle

	// Simulate typing "hello" then pressing Enter.
	for _, r := range "hello" {
		m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	m, cmd := update(t, m, tea.KeyMsg{Type: tea.KeyEnter})

	assert.Equal(t, stateThinking, m.state)
	assert.Equal(t, "", m.input.Value()) // input cleared
	require.NotNil(t, cmd)
	// User message should be appended.
	require.Len(t, m.items, 1)
	assert.Equal(t, itemUser, m.items[0].kind)
	assert.Equal(t, "hello", m.items[0].content)
}

func TestUpdate_Enter_EmptyInput_NoOp(t *testing.T) {
	m := baseModel()
	m.state = stateIdle
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyEnter})
	assert.Equal(t, stateIdle, m.state)
	assert.Empty(t, m.items)
}

// --- Prompt queueing ---

func TestUpdate_Queue_EnterWhileBusy_AddsToQueue(t *testing.T) {
	m := baseModel()
	m.state = stateThinking

	// Type "hello" into the input.
	for _, r := range "hello" {
		m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyEnter})

	require.Len(t, m.queuedPrompts, 1)
	assert.Equal(t, "hello", m.queuedPrompts[0].text)
	assert.Equal(t, "", m.input.Value(), "input should be cleared after queueing")
	// State must not change — we're still thinking.
	assert.Equal(t, stateThinking, m.state)
}

func TestUpdate_Queue_MultiplePrompts_FIFO(t *testing.T) {
	m := baseModel()
	m.state = stateStreaming

	for _, text := range []string{"first", "second", "third"} {
		for _, r := range text {
			m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		}
		m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyEnter})
	}

	require.Len(t, m.queuedPrompts, 3)
	assert.Equal(t, "first", m.queuedPrompts[0].text)
	assert.Equal(t, "second", m.queuedPrompts[1].text)
	assert.Equal(t, "third", m.queuedPrompts[2].text)
}

func TestUpdate_Queue_EscWithText_ClearsInput(t *testing.T) {
	m := baseModel()
	m.state = stateThinking

	for _, r := range "hello" {
		m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyEsc})

	assert.Equal(t, "", m.input.Value())
	assert.Empty(t, m.queuedPrompts)
}

func TestUpdate_Queue_EscEmptyInput_RemovesLastQueued(t *testing.T) {
	m := baseModel()
	m.state = stateStreaming
	m.queuedPrompts = []queuedPrompt{{text: "first"}, {text: "second"}}

	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyEsc})

	require.Len(t, m.queuedPrompts, 1)
	assert.Equal(t, "first", m.queuedPrompts[0].text)
}

func TestUpdate_Queue_EscEmptyInputNoQueue_NoOp(t *testing.T) {
	m := baseModel()
	m.state = stateThinking
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyEsc})
	assert.Empty(t, m.queuedPrompts)
	assert.Equal(t, stateThinking, m.state)
}

func TestUpdate_Queue_EnterEmpty_NotQueued(t *testing.T) {
	m := baseModel()
	m.state = stateThinking
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyEnter})
	assert.Empty(t, m.queuedPrompts)
}

func TestUpdate_ProcessQueueOrIdle_WithQueue_StartsNextTurn(t *testing.T) {
	sc := &mockStreamCompleter{}
	sc.On("CompleteStream", mock.Anything, mock.Anything, mock.Anything).
		Return(chanOf(ai.StreamChunk{Done: true, Msg: ai.Message{Role: "assistant", Content: "pong"}}))

	tm := &mockToolManager{}
	session := chat.NewSession(tm, nil, nil)
	m := initialModel(context.Background(), sc, session, nil, "m", nil, "dark", nil)
	m.state = stateStreaming
	m.queuedPrompts = []queuedPrompt{{text: "next question"}}

	// Simulate stream completing with no tool calls → processQueueOrIdle fires.
	finalMsg := ai.Message{Role: "assistant", Content: "answer"}
	ch := chanOf()
	m, cmd := update(t, m, streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Done: true, Msg: finalMsg}})

	// Queue drained, next prompt sent.
	assert.Empty(t, m.queuedPrompts)
	assert.Equal(t, stateThinking, m.state)
	// The completed assistant message plus the queued user message should appear.
	require.Len(t, m.items, 2)
	assert.Equal(t, itemAssistant, m.items[0].kind)
	assert.Equal(t, "answer", m.items[0].content)
	assert.Equal(t, itemUser, m.items[1].kind)
	assert.Equal(t, "next question", m.items[1].content)
	assert.NotNil(t, cmd)
}

func TestUpdate_ProcessQueueOrIdle_NoQueue_GoesIdle(t *testing.T) {
	m := baseModel()
	m.state = stateStreaming

	finalMsg := ai.Message{Role: "assistant", Content: "done"}
	ch := chanOf()
	m, _ = update(t, m, streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Done: true, Msg: finalMsg}})

	assert.Equal(t, stateIdle, m.state)
	assert.Empty(t, m.queuedPrompts)
	// Content arrived via Done (no prior deltas) — should be shown.
	require.Len(t, m.items, 1)
	assert.Equal(t, itemAssistant, m.items[0].kind)
	assert.Equal(t, "done", m.items[0].content)
}

func TestUpdate_Queue_SurvivesToolCalls(t *testing.T) {
	// Proves the new interjection behaviour: a queued prompt is injected as a
	// new user turn immediately after the current tool-call batch completes,
	// rather than waiting for the full agent turn to finish.

	sc := &mockStreamCompleter{}
	tm := &mockToolManager{}
	session := chat.NewSession(tm, []string{"my_tool"}, nil)
	m := initialModel(context.Background(), sc, session, nil, "m", nil, "dark", nil)

	// 1. Queue a follow-up while the first stream is in-flight.
	m.state = stateStreaming
	for _, r := range "follow-up" {
		m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyEnter})
	require.Len(t, m.queuedPrompts, 1)

	// 2. First stream finishes with a tool call.
	tc := ai.ToolCall{ID: "c1", Name: "my_tool"}
	firstDone := ai.Message{Role: "assistant", ToolCalls: []ai.ToolCall{tc}}
	ch := chanOf()
	m, _ = update(t, m, streamChunkMsg{ch: ch, chunk: ai.StreamChunk{Done: true, Msg: firstDone}})
	// Queue must still be intact — tool calls haven't finished yet.
	require.Len(t, m.queuedPrompts, 1, "queue must survive tool-call processing")
	assert.Equal(t, stateCallingTool, m.state)

	// 3. Tool result comes back; processNextCall sees an empty pendingCalls and a
	// non-empty queue → injects the queued prompt as a new turn immediately.
	sc.On("CompleteStream", mock.Anything, mock.Anything, mock.Anything).
		Return(chanOf(ai.StreamChunk{Done: true, Msg: ai.Message{Role: "assistant", Content: "all done"}}))
	m, _ = update(t, m, toolResultMsg{callID: "c1", toolName: "my_tool", result: "tool output"})
	// Queue drained — queued prompt was injected as a new turn.
	assert.Empty(t, m.queuedPrompts, "queue must be drained after tool batch completes (interjection)")
	assert.Equal(t, stateThinking, m.state, "new turn should start immediately after interjection")
}

// --- inputPlaceholder ---

func TestInputPlaceholder_States(t *testing.T) {
	assert.Contains(t, inputPlaceholder(stateIdle), "Enter")
	assert.Contains(t, inputPlaceholder(stateThinking), "Queue")
	assert.Contains(t, inputPlaceholder(stateStreaming), "Queue")
	assert.Contains(t, inputPlaceholder(stateCallingTool), "Queue")
	assert.Contains(t, inputPlaceholder(stateAwaitingApproval), "y")
}

// --- compactionKeepTurns ---

func TestCompactionKeepTurns(t *testing.T) {
	for i := 8000; i < 384000; i += 8000 {
		t.Logf("Context size %d -> %d turns", i, compactionKeepTurns(i))
	}
}

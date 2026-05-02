package copilot

import (
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/icedream/werkler/internal/ai"
	"github.com/stretchr/testify/require"
)

func githubTokenOrSkip(t *testing.T) string {
	t.Helper()
	if os.Getenv("WERKLER_INTEGRATION") == "" {
		t.Skip("Skipping Copilot integration test: set WERKLER_INTEGRATION=1 to run.")
		return ""
	}
	tok, err := LoadGitHubToken()
	if err != nil {
		t.Fatalf("failed to load GitHub token: %v", err)
	}
	if tok == nil || tok.AccessToken == "" {
		t.Skipf("No GitHub token found; skipping Copilot integration test.")
	}
	return tok.AccessToken
}

func TestCopilotMuxClient_ListModels_And_Completion(t *testing.T) {
	t.Parallel()
	ctx := t.Context()

	accessToken := githubTokenOrSkip(t)
	transport := ai.NewReasoningAliasTransport(NewTransport(accessToken))

	mux := NewCopilotMuxClient("", &http.Client{Transport: transport}, false)
	models, err := mux.ListModels(ctx)
	require.NoError(t, err)
	require.NotEmpty(t, models, "should return models")

	var testModel string
	for _, m := range models {
		if strings.Contains(strings.ToLower(m.Model), "gpt") {
			testModel = m.Model
			break
		}
	}
	if testModel == "" {
		t.Skipf("No GPT model found in Copilot model list for test.")
	}

	mux.SetModel(ai.ModelItem{Model: testModel})
	// Sanity: backend selection shouldn't panic or fail

	reqMsgs := []ai.Message{{Role: "user", Content: "Say hello as a test."}}
	resp, err := mux.Complete(ctx, reqMsgs, nil)
	require.NoError(t, err)
	require.Equal(t, "assistant", resp.Role)
	require.NotEmpty(t, resp.Content)
}

func TestCopilotMuxClient_BackendSwitch(t *testing.T) {
	t.Parallel()
	ctx := t.Context()

	accessToken := githubTokenOrSkip(t)
	transport := ai.NewReasoningAliasTransport(NewTransport(accessToken))
	mux := NewCopilotMuxClient("", &http.Client{Transport: transport}, false)
	models, err := mux.ListModels(ctx)
	require.NoError(t, err)
	require.NotEmpty(t, models)

	var chatModel, respModel string
	for _, m := range models {
		if strings.Contains(m.Model, "responses") {
			respModel = m.Model
		} else if strings.Contains(m.Model, "gpt") {
			chatModel = m.Model
		}
		if chatModel != "" && respModel != "" {
			break
		}
	}
	if chatModel == "" || respModel == "" {
		t.Skipf("Both chat and responses endpoint models required for mux switch test.")
	}

	mux.SetModel(ai.ModelItem{Model: chatModel})
	reqMsgs := []ai.Message{{Role: "user", Content: "Test for chat endpoint model."}}
	resp, err := mux.Complete(ctx, reqMsgs, nil)
	require.NoError(t, err)
	require.Equal(t, "assistant", resp.Role)
	require.NotEmpty(t, resp.Content)

	mux.SetModel(ai.ModelItem{Model: respModel})
	reqMsgs = []ai.Message{{Role: "user", Content: "Test for responses endpoint model."}}
	resp, err = mux.Complete(ctx, reqMsgs, nil)
	require.NoError(t, err)
	require.Equal(t, "assistant", resp.Role)
	require.NotEmpty(t, resp.Content)
}

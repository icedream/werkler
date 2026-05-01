package modeleval

import (
	"fmt"
	"strings"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/chat"
)

// configuredServersSection builds the system-prompt section werkler injects
// for lazy-connect MCP servers, replicating buildStreamMessages logic.
func configuredServersSection(servers []string) string {
	var sb strings.Builder
	sb.WriteString("## Configured MCP servers (not yet connected)\n")
	sb.WriteString("These servers are available but not yet connected. ")
	sb.WriteString("When the user's current request requires tools from one of these servers, ")
	sb.WriteString("call `connect_server` for it **immediately** — do not ask for permission, ")
	sb.WriteString("do not explain what you are about to do, just call it. ")
	sb.WriteString("Do NOT connect servers whose tools are not needed for the current task.\n")
	for _, name := range servers {
		sb.WriteString("- `")
		sb.WriteString(name)
		sb.WriteString("`\n")
	}
	return sb.String()
}

// connectServerTool builds the connect_server tool definition with the given
// server names listed in the description, mirroring werkler's tools.Manager.
func connectServerTool(servers []string) ai.ToolDefinition {
	return ai.ToolDefinition{
		Name: "connect_server",
		Description: "Connect to a configured MCP server to make its tools available. " +
			"Call this immediately when the user's request requires tools from that server — " +
			"do not ask for permission first and do not connect servers unrelated to the current task. " +
			"After connecting, the server's tools will be listed in the result. " +
			"Available servers: " + strings.Join(servers, ", ") + ".",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{
					"type":        "string",
					"description": "Name of the server to connect",
				},
			},
			"required": []string{"name"},
		},
	}
}

// minimalBuiltins returns a representative subset of werkler's built-in tools
// sufficient to exercise tool-calling without noise from the full list.
func minimalBuiltins() []ai.ToolDefinition {
	return []ai.ToolDefinition{
		{
			Name:        "file_read",
			Description: "Read the contents of a file.",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"path": map[string]any{"type": "string"}},
				"required":   []string{"path"},
			},
		},
		{
			Name:        "file_list",
			Description: "List the contents of a directory.",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"path": map[string]any{"type": "string", "description": "Directory to list; defaults to current directory"}},
				"required":   []string{},
			},
		},
		{
			Name:        "process_start",
			Description: "Start a subprocess and return its output.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{"type": "string"},
					"args":    map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				},
				"required": []string{"command"},
			},
		},
		{
			Name:        "ask_user",
			Description: "Ask the user a question and wait for their answer.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"question": map[string]any{"type": "string"},
					"choices":  map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				},
				"required": []string{"question"},
			},
		},
	}
}

// syntheticWeatherTool is a simple fictional tool for basic tool-call tests.
var syntheticWeatherTool = ai.ToolDefinition{
	Name:        "get_weather",
	Description: "Get the current weather for a city.",
	InputSchema: map[string]any{
		"type":       "object",
		"properties": map[string]any{"city": map[string]any{"type": "string", "description": "City name"}},
		"required":   []string{"city"},
	},
}

// cloudflareServers is the representative set of Cloudflare MCP server names
// that replicate a real ck-cauldron-style werkler config.
var cloudflareServers = []string{
	"cloudflare-docs",
	"cloudflare-dns-analytics",
	"cloudflare-radar",
	"cloudflare-auditlogs",
	"cloudflare-graphql",
	"cloudflare-browser-rendering",
}

// AllCases returns the full set of model evaluation test cases.
func AllCases() []*TestCase {
	return []*TestCase{
		caseTextResponse(),
		caseNotEmpty(),
		caseBasicToolCall(),
		caseToolCallWithPossibleTextPrefix(),
		caseConnectServerExact(),
		caseConnectServerGenericName(),
		caseNoSpuriousConnect(),
		caseCompactSummaryQuality(),
		caseCompactNoRefusal(),
	}
}

// caseTextResponse verifies the model produces a plain text answer when no
// tools are available or needed.
func caseTextResponse() *TestCase {
	msgs := chat.NewConversation()
	msgs = append(msgs, ai.Message{Role: "user", Content: "What is the capital of France? Reply in one sentence."})
	return &TestCase{
		Name:        "text-response",
		Description: "Model responds with plain text when no tools are needed",
		Messages:    msgs,
		Tools:       nil,
		Check:       CheckHasContent(),
	}
}

// caseNotEmpty verifies the model never produces a completely empty response
// (the silent-drop bug seen with Ollama/Mistral).
func caseNotEmpty() *TestCase {
	msgs := chat.NewConversation()
	msgs = append(msgs, ai.Message{Role: "user", Content: "List files in the current directory."})
	return &TestCase{
		Name:        "no-empty-response",
		Description: "Response is never completely empty (content or tool_calls must be present)",
		Messages:    msgs,
		Tools:       minimalBuiltins(),
		Check:       CheckNotEmpty(),
		Repeat:      5,
	}
}

// caseBasicToolCall verifies the model calls a single simple tool when asked.
func caseBasicToolCall() *TestCase {
	msgs := chat.NewConversation()
	msgs = append(msgs, ai.Message{Role: "user", Content: "What is the weather like in Berlin right now?"})
	return &TestCase{
		Name:        "basic-tool-call",
		Description: "Model calls a simple tool (get_weather) when explicitly needed",
		Messages:    msgs,
		Tools:       []ai.ToolDefinition{syntheticWeatherTool},
		Check:       CheckToolCall("get_weather"),
	}
}

// caseToolCallWithPossibleTextPrefix verifies a tool call is produced even when
// the model may write brief text before it. Ollama's Ministral parser requires
// some text before [TOOL_CALLS]; this case catches regressions if the system
// prompt suppresses that preamble too aggressively.
func caseToolCallWithPossibleTextPrefix() *TestCase {
	msgs := chat.NewConversation()
	msgs = append(msgs, ai.Message{Role: "user", Content: "Read the file /etc/hostname and tell me what's in it."})
	return &TestCase{
		Name:        "tool-call-with-text-prefix",
		Description: "Tool call is produced (content before tool call is acceptable)",
		Messages:    msgs,
		Tools:       minimalBuiltins(),
		Check:       CheckToolCall("file_read"),
		Repeat:      3,
	}
}

// caseConnectServerExact verifies the model calls connect_server when the user
// asks for something that requires a configured-but-not-connected server,
// using a task where the required server name appears explicitly.
func caseConnectServerExact() *TestCase {
	section := configuredServersSection(cloudflareServers)
	msgs := chat.NewConversation(section)
	msgs = append(msgs, ai.Message{
		Role:    "user",
		Content: "Can you query the cloudflare-dns-analytics server to list DNS zones for my account?",
	})
	tools := append(minimalBuiltins(), connectServerTool(cloudflareServers))
	return &TestCase{
		Name:        "connect-server-exact",
		Description: "Model calls connect_server when the exact server name appears in the request",
		Messages:    msgs,
		Tools:       tools,
		Check: All(
			CheckNotEmpty(),
			CheckToolCall("connect_server"),
			CheckToolCallArg("connect_server", "cloudflare"),
		),
		Repeat: 3,
	}
}

// caseConnectServerGenericName verifies the model calls connect_server when
// asked about "Cloudflare" without specifying the exact server name — the
// most common real-world failure pattern observed with devstral on Ollama.
func caseConnectServerGenericName() *TestCase {
	section := configuredServersSection(cloudflareServers)
	msgs := chat.NewConversation(section)
	msgs = append(msgs, ai.Message{
		Role:    "user",
		Content: "Can you check which DNS domains are registered on our Cloudflare account?",
	})
	tools := append(minimalBuiltins(), connectServerTool(cloudflareServers))
	return &TestCase{
		Name:        "connect-server-generic",
		Description: "Model calls connect_server when the user asks about a server by generic name (e.g. 'Cloudflare')",
		Messages:    msgs,
		Tools:       tools,
		Check: All(
			CheckNotEmpty(),
			CheckToolCall("connect_server"),
			CheckToolCallArg("connect_server", "cloudflare"),
		),
		Repeat: 5,
	}
}

// caseNoSpuriousConnect verifies the model does NOT call connect_server when
// the user asks something that can be answered with built-in tools only.
func caseNoSpuriousConnect() *TestCase {
	section := configuredServersSection(cloudflareServers)
	msgs := chat.NewConversation(section)
	msgs = append(msgs, ai.Message{
		Role:    "user",
		Content: "List the files in the current directory.",
	})
	tools := append(minimalBuiltins(), connectServerTool(cloudflareServers))
	return &TestCase{
		Name:        "no-spurious-connect",
		Description: "Model does NOT call connect_server for tasks that need only built-in tools",
		Messages:    msgs,
		Tools:       tools,
		Check: func(resp ai.Message) error {
			for _, tc := range resp.ToolCalls {
				if tc.Name == "connect_server" {
					return fmt.Errorf("spurious connect_server call (args=%v)", tc.Arguments)
				}
			}
			return CheckNotEmpty()(resp)
		},
	}
}

// compactionSystemPrompt is the summarizer system prompt used by doCompact,
// reproduced here so modeltest can exercise it against a live model without
// importing internal/ui.
const compactionSystemPrompt = "You are a conversation summarizer. " +
	"Write a concise but information-dense summary of the conversation transcript below. " +
	"You MUST preserve: the main objective and current status, " +
	"key decisions and rationale, ALL file paths created or modified (exact paths), " +
	"ALL tool calls with their key arguments and outcomes, " +
	"ALL unresolved errors and open items, and the last clear user intent. " +
	"Write in past tense. Only verifiable facts — no opinion. " +
	"Note: reasoning/thinking tokens are excluded from this transcript."

// syntheticTranscriptForCompact is a realistic multi-turn transcript containing
// a known file path and tool calls with arguments. Used to verify the model
// preserves specific information when acting as a conversation summarizer.
//
// The file path "/src/server/handler.go" and the unresolved error must both
// appear in a correct summary — these are the canary values checked by
// caseCompactSummaryQuality.
const syntheticTranscriptForCompact = `User: Can you add a logging statement to /src/server/handler.go at the top of the HandleRequest function?

Assistant called tool "file_read" args: {"path":"/src/server/handler.go"}
Tool result: package server

func HandleRequest(w http.ResponseWriter, r *http.Request) {
	// existing code
}

Assistant called tool "file_edit" args: {"path":"/src/server/handler.go","old_str":"func HandleRequest(w http.ResponseWriter, r *http.Request) {","new_str":"func HandleRequest(w http.ResponseWriter, r *http.Request) {\n\tlog.Printf(\"HandleRequest called\")"}
Tool result: error: old_str not found in file

User: It seems the edit failed. Can you try again?`

// caseCompactSummaryQuality sends the real compaction summarizer prompt with a
// synthetic transcript and verifies the model's summary preserves two canary
// values: the file path "/src/server/handler.go" and the word "error" (the
// unresolved edit failure). These are the exact pieces of context that would be
// lost across a compaction if the model ignores the summarizer instructions.
//
// This is the only compaction-related behaviour that actually depends on the
// model — all other compaction fixes (marker replacement, keepTurns loop,
// toolTokensCache) are pure Go logic covered by unit tests.
func caseCompactSummaryQuality() *TestCase {
	msgs := []ai.Message{
		{
			Role:    "system",
			Content: compactionSystemPrompt,
		},
		{
			Role:    "user",
			Content: "Summarize this conversation (incorporating the prior summary if present):\n\n" + syntheticTranscriptForCompact,
		},
	}
	return &TestCase{
		Name:        "compact-summary-quality",
		Description: "Summarizer preserves file paths and unresolved errors from the transcript",
		Messages:    msgs,
		Tools:       nil,
		Check: All(
			CheckHasContent(),
			CheckResponseContains("/src/server/handler.go",
				"summary must include the exact file path from the transcript"),
			CheckResponseContains("error",
				"summary must mention the unresolved tool error"),
		),
		Repeat: 3,
	}
}

// caseCompactNoRefusal verifies the model does not refuse or return an empty
// response when given the compaction summarizer prompt. Some smaller models
// protest when asked to summarise role:tool messages or produce no output at
// all when the system prompt is terse.
func caseCompactNoRefusal() *TestCase {
	msgs := []ai.Message{
		{
			Role:    "system",
			Content: compactionSystemPrompt,
		},
		{
			Role:    "user",
			Content: "Summarize this conversation (incorporating the prior summary if present):\n\n" + syntheticTranscriptForCompact,
		},
	}
	return &TestCase{
		Name:        "compact-no-refusal",
		Description: "Model does not refuse or return empty output when given the compaction summarizer prompt",
		Messages:    msgs,
		Tools:       nil,
		Check:       CheckNotEmpty(),
		Repeat:      3,
	}
}

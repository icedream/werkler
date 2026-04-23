package chat

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/mcp"
)

const systemPrompt = `You are Werkler, an AI assistant for software developers.
You help with tasks like writing and reviewing code, designing software, drafting tickets, and technical documentation.
When you need information from files or the filesystem, use the tools available to you.
Be concise and precise. Ask for clarification if a request is ambiguous.`

// write prints to l.out, treating write errors as fatal since they indicate
// a broken terminal/pipe from which the session cannot recover.
func (l *Loop) write(format string, args ...any) {
	if _, err := fmt.Fprintf(l.out, format, args...); err != nil {
		panic(fmt.Sprintf("output write error: %v", err))
	}
}

func (l *Loop) writeln(s string) {
	l.write("%s\n", s)
}

type ToolCaller interface {
	CallTool(ctx context.Context, name string, args map[string]any) (string, error)
}

// ToolLister can enumerate available tools as AI definitions.
type ToolLister interface {
	Tools(ctx context.Context) ([]ai.ToolDefinition, error)
}

// Loop holds the state for an interactive chat session.
type Loop struct {
	client      *ai.Client
	tools       ToolLister
	caller      ToolCaller
	messages    []ai.Message
	autoApprove []string // glob patterns for auto-approved tool names
	// sessionApproved holds tool names the user approved for the whole session.
	sessionApproved map[string]bool

	in  *bufio.Reader
	out io.Writer
}

// NewLoop creates a new chat loop.
// autoApproveGlobs is a list of glob patterns for tool names that skip confirmation.
func NewLoop(client *ai.Client, manager *mcp.Manager, autoApproveGlobs []string, in io.Reader, out io.Writer) *Loop {
	return &Loop{
		client:          client,
		tools:           manager,
		caller:          manager,
		autoApprove:     autoApproveGlobs,
		sessionApproved: make(map[string]bool),
		messages: []ai.Message{
			{Role: "system", Content: systemPrompt},
		},
		in:  bufio.NewReader(in),
		out: out,
	}
}

// Run starts the interactive REPL. It blocks until the user exits or ctx is cancelled.
func (l *Loop) Run(ctx context.Context) error {
	l.writeln("Werkler — type your message, or /exit to quit.")
	l.writeln("")

	for {
		l.write("You> ")
		line, err := l.in.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				l.writeln("")
				return nil
			}
			return fmt.Errorf("reading input: %w", err)
		}

		input := strings.TrimSpace(line)
		if input == "" {
			continue
		}
		if input == "/exit" || input == "/quit" {
			l.writeln("Goodbye.")
			return nil
		}

		if err := l.handleTurn(ctx, input); err != nil {
			l.write("Error: %v\n\n", err)
		}
	}
}

// maxAgentSteps is the maximum number of AI→tool round-trips per user turn,
// preventing runaway loops from misbehaving or looping models.
const maxAgentSteps = 50

func (l *Loop) handleTurn(ctx context.Context, userInput string) error {
	l.messages = append(l.messages, ai.Message{
		Role:    "user",
		Content: userInput,
	})

	tools, err := l.tools.Tools(ctx)
	if err != nil {
		return fmt.Errorf("fetching tools: %w", err)
	}

	for step := 0; step < maxAgentSteps; step++ {
		msg, err := l.client.Complete(ctx, l.messages, tools)
		if err != nil {
			return fmt.Errorf("AI completion: %w", err)
		}
		l.messages = append(l.messages, msg)

		if len(msg.ToolCalls) == 0 {
			// Final response — print it.
			l.write("\nWerkler> %s\n\n", msg.Content)
			return nil
		}

		// Execute each tool call in sequence.
		for _, tc := range msg.ToolCalls {
			result, err := l.executeToolCall(ctx, tc)
			if err != nil {
				return err
			}
			l.messages = append(l.messages, ai.Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Content:    result,
			})
		}
	}

	return fmt.Errorf("agent exceeded %d steps without a final response — aborting", maxAgentSteps)
}

func (l *Loop) executeToolCall(ctx context.Context, tc ai.ToolCall) (string, error) {
	argsJSON, _ := json.MarshalIndent(tc.Arguments, "  ", "  ")

	if !l.isApproved(tc.Name) {
		l.write("\n[Tool request] %s\n  Arguments: %s\n", tc.Name, argsJSON)
		approved, always, err := l.askApproval()
		if err != nil {
			return "", err
		}
		if !approved {
			return "(tool call was denied by the user)", nil
		}
		if always {
			l.sessionApproved[tc.Name] = true
		}
	}

	result, err := l.caller.CallTool(ctx, tc.Name, tc.Arguments)
	if err != nil {
		// Return errors as tool results so the AI can see and react to them.
		return fmt.Sprintf("error: %v", err), nil
	}
	return result, nil
}

// isApproved returns true if the tool should run without prompting.
func (l *Loop) isApproved(toolName string) bool {
	if l.sessionApproved[toolName] {
		return true
	}
	for _, pattern := range l.autoApprove {
		if matched, _ := filepath.Match(pattern, toolName); matched {
			return true
		}
	}
	return false
}

// askApproval prompts the user and returns (approved, alwaysApprove, error).
func (l *Loop) askApproval() (approved, always bool, err error) {
	for {
		l.write("Allow? [y]es / [n]o / [a]lways: ")
		line, err := l.in.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				return false, false, nil
			}
			return false, false, fmt.Errorf("reading input: %w", err)
		}
		switch strings.TrimSpace(strings.ToLower(line)) {
		case "y", "yes":
			return true, false, nil
		case "a", "always":
			return true, true, nil
		case "n", "no", "":
			return false, false, nil
		default:
			l.writeln("Please enter y, n, or a.")
		}
	}
}

// PrintContext writes the current conversation history for debugging.
func (l *Loop) PrintContext(w io.Writer) {
	for _, m := range l.messages {
		_, _ = fmt.Fprintf(w, "[%s] %s\n", m.Role, m.Content)
	}
}

// SetInput replaces the input reader (useful for testing).
func (l *Loop) SetInput(r io.Reader) {
	l.in = bufio.NewReader(r)
}

// SetOutput replaces the output writer (useful for testing).
func (l *Loop) SetOutput(w io.Writer) {
	l.out = w
}

// Messages returns a copy of the current conversation history.
func (l *Loop) Messages() []ai.Message {
	out := make([]ai.Message, len(l.messages))
	copy(out, l.messages)
	return out
}

// Reset clears the conversation history, keeping the system prompt.
func (l *Loop) Reset() {
	l.messages = []ai.Message{
		{Role: "system", Content: systemPrompt},
	}
	l.sessionApproved = make(map[string]bool)
	l.writeln("[Conversation reset]")
}

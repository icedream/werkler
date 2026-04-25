package chat

import (
	"context"
	"fmt"
	"io"

	"github.com/icedream/werkler/internal/ai"
)

// RunPrompt runs a single agentic conversation turn non-interactively.
// Tools are invoked only if pre-approved via config globs or session approvals;
// unapproved tool calls receive a "not approved" result so the AI can adapt.
// If progress is non-nil, tool call events are written to it.
func RunPrompt(ctx context.Context, client ai.Completer, session *Session, prompt string, progress io.Writer) (string, error) {
	messages := NewConversation()
	messages = append(messages, ai.Message{Role: "user", Content: prompt})

	tools, err := session.Tools(ctx)
	if err != nil {
		return "", fmt.Errorf("fetching tools: %w", err)
	}

	for step := 0; step < maxAgentSteps; step++ {
		msg, err := client.Complete(ctx, messages, tools)
		if err != nil {
			return "", fmt.Errorf("AI completion: %w", err)
		}
		messages = append(messages, msg)

		if len(msg.ToolCalls) == 0 {
			return msg.Content, nil
		}

		for _, tc := range msg.ToolCalls {
			if !session.IsApproved(tc.Name) {
				if progress != nil {
					_, _ = fmt.Fprintf(progress, "[tool denied (not pre-approved in non-interactive mode): %s]\n", tc.Name)
				}
				messages = append(messages, ai.Message{
					Role:       "tool",
					ToolCallID: tc.ID,
					Content:    "(tool call was not approved — run interactively to approve tools, or add to auto_approve_tools in config)",
				})
				continue
			}

			if progress != nil {
				_, _ = fmt.Fprintf(progress, "[tool: %s]\n", tc.Name)
			}

			result, err := session.CallTool(ctx, tc)
			if err != nil {
				return "", err
			}
			messages = append(messages, ai.Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Content:    result,
			})
		}
	}

	return "", fmt.Errorf("agent exceeded %d steps without a final response — aborting", maxAgentSteps)
}

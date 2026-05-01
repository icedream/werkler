package chat

import (
	"strings"
)

// MCPServerSection generates the system message section for MCP servers that are available but not yet connected.
// Pass the list of servers and a sanitizer function for inline text (used to scrub provided name/hint/URL).
func MCPServerSection(configuredServers []string, serverHints map[string]string, serverURLs map[string]string, sanitizerFn func(string) string) string {
	if len(configuredServers) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("\n\n## Configured MCP servers (not yet connected)\n")
	sb.WriteString("These servers are available but not yet connected. " +
		"When the user's current request requires tools from one of these servers, " +
		"call `connect_server` for it **immediately** — do not ask for permission, do not explain what you are about to do, just call it. " +
		"Do NOT connect servers whose tools are not needed for the current task:\n")
	for _, name := range configuredServers {
		sb.WriteString("- `")
		sb.WriteString(sanitizerFn(name))
		sb.WriteString("`")
		hint := serverHints[name]
		url := serverURLs[name]
		switch {
		case hint != "":
			sb.WriteString(": ")
			sb.WriteString(sanitizerFn(hint))
		case url != "":
			sb.WriteString(" (")
			sb.WriteString(sanitizerFn(url))
			sb.WriteString(")")
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

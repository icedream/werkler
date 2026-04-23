## Integrations

I would love for this tool to eventually be able to talk to the following things that should be machine-friendly to access.

This is just a selection of possibilities. Of course, there's also an entire MCP registry (https://registry.modelcontextprotocol.io/).

### AI model providers

- Access to OpenAI-compatible or Anthropic/Claude-compatible APIs (remote or self-hosted AI model access)
- GitHub Copilot (using its offered AI models on a Copilot business subscription directly - see https://github.com/ericc-ch/copilot-api for a proxy that exposes these models OpenAI/Anthropic AI-compatible)
- Google Gemini (Google AI models)

### Chat/information providers

- Slack (see https://github.com/korotovsky/slack-mcp-server)
- Google Meet/Drive (Meets create chat rooms, Meet also creates transcriptions from Meets that are then stored in Google Drive)
- E-mails (e.g. again Google, e-mails can contain links to above transcriptions or relevant infrastructure information, triggered alerts from OpsGenie etc.)

### Search providers

These are search engines I'd love to have the choice to use, or maybe use multiple of them to increase chances of finding varied search results.
They don't have to be directly implemented into this solution but they may also be running through MCPs that this solution can conveniently set up/boot up and then access on demand:

- DuckDuckGo (https://github.com/nickclyde/duckduckgo-mcp-server)
- GitHub Copilot MCP (GitHub's Copilot has an official MCP server that seems to provide its own search abilities)

These are search engines that would be nice to have but I don't have any knowledge how to integrate them into a solution yet:

- Google

### MCP servers

- GitHub (access to repositories for reviewing and creating PRs, linking with tickets, committing, pushing and rebasing code)
- Atlassian (Rovo) MCP (https://github.com/atlassian/atlassian-mcp-server#architecture-and-communication)
  - includes access to Atlassian Jira (ticket management, tracking stories, tracking bugs, tracking tasks, sprints)
  - includes access to Atlassian Confluence (wiki, documentation)
- DeepWiki (https://docs.devin.ai/work-with-devin/deepwiki-mcp, search through public repository documentation)

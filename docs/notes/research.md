## Research

As part of development, I heavily intend to research existing code on the Internet to reuse reverse engineering efforts done by other people to aid with the implementation of this solution.

### Ideas

The following ideas went through my mind to research how it could be done:

- Give the AI well sandboxed/isolated access to containers or its own VMs to develop and test software
- Give the AI (or the tools it needs to run in above isolated scopes) controlled access to SSH agent/keys, passwords/password manager APIs for private resource access
- Which modes should the AI have?
  - Existing AI coding tools implement an "Ask", "Plan" and "Implementation" mode, some have varying degrees of autonomy for the "Implementation" mode to allow the AI to just work without strong supervision

### Known gotchas

- Prompting AIs uses up context, and context is limited
  - Only relevant knowledge to the user's needed should be loaded into the context
    - It's better to only load parts of content instead of the whole content to seek out relevant information
  - Context management is key: The solution must be capable of leading long chats by intelligently compacting them
- Human control/validation is important even while striving for efficiency
  - Involving a human for checking the output increases the quality of the product
  - Involving a human also avoids the AI going rampant and doing things it should not be doing
    - That includes access to and leaking sensitive credentials - something like that should not reach context if possible and should instead be dealt with through commands runni
  - However not every single step needs human intervention
  - We should be able to define a set of common things an AI can run or access (CLI tools, paths, URLs, etc.) without causing harm
  - A lot of this could be handled using local configuration files that can be updated as we go
- When coding, obeying existing coding guidelines or known good patterns is key to decrease review cycles needed

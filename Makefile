.PHONY: fmt-docs

fmt-docs: ## Format markdown tables in docs/
	npx --yes prettier --write "docs/**/*.md"

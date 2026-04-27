BINARY := werkler
CMD     := ./cmd/werkler

.PHONY: all build test lint fmt fmt-docs

all: build

build: ## Build the werkler binary
	go build -o $(BINARY) $(CMD)

test: ## Run tests with race detector
	go test -race ./...

lint: ## Run golangci-lint
	golangci-lint run ./...

fmt: ## Format Go source code
	gofmt -w -s .

fmt-docs: ## Format markdown tables in docs/
	npx --yes prettier --write "docs/**/*.md"

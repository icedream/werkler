// modeltest evaluates AI model compatibility with werkler's tool definitions
// and system prompt by running a scripted test suite against any
// OpenAI-compatible endpoint.
//
// Usage (direct):
//
//	modeltest --base-url https://... --model my-model [--api-key sk-...] [--run case-name] [--repeat N]
//
// Usage (from werkler config):
//
//	modeltest --provider kubeai [--config ~/.config/werkler/config.toml] [--run case-name]
//	modeltest --provider copilot
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"slices"
	"strings"
	"syscall"
	"text/tabwriter"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/config"
	"github.com/icedream/werkler/internal/copilot"
	"github.com/icedream/werkler/internal/modeleval"
	"github.com/spf13/cobra"
	"net/http"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)

	err := newRootCmd().ExecuteContext(ctx)
	cancel()
	if err != nil {
		os.Exit(1)
	}
}

func newRootCmd() *cobra.Command {
	var (
		baseURL    string
		model      string
		apiKey     string
		provider   string
		configPath string
		run        []string
		repeat     int
		verbose    bool
	)

	cmd := &cobra.Command{
		Use:   "modeltest",
		Short: "Evaluate a model's tool-calling behaviour against werkler's prompts",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runEval(cmd.Context(), baseURL, model, apiKey, provider, configPath, run, repeat, verbose)
		},
	}

	cmd.Flags().StringVarP(&baseURL, "base-url", "u", "", "OpenAI-compatible API base URL (e.g. http://localhost:11434/v1)")
	cmd.Flags().StringVarP(&model, "model", "m", "", "Model name to test (overrides provider default when --provider is set)")
	cmd.Flags().StringVarP(&apiKey, "api-key", "k", envOr("OPENAI_API_KEY", ""), "API key (defaults to $OPENAI_API_KEY)")
	cmd.Flags().StringVarP(&provider, "provider", "p", "", "Use a named provider from the werkler config (e.g. copilot, kubeai)")
	cmd.Flags().StringVar(&configPath, "config", config.DefaultConfigPath(), "Path to werkler config.toml (used with --provider)")
	cmd.Flags().StringArrayVarP(&run, "run", "r", nil, "Run only the named test/scenario case(s); repeatable. Default: all.")
	cmd.Flags().IntVarP(&repeat, "repeat", "n", 0, "Override Repeat count for every case")
	cmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Print each run's full trace and error details")

	return cmd
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func buildClientFromConfig(providerName, configPath, modelOverride string) (ai.Completer, string, error) {
	cfg, err := config.Load(configPath)
	if err != nil {
		return nil, "", fmt.Errorf("loading config %s: %w", configPath, err)
	}

	providers, err := config.NormalizeProviders(&cfg.AI)
	if err != nil {
		return nil, "", fmt.Errorf("normalizing providers: %w", err)
	}

	var p config.ProviderConfig
	found := false
	for _, pp := range providers {
		if pp.Name == providerName {
			p = pp
			found = true
			break
		}
	}
	if !found {
		names := make([]string, len(providers))
		for i, pp := range providers {
			names[i] = pp.Name
		}
		return nil, "", fmt.Errorf("provider %q not found in config; available: %s", providerName, strings.Join(names, ", "))
	}

	if modelOverride != "" {
		p.Model = modelOverride
	}

	label := string(p.Type) + "/" + p.Model
	if p.Type == "" {
		label = "openai/" + p.Model
	}

	switch p.Type {
	case config.ProviderTypeOpenAI, "":
		c := ai.NewWithTransport(p.Endpoint, p.APIKey, p.Model, ai.NewReasoningAliasTransport(nil))
		return c, label, nil
	case config.ProviderTypeCopilot:
		tok, err := copilot.LoadGitHubToken()
		if err != nil {
			return nil, "", fmt.Errorf("loading Copilot token: %w", err)
		}
		if tok == nil {
			return nil, "", fmt.Errorf("Copilot not authenticated — run `werkler auth copilot` first")
		}
		transport := ai.NewReasoningAliasTransport(copilot.NewTransport(tok.AccessToken))
		c := ai.NewWithHTTPClient(copilot.CopilotAPIBaseURL, p.Model, &http.Client{Transport: transport}, ai.WithNoStreamUsage())
		return c, label, nil
	default:
		return nil, "", fmt.Errorf("unsupported provider type %q", p.Type)
	}
}

func runEval(ctx context.Context, baseURL, model, apiKey, provider, configPath string, run []string, repeat int, verbose bool) error {
	var client ai.Completer
	var label string

	switch {
	case provider != "":
		var err error
		client, label, err = buildClientFromConfig(provider, configPath, model)
		if err != nil {
			return err
		}
	case baseURL != "":
		if model == "" {
			model = "gpt-4o"
		}
		if apiKey == "" {
			apiKey = "unused"
		}
		client = ai.New(baseURL, apiKey, model)
		label = model + " @ " + baseURL
	default:
		return fmt.Errorf("specify either --provider <name> or --base-url <url>")
	}

	cases := modeleval.AllCases()
	scenarios := modeleval.AllScenarioCases()

	// Filter by --run flags (applies to both suites).
	if len(run) > 0 {
		filteredCases := cases[:0]
		for _, tc := range cases {
			if slices.Contains(run, tc.Name) {
				filteredCases = append(filteredCases, tc)
			}
		}
		filteredScenarios := scenarios[:0]
		for _, sc := range scenarios {
			if slices.Contains(run, sc.Name) {
				filteredScenarios = append(filteredScenarios, sc)
			}
		}
		if len(filteredCases) == 0 && len(filteredScenarios) == 0 {
			var available []string
			for _, tc := range cases {
				available = append(available, tc.Name)
			}
			for _, sc := range scenarios {
				available = append(available, sc.Name)
			}
			return fmt.Errorf("no cases matched %v; available: %s", run, strings.Join(available, ", "))
		}
		cases = filteredCases
		scenarios = filteredScenarios
	}

	// Apply --repeat override.
	if repeat > 0 {
		for _, tc := range cases {
			tc.Repeat = repeat
		}
		for _, sc := range scenarios {
			sc.Repeat = repeat
		}
	}

	fmt.Printf("Testing %q\n\n", label)

	anyFailed := false

	// ── single-turn tests ────────────────────────────────────────────────────
	if len(cases) > 0 {
		fmt.Println("=== Single-turn tests ===")
		results := modeleval.RunAll(ctx, client, cases)
		printResults(results, verbose)
		fmt.Println()
		for _, r := range results {
			if r.PassCount < len(r.Runs) {
				anyFailed = true
			}
		}
	}

	// ── multi-turn scenario tests ─────────────────────────────────────────────
	if len(scenarios) > 0 {
		fmt.Println("=== Multi-turn scenario tests ===")
		scenarioResults := modeleval.RunAllScenarios(ctx, client, scenarios)
		printScenarioResults(scenarioResults, verbose)
		for _, r := range scenarioResults {
			if r.PassCount < len(r.Runs) {
				anyFailed = true
			}
		}
	}

	if anyFailed {
		os.Exit(1)
	}
	return nil
}

// printResults renders the single-turn TestCase results table.
func printResults(results []*modeleval.Result, verbose bool) {
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	_, _ = fmt.Fprintf(tw, "CASE\tRESULT\tPASS RATE\tDESCRIPTION\n")
	_, _ = fmt.Fprintf(tw, "----\t------\t---------\t-----------\n")

	var totalPass, totalFail int
	for _, r := range results {
		pass := r.PassCount
		runs := len(r.Runs)
		fail := runs - pass

		status := "✓ PASS"
		if pass < runs {
			status = "✗ FAIL"
		}

		pct := passRateStr(pass, runs)
		_, _ = fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", r.Case.Name, status, pct, r.Case.Description)

		totalPass += pass
		totalFail += fail

		if verbose {
			for i, rr := range r.Runs {
				prefix := "  run"
				if runs > 1 {
					prefix = fmt.Sprintf("  run %d", i+1)
				}
				if rr.Passed {
					_, _ = fmt.Fprintf(tw, "%s\t✓\t%s\tcontent=%q tool_calls=%d\n",
						prefix, rr.Elapsed.Round(1e6), rr.Response.Content, len(rr.Response.ToolCalls))
				} else {
					content := rr.Response.Content
					if len(content) > 80 {
						content = content[:80] + "…"
					}
					_, _ = fmt.Fprintf(tw, "%s\t✗\t%s\t%v (content=%q tool_calls=%d)\n",
						prefix, rr.Elapsed.Round(1e6), rr.Err, content, len(rr.Response.ToolCalls))
				}
			}
		}
	}
	_ = tw.Flush()

	total := totalPass + totalFail
	if totalFail == 0 {
		fmt.Printf("All %d run(s) passed.\n", total)
	} else {
		fmt.Printf("%d/%d run(s) passed, %d failed.\n", totalPass, total, totalFail)
	}
}

// printScenarioResults renders the multi-turn ScenarioCase results table and,
// in verbose mode, the full tool call trace for each run.
func printScenarioResults(results []*modeleval.ScenarioResult, verbose bool) {
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	_, _ = fmt.Fprintf(tw, "CASE\tRESULT\tPASS RATE\tDESCRIPTION\n")
	_, _ = fmt.Fprintf(tw, "----\t------\t---------\t-----------\n")

	var totalPass, totalFail int
	for _, r := range results {
		pass := r.PassCount
		runs := len(r.Runs)
		fail := runs - pass

		status := "✓ PASS"
		if pass < runs {
			status = "✗ FAIL"
		}

		pct := passRateStr(pass, runs)
		_, _ = fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", r.Case.Name, status, pct, r.Case.Description)

		totalPass += pass
		totalFail += fail

		if verbose {
			for i, rr := range r.Runs {
				runLabel := "  run"
				if runs > 1 {
					runLabel = fmt.Sprintf("  run %d", i+1)
				}
				outcome := "✓"
				if !rr.Passed {
					outcome = "✗"
				}
				_, _ = fmt.Fprintf(tw, "%s\t%s\t%d turns\t%v\n", runLabel, outcome, len(rr.Turns), rr.Err)
				for j, t := range rr.Turns {
					for _, tr := range t.ToolResults {
						argsJSON, _ := json.Marshal(tr.Args)
						_, _ = fmt.Fprintf(tw, "    turn %d\t→\t%s(%s)\t= %s\n",
							j+1, tr.Name, truncate(string(argsJSON), 60), truncate(tr.Output, 80))
					}
					if len(t.ToolResults) == 0 && t.ModelResponse.Content != "" {
						_, _ = fmt.Fprintf(tw, "    turn %d\t→\t[answer]\t%s\n",
							j+1, truncate(t.ModelResponse.Content, 120))
					}
				}
			}
		}
	}
	_ = tw.Flush()

	total := totalPass + totalFail
	if totalFail == 0 {
		fmt.Printf("All %d run(s) passed.\n", total)
	} else {
		fmt.Printf("%d/%d run(s) passed, %d failed.\n", totalPass, total, totalFail)
	}
}

func passRateStr(pass, runs int) string {
	if runs > 1 {
		return fmt.Sprintf("%.0f%% (%d/%d)", float64(pass)/float64(runs)*100, pass, runs)
	}
	if pass == 1 {
		return "passed"
	}
	return "failed"
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

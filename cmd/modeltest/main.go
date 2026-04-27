// modeltest evaluates AI model compatibility with werkler's tool definitions
// and system prompt by running a scripted test suite against any
// OpenAI-compatible endpoint.
//
// Usage:
//
//	modeltest --base-url https://... --model my-model [--api-key sk-...] [--run case-name] [--repeat N]
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"slices"
	"strings"
	"syscall"
	"text/tabwriter"

	"github.com/icedream/werkler/internal/ai"
	"github.com/icedream/werkler/internal/modeleval"
	"github.com/spf13/cobra"
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
		baseURL string
		model   string
		apiKey  string
		run     []string
		repeat  int
		verbose bool
	)

	cmd := &cobra.Command{
		Use:   "modeltest",
		Short: "Evaluate a model's tool-calling behaviour against werkler's prompts",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runEval(cmd.Context(), baseURL, model, apiKey, run, repeat, verbose)
		},
	}

	cmd.Flags().StringVarP(&baseURL, "base-url", "u", "", "OpenAI-compatible API base URL (e.g. http://localhost:11434/v1)")
	cmd.Flags().StringVarP(&model, "model", "m", "gpt-4o", "Model name to test")
	cmd.Flags().StringVarP(&apiKey, "api-key", "k", envOr("OPENAI_API_KEY", "unused"), "API key (defaults to $OPENAI_API_KEY)")
	cmd.Flags().StringArrayVarP(&run, "run", "r", nil, "Run only the named test case(s); repeatable. Default: all.")
	cmd.Flags().IntVarP(&repeat, "repeat", "n", 0, "Override Repeat count for every test case")
	cmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Print each run's response and error details")
	_ = cmd.MarkFlagRequired("base-url")

	return cmd
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func runEval(ctx context.Context, baseURL, model, apiKey string, run []string, repeat int, verbose bool) error {
	client := ai.New(baseURL, apiKey, model)

	cases := modeleval.AllCases()

	// Filter by --run flags.
	if len(run) > 0 {
		filtered := cases[:0]
		for _, tc := range cases {
			if slices.Contains(run, tc.Name) {
				filtered = append(filtered, tc)
			}
		}
		if len(filtered) == 0 {
			available := make([]string, len(cases))
			for i, tc := range cases {
				available[i] = tc.Name
			}
			return fmt.Errorf("no cases matched %v; available: %s", run, strings.Join(available, ", "))
		}
		cases = filtered
	}

	// Apply --repeat override.
	if repeat > 0 {
		for _, tc := range cases {
			tc.Repeat = repeat
		}
	}

	fmt.Printf("Testing model %q against %s\n\n", model, baseURL)

	results := modeleval.RunAll(ctx, client, cases)

	printResults(results, verbose)
	return nil
}

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

		pct := ""
		if runs > 1 {
			pct = fmt.Sprintf("%.0f%% (%d/%d)", r.PassRate()*100, pass, runs)
		} else {
			if pass == 1 {
				pct = "passed"
			} else {
				pct = "failed"
			}
		}

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

	fmt.Printf("\n")

	total := totalPass + totalFail
	if totalFail == 0 {
		fmt.Printf("All %d run(s) passed.\n", total)
	} else {
		fmt.Printf("%d/%d run(s) passed, %d failed.\n", totalPass, total, totalFail)
	}
}

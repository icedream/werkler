package main

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/icedream/werkler/internal/sessionstore"
)

var sessionsCmd = &cobra.Command{
	Use:   "sessions",
	Short: "Manage saved chat sessions",
}

var sessionsListCmd = &cobra.Command{
	Use:   "list",
	Short: "List saved chat sessions",
	RunE:  runSessionsList,
}

var sessionsDeleteCmd = &cobra.Command{
	Use:   "delete <id>",
	Short: "Delete a saved chat session by ID or unique prefix",
	Args:  cobra.ExactArgs(1),
	RunE:  runSessionsDelete,
}

var sessionsDeleteForce bool

func init() {
	sessionsDeleteCmd.Flags().BoolVarP(&sessionsDeleteForce, "force", "f", false, "Skip confirmation prompt")
	sessionsCmd.AddCommand(sessionsListCmd, sessionsDeleteCmd)
	rootCmd.AddCommand(sessionsCmd)
}

func runSessionsList(_ *cobra.Command, _ []string) error {
	store := sessionstore.New(sessionstore.DefaultDir())
	sessions, err := store.List()
	if err != nil {
		return fmt.Errorf("listing sessions: %w", err)
	}
	if len(sessions) == 0 {
		fmt.Println("No saved sessions.")
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	_, _ = fmt.Fprintln(w, "ID\tTITLE\tUPDATED\tCWD")
	_, _ = fmt.Fprintln(w, "──────────────────────\t────────────────────────────────────────────────────────────\t─────────────\t─────────────────────────────────")
	for _, sess := range sessions {
		title := sess.Title
		if len([]rune(title)) > 58 {
			title = string([]rune(title)[:55]) + "…"
		}
		updated := formatAge(sess.UpdatedAt)
		cwd := shortenPath(sess.CWD)
		_, _ = fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", sess.ID, title, updated, cwd)
	}
	return w.Flush()
}

func runSessionsDelete(_ *cobra.Command, args []string) error {
	store := sessionstore.New(sessionstore.DefaultDir())
	sess, err := store.LoadByPrefix(args[0])
	if err != nil {
		return fmt.Errorf("finding session: %w", err)
	}

	if !sessionsDeleteForce {
		fmt.Printf("Delete session %q (%s)? [y/N] ", sess.Title, sess.ID)
		var answer string
		fmt.Scanln(&answer) //nolint:errcheck
		if !strings.EqualFold(strings.TrimSpace(answer), "y") {
			fmt.Println("Cancelled.")
			return nil
		}
	}

	if err := store.Delete(sess.ID); err != nil {
		return fmt.Errorf("deleting session: %w", err)
	}
	fmt.Printf("Deleted session %q (%s)\n", sess.Title, sess.ID)
	return nil
}

// formatAge returns a human-friendly relative time string.
func formatAge(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	case d < 48*time.Hour:
		return "yesterday"
	case d < 7*24*time.Hour:
		return fmt.Sprintf("%d days ago", int(d.Hours()/24))
	default:
		return t.Format("2006-01-02")
	}
}

// shortenPath replaces the home directory prefix with ~.
func shortenPath(p string) string {
	home, err := os.UserHomeDir()
	if err != nil {
		return p
	}
	if strings.HasPrefix(p, home+"/") || strings.HasPrefix(p, home+"\\") {
		return "~" + p[len(home):]
	}
	if p == home {
		return "~"
	}
	return p
}

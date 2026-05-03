package tools

import (
	"bytes"
	"fmt"
	"strings"
)

// diffLineCap is the maximum number of output lines before truncation.
const diffLineCap = 150

// computeUnifiedDiff returns a unified diff string (--- a/path, +++ b/path,
// @@ hunks) for the given old→new content transition.
// Returns "" when old==new, either file is binary, or both are empty.
func computeUnifiedDiff(oldContent, newContent, path string) string {
	if oldContent == newContent {
		return ""
	}
	// Treat files with null bytes as binary.
	if bytes.ContainsRune([]byte(oldContent), 0) || bytes.ContainsRune([]byte(newContent), 0) {
		return ""
	}

	oldLines := splitLines(oldContent)
	newLines := splitLines(newContent)

	// Myers LCS edit script.
	edits := myersDiff(oldLines, newLines)

	// Build hunks (groups of changes with ±3 context lines).
	hunks := buildHunks(edits, oldLines, newLines, 3)
	if len(hunks) == 0 {
		return ""
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "--- a/%s\n", path)
	fmt.Fprintf(&sb, "+++ b/%s\n", path)

	totalLines := 0
	truncated := 0
	for _, h := range hunks {
		if totalLines >= diffLineCap {
			truncated += len(h.lines)
			continue
		}
		sb.WriteString(h.header)
		sb.WriteString("\n")
		for _, l := range h.lines {
			if totalLines >= diffLineCap {
				truncated++
				continue
			}
			sb.WriteString(l)
			sb.WriteString("\n")
			totalLines++
		}
	}
	if truncated > 0 {
		fmt.Fprintf(&sb, "\\ ...%d more line%s\n", truncated, pluralS(truncated))
	}
	return strings.TrimRight(sb.String(), "\n")
}

func pluralS(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// splitLines splits content into lines (without trailing newlines).
// A trailing newline in the file produces no extra empty element.
func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	lines := strings.Split(s, "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

// edit represents a single diff operation.
type edit struct {
	kind    editKind
	oldLine int // 0-based index into old
	newLine int // 0-based index into new
}

type editKind int8

const (
	editKeep editKind = iota
	editDel
	editIns
)

// myersDiff implements Myers' O(ND) diff algorithm and returns an edit script.
func myersDiff(a, b []string) []edit {
	n, m := len(a), len(b)
	if n == 0 && m == 0 {
		return nil
	}

	max := n + m
	// v maps diagonal k → furthest-reaching x on that diagonal.
	v := make([]int, 2*max+1)
	// trace stores the v array after each step for backtracking.
	trace := [][]int{}

	for d := 0; d <= max; d++ {
		snap := make([]int, len(v))
		copy(snap, v)
		trace = append(trace, snap)

		for k := -d; k <= d; k += 2 {
			var x int
			idx := k + max
			if k == -d || (k != d && v[idx-1] < v[idx+1]) {
				x = v[idx+1]
			} else {
				x = v[idx-1] + 1
			}
			y := x - k
			for x < n && y < m && a[x] == b[y] {
				x++
				y++
			}
			v[idx] = x
			if x >= n && y >= m {
				return backtrack(trace, a, b, max)
			}
		}
	}
	return backtrack(trace, a, b, max)
}

func backtrack(trace [][]int, a, b []string, offset int) []edit {
	x, y := len(a), len(b)
	var edits []edit
	for d := len(trace) - 1; d >= 0 && (x > 0 || y > 0); d-- {
		v := trace[d]
		k := x - y
		idx := k + offset
		var prevK int
		if k == -d || (k != d && v[idx-1] < v[idx+1]) {
			prevK = k + 1
		} else {
			prevK = k - 1
		}
		prevX := v[prevK+offset]
		prevY := prevX - prevK

		// Diagonal moves (keeps).
		for x > prevX && y > prevY {
			x--
			y--
			edits = append(edits, edit{editKeep, x, y})
		}
		if d > 0 {
			if x == prevX {
				// Insertion
				y--
				edits = append(edits, edit{editIns, x, y})
			} else {
				// Deletion
				x--
				edits = append(edits, edit{editDel, x, y})
			}
		}
	}
	// Reverse (we built it backwards).
	for i, j := 0, len(edits)-1; i < j; i, j = i+1, j-1 {
		edits[i], edits[j] = edits[j], edits[i]
	}
	return edits
}

type hunk struct {
	header string
	lines  []string
}

func buildHunks(edits []edit, a, b []string, ctx int) []hunk {
	if len(edits) == 0 {
		return nil
	}

	// Collect change positions in old and new.
	type span struct{ start, end int } // inclusive, in edit-script index
	var changeSpans []span
	inChange := false
	start := 0
	for i, e := range edits {
		if e.kind != editKeep {
			if !inChange {
				start = i
				inChange = true
			}
		} else if inChange {
			changeSpans = append(changeSpans, span{start, i - 1})
			inChange = false
		}
	}
	if inChange {
		changeSpans = append(changeSpans, span{start, len(edits) - 1})
	}
	if len(changeSpans) == 0 {
		return nil
	}

	// Merge overlapping context windows.
	type region struct{ start, end int } // edit-script indices
	var regions []region
	for _, cs := range changeSpans {
		lo := max(0, cs.start-ctx)
		hi := min(len(edits)-1, cs.end+ctx)
		if len(regions) > 0 && lo <= regions[len(regions)-1].end+1 {
			regions[len(regions)-1].end = hi
		} else {
			regions = append(regions, region{lo, hi})
		}
	}

	var hunks []hunk
	for _, r := range regions {
		var oldStart, oldCount, newStart, newCount int
		lines := []string{}

		for i := r.start; i <= r.end; i++ {
			e := edits[i]
			switch e.kind {
			case editKeep:
				if i == r.start {
					oldStart = e.oldLine + 1
					newStart = e.newLine + 1
				}
				lines = append(lines, " "+a[e.oldLine])
				oldCount++
				newCount++
			case editDel:
				if oldCount == 0 && newCount == 0 && i == r.start {
					oldStart = e.oldLine + 1
					// newStart is the insertion point for pure-deletion hunks.
					if e.newLine < len(b) {
						newStart = e.newLine + 1
					} else {
						newStart = len(b) + 1
					}
				} else if oldCount == 0 && newCount == 0 && oldStart == 0 {
					oldStart = e.oldLine + 1
				}
				lines = append(lines, "-"+a[e.oldLine])
				oldCount++
			case editIns:
				if oldCount == 0 && newCount == 0 && newStart == 0 {
					newStart = e.newLine + 1
				}
				lines = append(lines, "+"+b[e.newLine])
				newCount++
			}
		}
		if oldStart == 0 {
			oldStart = 1
		}
		if newStart == 0 {
			newStart = 1
		}
		header := fmt.Sprintf("@@ -%d,%d +%d,%d @@", oldStart, oldCount, newStart, newCount)
		hunks = append(hunks, hunk{header, lines})
	}
	return hunks
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// IntralineDiff returns a slice of (text, added bool) segments representing
// character-level differences between oldLine and newLine (the line content
// without the leading +/-). Short common prefix/suffix is preserved as context;
// the differing middle section is highlighted as a block.
//
// The algorithm is a simple longest-common-prefix/suffix approach: fast and
// good enough for single-line diffs without importing a full diff library.
func IntralineDiff(oldLine, newLine string) []IntralineSegment {
	// Find longest common prefix.
	prefixEnd := 0
	for prefixEnd < len(oldLine) && prefixEnd < len(newLine) && oldLine[prefixEnd] == newLine[prefixEnd] {
		prefixEnd++
	}
	// Find longest common suffix (after the prefix).
	oldSuffix := len(oldLine)
	newSuffix := len(newLine)
	for oldSuffix > prefixEnd && newSuffix > prefixEnd && oldLine[oldSuffix-1] == newLine[newSuffix-1] {
		oldSuffix--
		newSuffix--
	}

	prefix := oldLine[:prefixEnd]
	oldMid := oldLine[prefixEnd:oldSuffix]
	newMid := newLine[prefixEnd:newSuffix]
	suffix := oldLine[oldSuffix:]

	var segs []IntralineSegment
	if prefix != "" {
		segs = append(segs, IntralineSegment{Text: prefix})
	}
	if oldMid != "" {
		segs = append(segs, IntralineSegment{Text: oldMid, Removed: true})
	}
	if newMid != "" {
		segs = append(segs, IntralineSegment{Text: newMid, Added: true})
	}
	if suffix != "" {
		segs = append(segs, IntralineSegment{Text: suffix})
	}
	if len(segs) == 0 {
		segs = append(segs, IntralineSegment{Text: oldLine})
	}
	return segs
}

// IntralineSegment is one piece of an intra-line diff.
type IntralineSegment struct {
	Text    string
	Added   bool // part is in newLine but not oldLine
	Removed bool // part is in oldLine but not newLine
}

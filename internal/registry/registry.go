// Package registry fetches MCP server listings from the official
// Model Context Protocol registry at registry.modelcontextprotocol.io.
package registry

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

const baseURL = "https://registry.modelcontextprotocol.io/v0.1/servers"

// Remote describes a single connection endpoint for a remote MCP server.
type Remote struct {
	Type string `json:"type"` // "streamable-http"
	URL  string `json:"url"`
}

// Server is a single listing from the registry.
type Server struct {
	// Name is the registry-assigned dotted identifier (e.g. "io.github/owner/repo").
	Name string
	// Title is the human-readable display name.
	Title string
	// Description is the short description shown in the registry.
	Description string
	// Remotes holds streamable-http connection endpoints.
	Remotes []Remote
	// HasPackage is true when the server requires a local install (npm/pip/etc.)
	// rather than a remote URL. Such servers are not supported for direct add in v1.
	HasPackage bool
}

// Page is a single response page from the registry.
type Page struct {
	Servers    []Server
	NextCursor string
}

// Fetch returns a page of servers from the registry. Use an empty cursor for
// the first page. Pass a non-empty search string to filter results.
// latestOnly skips older versions of the same server name.
func Fetch(ctx context.Context, search, cursor string, limit int) (Page, error) {
	q := url.Values{}
	q.Set("limit", fmt.Sprintf("%d", limit))
	q.Set("latestOnly", "true")
	if search != "" {
		q.Set("search", search)
	}
	if cursor != "" {
		q.Set("cursor", cursor)
	}

	reqURL := baseURL + "?" + q.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return Page{}, fmt.Errorf("building request: %w", err)
	}
	req.Header.Set("User-Agent", "werkler/1")
	req.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return Page{}, fmt.Errorf("fetching registry: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return Page{}, fmt.Errorf("registry returned HTTP %d", resp.StatusCode)
	}

	var raw struct {
		Servers []struct {
			Server struct {
				Name        string   `json:"name"`
				Title       string   `json:"title"`
				Description string   `json:"description"`
				Remotes     []Remote `json:"remotes"`
				Packages    []any    `json:"packages"`
			} `json:"server"`
		} `json:"servers"`
		Metadata struct {
			NextCursor string `json:"nextCursor"`
		} `json:"metadata"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return Page{}, fmt.Errorf("decoding registry response: %w", err)
	}

	page := Page{NextCursor: raw.Metadata.NextCursor}
	for _, entry := range raw.Servers {
		s := entry.Server
		srv := Server{
			Name:        s.Name,
			Title:       s.Title,
			Description: s.Description,
			Remotes:     s.Remotes,
			HasPackage:  len(s.Packages) > 0 && len(s.Remotes) == 0,
		}
		if srv.Title == "" {
			srv.Title = s.Name
		}
		page.Servers = append(page.Servers, srv)
	}
	return page, nil
}

// FirstRemoteURL returns the URL of the first streamable-http remote, or "" if none.
func (s Server) FirstRemoteURL() string {
	for _, r := range s.Remotes {
		if r.Type == "streamable-http" && r.URL != "" {
			return r.URL
		}
	}
	return ""
}

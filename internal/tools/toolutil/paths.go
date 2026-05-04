package toolutil

import (
	"fmt"
	"strings"

	"github.com/icedream/werkler/internal/chat"
)

// PathAccessRequest is re-exported from internal/chat for use in subpackage
// Context interfaces without importing the full chat package separately.
type PathAccessRequest = chat.PathAccessRequest

// PathApprovalError is re-exported from internal/chat.
type PathApprovalError = chat.PathApprovalError

// UnapprovedPathsError is returned by path-check helpers when one or more
// file paths have not been approved for the requested access level.
// It implements chat.PathApprovalError so the TUI can intercept it via errors.As.
type UnapprovedPathsError struct {
	Requests []PathAccessRequest
}

func (e *UnapprovedPathsError) Error() string {
	paths := make([]string, len(e.Requests))
	for i, r := range e.Requests {
		paths[i] = r.Path
	}
	return fmt.Sprintf("path access not approved: %s", strings.Join(paths, ", "))
}

// AccessRequests implements chat.PathApprovalError.
func (e *UnapprovedPathsError) AccessRequests() []PathAccessRequest {
	return e.Requests
}

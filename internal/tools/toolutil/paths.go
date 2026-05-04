package toolutil

import "github.com/icedream/werkler/internal/chat"

// PathAccessRequest is re-exported from internal/chat for use in subpackage
// Context interfaces without importing the full chat package separately.
type PathAccessRequest = chat.PathAccessRequest
type PathApprovalError = chat.PathApprovalError

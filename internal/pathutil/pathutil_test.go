package pathutil_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/icedream/werkler/internal/pathutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExpandTilde(t *testing.T) {
	home, err := os.UserHomeDir()
	require.NoError(t, err)

	assert.Equal(t, home, pathutil.ExpandTilde("~"))
	assert.Equal(t, filepath.Join(home, ".agents", "skills"), pathutil.ExpandTilde("~/.agents/skills"))
	assert.Equal(t, "/absolute/path", pathutil.ExpandTilde("/absolute/path"))
	assert.Equal(t, "relative/path", pathutil.ExpandTilde("relative/path"))
}

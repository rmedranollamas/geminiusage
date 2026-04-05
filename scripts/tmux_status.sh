#!/bin/bash
# scripts/tmux_status.sh - Optimized token usage reporter for tmux status line
# Caches results to minimize Python overhead in high-frequency status bars.

# Default interval: 60 seconds
INTERVAL=${1:-60}
CACHE_FILE="${HOME}/.gemini/tmux_status.cache"
# Ensure the directory exists
mkdir -p "$(dirname "$CACHE_FILE")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/token_usage.py"

# Ensure the python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: token_usage.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

# Detect platform for stat command
if stat -c %Y "$PYTHON_SCRIPT" >/dev/null 2>&1; then
    STAT_CMD="stat -c %Y"
else
    STAT_CMD="stat -f %m"
fi

# Only run heavy logic if cache is old or doesn't exist
NEEDS_UPDATE=false
if [[ ! -f "$CACHE_FILE" ]]; then
    NEEDS_UPDATE=true
else
    LAST_UPDATE=$($STAT_CMD "$CACHE_FILE" 2>/dev/null || echo 0)
    CURRENT_TIME=$(date +%s)
    if (( CURRENT_TIME - LAST_UPDATE >= INTERVAL )); then
        NEEDS_UPDATE=true
    fi
fi

if [[ "$NEEDS_UPDATE" == "true" ]]; then
    # Run the python script and format to millions (e.g., 1.2M)
    # Redirect stderr to /dev/null to keep tmux status clean
    # We use a temporary file for atomic updates
    TEMP_FILE=$(mktemp "${CACHE_FILE}.XXXXXX")
    TOTAL_TOKENS=$(python3 "$PYTHON_SCRIPT" --today --raw 2>/dev/null)

    if [[ -n "$TOTAL_TOKENS" && "$TOTAL_TOKENS" =~ ^[0-9]+$ ]]; then
        # Use awk for floating point division and formatting
        echo "$TOTAL_TOKENS" | awk '{printf "%.1fM", $1/1000000}' > "$TEMP_FILE"
        mv "$TEMP_FILE" "$CACHE_FILE"
    else
        # If it fails, we keep the old cache if it exists, or write 0.0M
        if [[ ! -f "$CACHE_FILE" ]]; then
            echo "0.0M" > "$CACHE_FILE"
        fi
    fi
    # Clean up temp file if it still exists (e.g. if mv failed or token check failed)
    rm -f "$TEMP_FILE"
fi

if [[ -f "$CACHE_FILE" ]]; then
    cat "$CACHE_FILE"
else
    echo "0.0M"
fi

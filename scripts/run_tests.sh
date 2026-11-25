#!/bin/sh
log="${TEST_LOG:-$(mktemp)}"
uv run python -m pytest -p no:cacheprovider "$@" > "$log" 2>&1
status=$?
grep -E "^(FAILED|ERROR)|[0-9]+ (passed|failed|skipped)" "$log" | tail -40
echo "pytest exit status: $status (full log: $log)"
exit $status

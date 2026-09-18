#!/usr/bin/env bash
# Kill processes whose command line matches $1, without killing ourselves or our parent shell.
pat="$1"
n=0
for p in $(pgrep -f -- "$pat"); do
  [ "$p" = "$$" ] && continue
  [ "$p" = "$PPID" ] && continue
  kill "$p" 2>/dev/null && n=$((n+1))
done
echo "killed $n process(es) matching: $pat"

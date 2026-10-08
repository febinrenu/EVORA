#!/usr/bin/env bash
# Installs the local-only session files and git hooks into a KAAVAL clone.
# Usage: bash setup-local.sh [path/to/repo]     (default: current directory)
# Safe to run more than once.
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-.}"
cd "$REPO"
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository: $REPO" >&2; exit 1
fi
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 1) Local-only files
cp "$KIT_DIR/PLAN.md" ./PLAN.md
cp "$KIT_DIR/CLAUDE.md" ./CLAUDE.md
mkdir -p .claude
cp "$KIT_DIR/claude-settings.local.json" .claude/settings.local.json

# 2) Local ignore rules (in .git/info/exclude, which is never committed)
EXCL="$(git rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$EXCL")"; touch "$EXCL"
for p in /PLAN.md /CLAUDE.md /CLAUDE.local.md /.claude/ /.env /.env.local /data/ /models/ /workspaces/ /logs/; do
  grep -qxF "$p" "$EXCL" || echo "$p" >> "$EXCL"
done

# 3) Hooks
if git config --get core.hooksPath >/dev/null 2>&1; then
  echo "Note: core.hooksPath is set to '$(git config --get core.hooksPath)'; unsetting it for this repo so local hooks run."
  git config --unset core.hooksPath
fi
HOOKS="$(git rev-parse --git-path hooks)"
mkdir -p "$HOOKS"
cp "$KIT_DIR/hooks/commit-msg" "$HOOKS/commit-msg"
cp "$KIT_DIR/hooks/pre-commit" "$HOOKS/pre-commit"
chmod +x "$HOOKS/commit-msg" "$HOOKS/pre-commit"

# 4) Shared files: create only if missing (commit these once, from M1's machine)
[ -f PROGRESS.md ] || cp "$KIT_DIR/PROGRESS.md" ./PROGRESS.md
touch .gitattributes
while IFS= read -r line; do
  [ -z "$line" ] && continue
  grep -qxF "$line" .gitattributes || echo "$line" >> .gitattributes
done < "$KIT_DIR/gitattributes"

echo "Installed in $ROOT"
echo "  local only : PLAN.md, CLAUDE.md, .claude/settings.local.json, hooks, .git/info/exclude rules"
echo "  shared     : PROGRESS.md, .gitattributes  (commit these once if they are new)"
echo "Next: git config user.name / user.email, then start your session and run /status."

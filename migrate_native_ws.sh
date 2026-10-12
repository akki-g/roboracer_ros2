#!/usr/bin/env bash
# One-time move: copy the packages from the native workspace that already drives
# the car into this repo's src/, minus their .git folders, so your controller
# mapping and calibration (joy_teleop.yaml, vesc.yaml) get committed here.
# The native workspace is only read, never changed.
#   bash migrate_native_ws.sh [native_workspace]   (default: ~/f1tenth_ws)
set -euo pipefail

NATIVE="${1:-$HOME/f1tenth_ws}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$REPO_DIR/src"

if [ ! -d "$NATIVE/src" ]; then
  echo "No src/ folder in $NATIVE. Pass the native workspace path, e.g.: bash migrate_native_ws.sh /home/<user>/f1tenth_ws" >&2
  exit 1
fi
mkdir -p "$DEST"

copied=0
for dir in "$NATIVE"/src/*; do
  [ -d "$dir" ] || continue
  name=$(basename "$dir")
  if [ -e "$DEST/$name" ]; then
    echo "Skipping $name: src/$name already exists in the repo"
    continue
  fi
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='.git' --exclude='.gitmodules' "$dir/" "$DEST/$name/"
  else
    cp -a "$dir" "$DEST/$name"
    # Only touches the fresh copy inside this repo, never the native workspace
    find "$DEST/$name" \( -name .git -o -name .gitmodules \) -prune -exec rm -rf {} +
  fi
  echo "Copied $name"
  copied=$((copied + 1))
done

echo
echo "Packages now in src/:"
find "$DEST" -name package.xml -exec sed -n 's:.*<name>\(.*\)</name>.*:  \1:p' {} \; | sort
echo
if [ "$copied" -gt 0 ]; then
  echo "Next: commit them so the whole team has them:"
  echo "  git add src && git commit -m 'Add driver stack from the working native workspace' && git push"
  echo "Then: bash setup_docker.sh"
fi

#!/usr/bin/env bash
# Create ~/.kaggle/kaggle.json (Kaggle API login, needed only for the Chest X-ray download).
#   bash scripts/setup_kaggle.sh                       # asks for username and key
#   bash scripts/setup_kaggle.sh myname a1b2c3...      # or pass them directly
# Get the key on kaggle.com -> profile picture -> Settings -> API -> Create New Token.
set -euo pipefail
USER_NAME=${1:-}
KEY=${2:-}
[ -z "$USER_NAME" ] && read -rp "Kaggle username: " USER_NAME
[ -z "$KEY" ] && { read -rsp "Kaggle API key (hidden): " KEY; echo; }
[ -z "$USER_NAME" ] || [ -z "$KEY" ] && { echo "username and key are both required" >&2; exit 1; }

FILE="$HOME/.kaggle/kaggle.json"
mkdir -p "$HOME/.kaggle"
if [ -f "$FILE" ]; then
  read -rp "$FILE already exists. Overwrite? [y/N] " ans
  [[ "$ans" == [yY]* ]] || { echo "kept the existing file"; exit 0; }
fi
umask 077
printf '{"username":"%s","key":"%s"}\n' "$USER_NAME" "$KEY" > "$FILE"
chmod 600 "$FILE"
echo "wrote $FILE (readable only by you)"

# quick check if the kaggle CLI is available (it is inside the project's .venv)
REPO=$(cd "$(dirname "$0")/.." && pwd)
KAGGLE=$(command -v kaggle || echo "$REPO/.venv/bin/kaggle")
if [ -x "$KAGGLE" ]; then
  if "$KAGGLE" datasets list -s chest-xray >/dev/null 2>&1; then
    echo "login works"
  else
    echo "login test FAILED - check username/key (or no internet on this node)" >&2
  fi
fi

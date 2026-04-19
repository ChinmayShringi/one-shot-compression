#!/usr/bin/env bash
# Sync compress project to/from alon. Usage:
#   ./sync.sh push   — local → remote (default)
#   ./sync.sh pull   — remote → local
#   ./sync.sh both   — push then pull results

HOST="chinmay@172.24.113.178"
REMOTE="~/compress/"
LOCAL="$(cd "$(dirname "$0")" && pwd)/"
EXCLUDE="--exclude=vid.mp4 --exclude=__pycache__ --exclude=venv --exclude=.ruff_cache --exclude=nexus_frames"

push() {
    echo "==> Pushing local → alon..."
    sshpass -p 'Bon*Chon!White#Rice$' rsync -avz $EXCLUDE \
        -e "ssh -o StrictHostKeyChecking=no" \
        "$LOCAL" "$HOST:$REMOTE"
    echo "==> Push complete."
}

pull() {
    echo "==> Pulling alon → local..."
    sshpass -p 'Bon*Chon!White#Rice$' rsync -avz $EXCLUDE \
        -e "ssh -o StrictHostKeyChecking=no" \
        "$HOST:$REMOTE" "$LOCAL"
    echo "==> Pull complete."
}

case "${1:-push}" in
    push) push ;;
    pull) pull ;;
    both) push && pull ;;
    *) echo "Usage: $0 {push|pull|both}" ;;
esac

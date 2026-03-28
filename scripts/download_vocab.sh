#!/usr/bin/env bash
# Download the ORB vocabulary file required by ORB-SLAM
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VOCAB_DIR="$SCRIPT_DIR/../vocab"
VOCAB_FILE="$VOCAB_DIR/ORBvoc.txt"

if [ -f "$VOCAB_FILE" ]; then
  echo "ORBvoc.txt already exists at $VOCAB_FILE"
  exit 0
fi

mkdir -p "$VOCAB_DIR"

VOCAB_URL="https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz"
TMP_FILE="$(mktemp /tmp/orbvoc.XXXXXXXXXX)"

echo "Downloading ORB vocabulary..."
curl -fSL "$VOCAB_URL" -o "$TMP_FILE"

echo "Extracting..."
tar -xzf "$TMP_FILE" -C "$VOCAB_DIR"
rm -f "$TMP_FILE"

echo "Done: $VOCAB_FILE"

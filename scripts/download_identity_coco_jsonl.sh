#!/usr/bin/env bash
# Download the Karpathy identity_conversations JSONL into COCO_data and report
# Usage: ./download_identity_coco_jsonl.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COCO_DIR="$REPO_ROOT/COCO_data"
mkdir -p "$COCO_DIR"

TRAIN_URL="https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"

echo "Downloading identity JSONL to $COCO_DIR"
curl -L -o "$COCO_DIR/coco_train.jsonl" "$TRAIN_URL"
curl -L -o "$COCO_DIR/coco_val.jsonl" "$TRAIN_URL"

echo
ls -l "$COCO_DIR"/coco_*.jsonl || true
echo
echo "Line counts:" 
wc -l "$COCO_DIR/coco_train.jsonl" "$COCO_DIR/coco_val.jsonl" || true

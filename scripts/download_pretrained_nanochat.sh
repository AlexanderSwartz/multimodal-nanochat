#!/bin/bash

CACHE_DIR="$HOME/.cache/nanochat"
echo "--- Saving pre-trained weights and tokenizer to $CACHE_DIR---"

echo "--- Downloading Nanochat Checkpoints ---"
hf download sdobson/nanochat model_000650.pt \
    --local-dir "$CACHE_DIR/base_checkpoints/d20"

hf download sdobson/nanochat meta_000650.json \
    --local-dir "$CACHE_DIR/base_checkpoints/d20"

echo "--- Downloading Tokenizer Files ---"
hf download sdobson/nanochat tokenizer.pkl \
    --local-dir "$CACHE_DIR/tokenizer"

hf download sdobson/nanochat token_bytes.pt \
    --local-dir "$CACHE_DIR/tokenizer"

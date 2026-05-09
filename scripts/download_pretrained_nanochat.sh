#!/bin/bash

CACHE_DIR="$HOME/.cache/nanochat"
echo "--- Saving pre-trained weights and tokenizer to $CACHE_DIR---"

echo "--- Installing Hugging Face Hub ---"
uv pip install huggingface_hub

echo "--- Downloading Nanochat Checkpoints ---"
huggingface-cli download sdobson/nanochat model_000650.pt \
    --local-dir "$CACHE_DIR/chatsft_checkpoints/d20"

huggingface-cli download sdobson/nanochat meta_000650.json \
    --local-dir "$CACHE_DIR/chatsft_checkpoints/d20"

echo "--- Downloading Tokenizer Files ---"
huggingface-cli download sdobson/nanochat tokenizer.pkl \
    --local-dir "$CACHE_DIR/tokenizer"

huggingface-cli download sdobson/nanochat token_bytes.pt \
    --local-dir "$CACHE_DIR/tokenizer"

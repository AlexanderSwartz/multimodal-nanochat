from PIL import Image
import os
import torch
import logging
from transformers import CLIPProcessor, CLIPVisionModel

# Reduce noisy INFO logs from httpx/transformers when loading models in workers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Cache loaded CLIP processor + vision model per process to avoid repeated
# downloads and repeated model instantiation when `compute_image_embedding_for_id`
# is called frequently (e.g., in DataLoader workers).
_CLIP_CACHE = {}


def _find_image_path(image_id, repo_root, split):
    # Try several common COCO filename patterns
    img_dir = os.path.join(repo_root, 'COCO_data', f"{split}2017")
    candidates = []
    try:
        iid = int(image_id)
        candidates.append(os.path.join(img_dir, f"{iid:012d}.jpg"))
        candidates.append(os.path.join(img_dir, f"{iid:06d}.jpg"))
        candidates.append(os.path.join(img_dir, f"{iid}.jpg"))
    except Exception:
        candidates.append(os.path.join(img_dir, image_id))
        candidates.append(os.path.join(img_dir, image_id + '.jpg'))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def compute_image_embedding_for_id(image_id, repo_root, split="val", num_tokens=None, target_dim=768, clip_model_name="openai/clip-vit-base-patch32"):
    """
    Compute a CLIP image embedding for `image_id` using Hugging Face transformers.

    By default this returns the full CLIP `last_hidden_state` tensor of shape
    `(num_patches+1, dim)` on CPU (matching how `CLIP_loader.py` constructs
    embeddings). If `num_tokens` is provided (int), the function will instead
    return a repeated pooled vector of shape `(num_tokens, target_dim)` to
    preserve compatibility with callers that expect a fixed number of visual
    tokens.

    The function does not save files to disk — it returns the tensor on CPU.
    """
    img_path = _find_image_path(image_id, repo_root, split)
    if img_path is None:
        raise FileNotFoundError(f"Image file for id {image_id} not found under {repo_root}/COCO_data/{split}2017")

    cache_key = clip_model_name
    if cache_key in _CLIP_CACHE:
        processor, vision_model = _CLIP_CACHE[cache_key]
    else:
        processor = CLIPProcessor.from_pretrained(clip_model_name)
        vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        vision_model.eval()
        _CLIP_CACHE[cache_key] = (processor, vision_model)

    with torch.no_grad():
        img = Image.open(img_path).convert('RGB')
        inputs = processor(images=img, return_tensors='pt')
        # Keep everything on CPU for worker-safe, CPU-only embedding computation
        outputs = vision_model(**inputs)

        # Prefer returning full per-patch embeddings (last_hidden_state)
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            emb_seq = outputs.last_hidden_state.squeeze(0).cpu().to(dtype=torch.float32)
        else:
            # Fallback: use pooler_output or mean-pooled last hidden state
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                emb_seq = outputs.pooler_output.squeeze(0).unsqueeze(0).cpu().to(dtype=torch.float32)
            else:
                # This should rarely happen but keep robust
                raise RuntimeError("Unexpected CLIP output format; no last_hidden_state or pooler_output present")

    # If num_tokens provided, return a repeated pooled vector (compatibility mode)
    if num_tokens is not None:
        # get pooled vector if available, otherwise mean-pool the sequence
        with torch.no_grad():
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled = outputs.pooler_output.squeeze(0).cpu().to(dtype=torch.float32)
            else:
                pooled = emb_seq.mean(dim=0)
            D = pooled.numel()
            if D >= target_dim:
                feat = pooled[:target_dim]
            else:
                pad = torch.zeros((target_dim - D,), dtype=torch.float32)
                feat = torch.cat([pooled, pad], dim=0)
            out = feat.unsqueeze(0).expand(num_tokens, -1).contiguous()
            return out

    # Otherwise return the sequence of patch embeddings (may vary in length)
    # Ensure last dim matches target_dim by truncating/padding channels if needed
    seq_len, dim = emb_seq.shape
    if dim != target_dim:
        if dim > target_dim:
            emb_seq = emb_seq[:, :target_dim]
        else:
            pad = torch.zeros((seq_len, target_dim - dim), dtype=torch.float32)
            emb_seq = torch.cat([emb_seq, pad], dim=1)

    return emb_seq

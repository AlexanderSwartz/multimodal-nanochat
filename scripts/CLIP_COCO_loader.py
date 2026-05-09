#!/usr/bin/env python3
"""
Lightweight script to compute CLIP image embeddings for COCO images and save each
image's embeddings as a separate .pt file.

Usage for embedding small portion of training data (remove --max-samples to embed all training images):
    python scripts/CLIP_COCO_loader.py --images-dir COCO_data/train2017 \
        --ann-file COCO_data/annotations/captions_train2017.json \
        --save-dir COCO_data/embeddings_train

Usage for validation data:
    python scripts/CLIP_COCO_loader.py --images-dir COCO_data/val2017 \
        --ann-file COCO_data/annotations/captions_val2017.json \
        --save-dir COCO_data/embeddings_val
"""

import argparse
import os
import sys
from typing import List

from pycocotools.coco import COCO
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # fallback
        return x

from transformers import CLIPProcessor, CLIPVisionModel

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute and save CLIP image embeddings for COCO images."
    )
    p.add_argument("--images-dir", default="COCO_data/val2017", help="Path to COCO images directory (default: COCO_data/val2017)")
    p.add_argument("--ann-file", default="COCO_data/annotations/captions_val2017.json", help="Path to COCO annotations json (default: COCO_data/annotations/captions_val2017.json)")
    p.add_argument("--save-dir", default="COCO_data/embeddings_val", help="Directory where .pt files are written (default: COCO_data/embeddings_val)")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32", help="HuggingFace CLIP model")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu. Auto-detected if omitted")
    p.add_argument("--skip-existing", action="store_true", help="Skip images which already have saved embeddings")
    return p.parse_args()


class CocoImagePathsDataset(Dataset):
    def __init__(self, images_dir: str, ann_file: str):
        self.images_dir = images_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        return {"img_id": img_id, "img_path": img_path}


def collate_identity(batch: List[dict]) -> List[dict]:
    return batch


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if not os.path.exists(args.images_dir):
        print(f"images-dir does not exist: {args.images_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.ann_file):
        print(f"ann-file does not exist: {args.ann_file}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = CocoImagePathsDataset(args.images_dir, args.ann_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_identity,
    )

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    vision_model = CLIPVisionModel.from_pretrained(args.clip_model).to(device)
    vision_model.eval()

    total = len(dataloader)
    for batch in tqdm(dataloader, total=total):
        # batch is a list of {"img_id","img_path"}
        img_ids = [b["img_id"] for b in batch]
        img_paths = [b["img_path"] for b in batch]

        images = []
        save_paths = []
        for img_id, img_path in zip(img_ids, img_paths):
            save_path = os.path.join(args.save_dir, f"{img_id}.pt")
            if args.skip_existing and os.path.exists(save_path):
                save_paths.append(None)  # placeholder to keep indexing aligned
                continue
            if not os.path.exists(img_path):
                print(f"Warning: image not found, skipping: {img_path}", file=sys.stderr)
                save_paths.append(None)
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Warning: failed to open {img_path}: {e}", file=sys.stderr)
                save_paths.append(None)
                continue
            images.append(img)
            save_paths.append(save_path)

        # if no images to process in this batch, continue
        if not any(p is not None for p in save_paths):
            continue

        # Filter out None entries in save_paths and images to keep alignment
        proc_images = [img for img, sp in zip(images, save_paths) if sp is not None]
        proc_save_paths = [sp for sp in save_paths if sp is not None]

        inputs = processor(images=proc_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**inputs)
            image_embeddings = outputs.last_hidden_state  # shape [B, seq_len, dim]
            # save per-image
            for i, sp in enumerate(proc_save_paths):
                emb = image_embeddings[i].cpu()
                torch.save(emb, sp)

    print(f"Done — embeddings saved to {args.save_dir}")


if __name__ == "__main__":
    main()

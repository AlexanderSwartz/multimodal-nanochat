import argparse, json, os, sys
from collections import defaultdict

parser = argparse.ArgumentParser(description="Generate coco_<split>.jsonl from COCO captions")
parser.add_argument("--split", choices=("train", "val"), required=True)
parser.add_argument("--ann-dir", default="COCO_data/annotations")
parser.add_argument("--out-dir", default="COCO_data")
args = parser.parse_args()

ann_file = os.path.join(args.ann_dir, f"captions_{args.split}2017.json")
out_file = os.path.join(args.out_dir, f"coco_{args.split}.jsonl")

if not os.path.exists(ann_file):
    print(f"Annotation file not found: {ann_file}")
    sys.exit(1)

ann = json.load(open(ann_file, encoding="utf-8"))
caps = defaultdict(list)
for a in ann.get("annotations", []):
    img = a.get("image_id")
    if img is None:
        continue
    caps[img].append(a.get("caption", ""))

os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
with open(out_file, "w", encoding="utf-8") as out:
    for im in ann.get("images", []):
        i = im.get("id")
        if i is None:
            continue
        c = caps.get(i, [""])[0] if caps.get(i) else ""
        conv = {"image_id": i, "messages": [{"role": "user", "content": "Describe this image."}, {"role": "assistant", "content": c}]}
        out.write(json.dumps(conv, ensure_ascii=False) + "\n")

print(f"Wrote {out_file}")
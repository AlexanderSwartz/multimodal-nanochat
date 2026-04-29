import random
import argparse
import os
import json
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--infile", default="COCO_data/coco_val.jsonl")
parser.add_argument("--train_out", default="COCO_data/coco_val_train.jsonl")
parser.add_argument("--val_out", default="COCO_data/coco_val_val.jsonl")
parser.add_argument("--test_out", default="COCO_data/coco_test.jsonl")
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--val_frac", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--expand", action="store_true", help="Expand each image into multiple entries using COCO annotations")
parser.add_argument("--ann_file", default="COCO_data/annotations/captions_val2017.json", help="COCO annotations JSON file (used when --expand)")
args = parser.parse_args()

with open(args.infile, "r", encoding="utf-8") as f:
    raw_lines = [l.rstrip("\n") for l in f if l.strip()]

lines = []
if args.expand:
    # load COCO captions annotations and map image_id -> [captions]
    with open(args.ann_file, "r", encoding="utf-8") as af:
        ann_json = json.load(af)
    anns_by_img = {}
    for ann in ann_json.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        anns_by_img.setdefault(int(img_id), []).append(ann.get("caption", ""))

    for line in raw_lines:
        try:
            obj = json.loads(line)
        except Exception:
            # keep unknown lines as-is
            lines.append(line + "\n")
            continue

        img_id = obj.get("image_id")
        if img_id is None:
            # nothing to expand, keep original
            lines.append(line + "\n")
            continue

        # ensure int key
        try:
            key = int(img_id)
        except Exception:
            key = img_id

        caps = anns_by_img.get(key, [])
        if not caps:
            # fallback: keep original line
            lines.append(line + "\n")
            continue

        for cap in caps:
            new_obj = copy.deepcopy(obj)
            if "messages" in new_obj and isinstance(new_obj["messages"], list):
                found = False
                for m in new_obj["messages"]:
                    if m.get("role") == "assistant":
                        m["content"] = cap
                        found = True
                        break
                if not found:
                    new_obj["messages"].append({"role": "assistant", "content": cap})
            else:
                new_obj["messages"] = [{"role": "user", "content": "Describe this image."}, {"role": "assistant", "content": cap}]

            lines.append(json.dumps(new_obj, ensure_ascii=False) + "\n")
else:
    # no expansion requested; keep original lines with newline
    lines = [l + "\n" for l in raw_lines]

random.seed(args.seed)
random.shuffle(lines)

n = len(lines)
ntrain = int(n * args.train_frac)
nval = int(n * args.val_frac)
train = lines[:ntrain]
val = lines[ntrain:ntrain + nval]
test = lines[ntrain + nval:]

for path, block in [(args.train_out, train), (args.val_out, val), (args.test_out, test)]:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        out.writelines(block)

print(f"Split {n} -> train={len(train)} val={len(val)} test={len(test)}")

import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--infile", default="COCO_data/coco_val.jsonl")
parser.add_argument("--train_out", default="COCO_data/coco_val_train.jsonl")
parser.add_argument("--val_out", default="COCO_data/coco_val_val.jsonl")
parser.add_argument("--test_out", default="COCO_data/coco_test.jsonl")
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--val_frac", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

with open(args.infile, "r", encoding="utf-8") as f:
    lines = [l for l in f if l.strip()]

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

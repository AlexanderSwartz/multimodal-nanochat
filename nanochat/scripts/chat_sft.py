"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""

import gc
import argparse
import os
# Improve CUDA caching allocator configuration to reduce fragmentation.
# Prefer the CUDA-specific variable; keep the legacy name as a fallback.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import glob
try:
    import wandb
except Exception:
    wandb = None
import torch
import torch.profiler as torch_profiler
from torch.profiler import ProfilerActivity, tensorboard_trace_handler, _ExperimentalConfig
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanochat.vision import compute_image_embedding_for_id
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3
from torch.utils.data import DataLoader, Dataset

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

from sentence_transformers import SentenceTransformer, util
import contextlib

# load model to CPU to keep separate from GPU memory stats
semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
default_train_jsonl = os.path.join(repo_root, "COCO_data", "coco_train.jsonl")
default_val_jsonl = os.path.join(repo_root, "COCO_data", "coco_val_split.jsonl")
default_test_jsonl = os.path.join(repo_root, "COCO_data", "coco_test.jsonl")

# Discover train/val embeddings directories under COCO_data (embeddings_train, embeddings_val)
emb_base = os.path.join(repo_root, "COCO_data")
embeddings_train_dir = os.path.join(emb_base, "embeddings_train")
embeddings_val_dir = os.path.join(emb_base, "embeddings_val")


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
parser.add_argument("--train-vision-only", action='store_true', help="Freeze all model params except `vision_proj` and train only the vision projection layer")
parser.add_argument("--vision-lr", type=float, default=None, help="learning rate for `vision_proj` when using --train-vision-only (default: args.embedding_lr or 1e-3)")
parser.add_argument("--pin-memory", action='store_true', help="Use pinned memory for faster CPU to GPU Host-to-Device transfers")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes (default: inherit from pretrained checkpoint)
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain)")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit from pretrain)")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit from pretrain)")
# Online image embedding
parser.add_argument("--compute-embeddings", action='store_true', help="Compute image embeddings online instead of loading .pt files")
parser.add_argument("--embed-num-tokens", type=int, default=50, help="Number of image token rows to expand pooled embedding to (when computing on-the-fly)")
# `embed_target_dim` is fixed to 768; remove CLI override.
parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers for online embedding computation")
parser.add_argument("--persistent-workers", action='store_true', help="Use persistent DataLoader workers for online embedding computation")
# Optimization (default: inherit from pretrained checkpoint)
parser.add_argument("--embedding-lr", type=float, default=None, help="learning rate for embedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--unembedding-lr", type=float, default=None, help="learning rate for unembedding parameters (Adam) (default: inherit from pretrain)")
parser.add_argument("--matrix-lr", type=float, default=None, help="learning rate for matrix parameters (Muon) (default: inherit from pretrain)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--num-previews", type=int, default=10, help="number of preview generations to produce during each evaluation")
parser.add_argument("--preview-batch-size", type=int, default=8, help="chunk size for batched preview generation to limit GPU memory")
# Data mixture
parser.add_argument("--mmlu-epochs", type=int, default=3, help="number of epochs of MMLU in training mixture (teaches Multiple Choice)")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="number of epochs of GSM8K in training mixture (teaches Math and Tool Use)")
# COCO captioning data
parser.add_argument("--train-jsonl", type=str, default=default_train_jsonl, help="training JSONL file")
parser.add_argument("--val-jsonl", type=str, default=default_val_jsonl, help="validation JSONL file")
parser.add_argument("--test-jsonl", type=str, default=default_test_jsonl, help="test JSONL file")
# Debugging flags
parser.add_argument("--disable-image", action='store_true', help="Disable using image embeddings during training/eval (text-only debug)")
parser.add_argument("--kv-cache", action='store_true', help="Use KV cache (Engine) for preview generation (faster inference)")
parser.add_argument("--wandb-group", type=str, default=None, help="WandB group name to assign this run to")
parser.add_argument("--profile", action='store_true', help="Run with PyTorch profiler")
args = parser.parse_args()
user_config = vars(args).copy()
user_config["COMPUTE_DTYPE"] = str(COMPUTE_DTYPE)
# Allow disabling image handling via env var as well
# e.g. export NANOCHAT_DISABLE_IMAGE=1
disable_image = args.disable_image or os.environ.get("NANOCHAT_DISABLE_IMAGE", "0").lower() in ("1", "true", "yes")
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# Profiler setup (only now that `device` is available)
if args.profile:
    activities = [torch_profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch_profiler.ProfilerActivity.CUDA)
    schedule = torch_profiler.schedule(wait=1, warmup=20, active=5, repeat=1)
    prof_logdir = os.path.join(repo_root, "runs", args.run, "profiler")
    profiler = torch_profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch_profiler.tensorboard_trace_handler(prof_logdir),
        record_shapes=True,
        profile_memory=True,
    )
    profiler.start()

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="Multimodal-Nanochat", name=args.run, config=user_config, group=args.wandb_group)

# Flash Attention status
if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.")

# Load the model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),
    ("device_batch_size", 32,    meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

# Use a single eager model instance (do not call torch.compile)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer
# Optionally support training *only* the vision projection layer (freeze everything else)
if args.train_vision_only:
    print0("Freezing all parameters except vision_proj; training vision projection only")
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False
    # Ensure the model actually has a vision projection
    if not hasattr(model, 'vision_proj'):
        raise RuntimeError("Model has no 'vision_proj' attribute to train (cannot use --train-vision-only)")
    # Unfreeze the vision projection parameters
    for p in model.vision_proj.parameters():
        p.requires_grad = True
    # Build a small AdamW optimizer for the vision projection only
    import torch.optim as optim
    vis_lr = args.vision_lr if args.vision_lr is not None else (args.embedding_lr if args.embedding_lr is not None else 1e-3)
    optimizer = optim.AdamW(list(model.vision_proj.parameters()), lr=vis_lr, weight_decay=0.0)
    # Make the optimizer param groups compatible with the rest of the training loop
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
        group['kind'] = 'adamw'
else:
    # Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
    # Note that pretraining ramps weight_decay to zero by end of pretraining, so SFT continues with zero
    optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)

# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)
# Note: load_state_dict overwrites param_group metadata (LRs, betas, etc.) with the
# pretrained values. Since pretraining warmdown brings LRs to ~0, we must save and
# restore our fresh SFT LRs after loading.
base_dir = get_base_dir()
# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)
# Skip warm-start when we intentionally only train the vision projection layer because
# optimizer states from the original checkpoint won't match the small optimizer.
if args.load_optimizer and not args.train_vision_only:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
    else:
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")
elif args.load_optimizer and args.train_vision_only:
    print0("Skipping optimizer warm-start because --train-vision-only is set")

# GradScaler for fp16 training (bf16/fp32 don't need it)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# Override the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# SFT data: COCO-only captioning splits by default
print0(f"COCO train JSONL: {args.train_jsonl}")
print0(f"COCO val JSONL: {args.val_jsonl}")
train_dataset = CustomJSON(filepath=args.train_jsonl)
val_dataset = CustomJSON(filepath=args.val_jsonl)
print0(f"COCO dataset sizes: train={len(train_dataset):,} val={len(val_dataset):,}")
# When --compute-embeddings is set we compute embeddings on-the-fly (worker or inline).
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = False # we will toggle this to True when we reach the end of the training dataset
approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
current_epoch = 1 # track epoch for logging
def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()
    # If set, enforce that image-conditioned examples must contain a valid embedding file.
    # Useful for debugging dataset/embedding mismatches: set NANOCHAT_STRICT_IMAGE_EMBEDDINGS=1
    strict_image_embeddings = os.environ.get("NANOCHAT_STRICT_IMAGE_EMBEDDINGS", "0").lower() in ("1", "true", "yes")

    # Cache a fallback embedding shape once so refill_buffer does not rescan the
    # embeddings directory on every call. In the COCO-only setup this should
    # almost never be used, but it keeps the code robust for missing rows.
    # Determine a sensible default embedding by scanning the appropriate
    # embeddings directory for this split (train -> embeddings_train,
    # val -> embeddings_val). Do not fallback to other dirs.
    default_image_emb = None
    try:
        emb_dir_to_scan = embeddings_train_dir if split == "train" else embeddings_val_dir
        if os.path.isdir(emb_dir_to_scan):
            emb_files = glob.glob(os.path.join(emb_dir_to_scan, "*.pt"))
            for ef in emb_files:
                try:
                    t = torch.load(ef, map_location="cpu")
                except Exception:
                    continue
                if t.dim() == 3 and t.shape[0] == 1:
                    t = t.squeeze(0)
                if t.dim() == 2:
                    default_image_emb = t
                    break
    except Exception:
        default_image_emb = None
    if default_image_emb is None:
        default_image_emb = torch.zeros((50, 768), dtype=torch.float32)

    # Conversation buffer: list of (token_ids, loss_mask) tuples
    conv_buffer = []
    cursor = ddp_rank  # Each rank processes different conversations (for fetching)
    consumed = ddp_rank  # Track actual consumption separately from buffering
    epoch = 1
    it = 0  # iteration counter
    # Optionally compute embeddings inside DataLoader workers instead of main process
    compute_in_worker = args.compute_embeddings and args.num_workers > 0
    worker_iter = None
    if compute_in_worker:
        class WorkerConvDataset(Dataset):
            def __init__(self, base_dataset, repo_root, split, num_tokens, target_dim, default_emb):
                self.base = base_dataset
                self.repo_root = repo_root
                self.split = split
                self.num_tokens = num_tokens
                self.target_dim = target_dim
                self.default_emb = default_emb

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                conv = self.base[idx]
                img_id = conv.get('image_id')
                if img_id is None:
                    emb = self.default_emb.clone()
                else:
                    try:
                        emb = compute_image_embedding_for_id(
                            img_id,
                            self.repo_root,
                            split=self.split,
                            num_tokens=self.num_tokens,
                            target_dim=self.target_dim,
                        )
                    except Exception:
                        emb = self.default_emb.clone()
                new_conv = dict(conv)
                new_conv['_computed_image_emb'] = emb
                return new_conv

        worker_ds = WorkerConvDataset(dataset, repo_root, split, args.embed_num_tokens, 768, default_image_emb)
        worker_dl = DataLoader(
            worker_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda x: x[0],
            persistent_workers=args.persistent_workers,
        )
        worker_iter = iter(worker_dl)

    def refill_buffer():
        nonlocal cursor, epoch, worker_iter
        while len(conv_buffer) < buffer_size:
            if compute_in_worker:
                try:
                    batch = next(worker_iter)
                except StopIteration:
                    worker_iter = iter(worker_dl)
                    batch = next(worker_iter)
                conversation = batch
            else:
                conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            
            # --- NANOCHAT_V ---
            # Guard against empty/malformed dataset rows
            if not ids:
                cursor += ddp_world_size
                continue
                
            # Determine whether the current dataset row originates from an
            # image-conditioned task (e.g., CustomJSON). If so, strict image
            # embedding checks will be enforced when requested.
            if hasattr(dataset, 'index_map'):
                task_idx, _ = dataset.index_map[cursor]
                current_task = dataset.tasks[task_idx]
            else:
                current_task = dataset

            is_image_task = isinstance(current_task, CustomJSON)

            # --- ROBUST MULTIMODAL INJECTION ---
            img_id = conversation.get('image_id') # Ensure your dataset dict has this key

            # Load the tensor and ensure it's on CPU for the dataloader.
            # If strict mode is enabled, raise a helpful error when image_id is
            # missing or the embedding file cannot be found. Otherwise fall back
            # to a default zero embedding to allow mixing non-image tasks.
            if img_id is None:
                # Only error out for missing image_id if this row is from an
                # image task and strict mode is enabled. Otherwise fall back to
                # default embeddings so non-image tasks can be mixed.
                if strict_image_embeddings and is_image_task:
                    raise ValueError(
                        f"Conversation at dataset index {cursor} is missing 'image_id'.\n"
                        "This dataset row comes from an image task and strict image embedding\n"
                        "mode is enabled. Add an 'image_id' to this conversation or disable\n"
                        "strict mode by unsetting NANOCHAT_STRICT_IMAGE_EMBEDDINGS."
                    )
                image_emb = default_image_emb.clone()
            else:
                # Use the split-specific embeddings directory for this image id
                emb_dir_to_use = embeddings_train_dir if split == "train" else embeddings_val_dir
                emb_path = None
                if os.path.isdir(emb_dir_to_use):
                    candidate = os.path.join(emb_dir_to_use, f"{img_id}.pt")
                    if os.path.exists(candidate):
                        emb_path = candidate
                if args.compute_embeddings:
                    # Prefer worker-computed embedding if running with workers; otherwise compute on-the-fly
                    if compute_in_worker and conversation.get('_computed_image_emb') is not None:
                        image_emb = conversation.get('_computed_image_emb')
                    else:
                        try:
                            image_emb = compute_image_embedding_for_id(
                                img_id,
                                repo_root,
                                split=split,
                                num_tokens=args.embed_num_tokens,
                                target_dim=768,
                            )
                        except Exception as e:
                            if strict_image_embeddings and is_image_task:
                                raise
                            print0(f"Warning: failed to compute embedding for image_id {img_id}: {e}; using default embedding")
                            image_emb = default_image_emb.clone()
                else:
                    if emb_path is None:
                        if strict_image_embeddings and is_image_task:
                            raise FileNotFoundError(
                                f"Embedding file not found for image_id {img_id} in {emb_dir_to_use}\n"
                                "Run the CLIP_COCO_loader.ipynb to generate embeddings in that directory"
                            )
                        print0(f"Warning: embedding for image_id {img_id} not found in {emb_dir_to_use}; using default embedding")
                        image_emb = default_image_emb.clone()
                    else:
                        image_emb = torch.load(emb_path, map_location="cpu")
                        # Standardize shape to (seq_len, n_embd) in case the saved tensor has a batch dim of 1
                        if image_emb.dim() == 3 and image_emb.shape[0] == 1:
                            image_emb = image_emb.squeeze(0)

            # Ensure 2D shape (num_img_tokens, dim)
            if image_emb.dim() == 1:
                image_emb = image_emb.unsqueeze(0)
            num_img_tokens = image_emb.shape[0]

            # Pop the <|bos|> token off the front safely
            bos_id = ids.pop(0)
            bos_mask = mask.pop(0)

            # Rebuild the lists using bos_id as the safe, valid placeholder!
            ids = [bos_id] + ([bos_id] * num_img_tokens) + ids
            mask = [bos_mask] + ([0] * num_img_tokens) + mask
            # ------------------------------------------

            # Append 3 items now: ids, mask, and the actual image tensor
            conv_buffer.append((ids, mask, image_emb))
            
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
                # Note: last_step is now triggered based on consumption, not fetching
            approx_progress = (cursor + (epoch - 1) * dataset_size) / (args.num_iterations * args.total_batch_size)
            
    while True:
        rows = []
        mask_rows = []
        img_rows = []  # NEW: Array to hold exactly one image per batch row

        for _ in range(args.device_batch_size):
            # Ensure buffer has conversations
            while len(conv_buffer) < buffer_size:
                refill_buffer()

            # 1. Pop EXACTLY ONE conversation from the buffer
            # (This safely disables the multi-conversation packing algorithm)
            conv_ids, conv_mask, img_emb = conv_buffer.pop(0)

            # 2. Truncate if the caption + image somehow exceeds max sequence length
            if len(conv_ids) > row_capacity:
                conv_ids = conv_ids[:row_capacity]
                conv_mask = conv_mask[:row_capacity]

            row = list(conv_ids)
            mask_row = list(conv_mask)

            # 3. Pad the remainder of the row to match row_capacity
            remaining = row_capacity - len(row)
            if remaining > 0:
                row.extend([bos_token] * remaining)
                mask_row.extend([0] * remaining)  # Pad mask with 0 so it's ignored in loss

            rows.append(row)
            mask_rows.append(mask_row)
            img_rows.append(img_emb)  # Store the image tensor

        # --- TENSOR CREATION (Exact nanochat logic) ---
        # Create CPU tensors here (optionally pinned) and move to device later
        inputs = torch.tensor(rows, dtype=torch.long)
        targets = torch.tensor(rows, dtype=torch.long)
        mask_tensor = torch.tensor(mask_rows, dtype=torch.long)

        # Apply the shifted mask to targets
        mask_targets = mask_tensor[:, 1:]
        targets = targets[:, 1:]
        targets[mask_targets == 0] = -1

        # Shift inputs
        inputs = inputs[:, :-1]

        # Stack the batch of images into a single CPU float32 tensor (pin if requested).
        # GPU copy is done in the training loop to allow async host->device transfer.
        image_embeddings = torch.stack(img_rows).to(dtype=torch.float32).contiguous()

        if args.pin_memory:
            inputs = inputs.pin_memory()
            targets = targets.pin_memory()
            mask_tensor = mask_tensor.pin_memory()
            image_embeddings = image_embeddings.pin_memory()
            
        # Yield all three components to the training loop!
        yield inputs, targets, image_embeddings

train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0 # will go from 0 to 1 over the course of the epoch

# Learning rate schedule (linear warmup, constant, linear warmdown)
# Same shape as base_train but uses progress (0→1) instead of absolute step counts,
# because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
# x, y = next(train_loader) # prefetch the very first batch of data

#   
x, y, img_feats = next(train_loader)
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
warmup_time = 0.0 # sum of dt for the first 10 iterations (seconds)
step = 0
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(
            model,
            val_loader,
            eval_steps,
            token_bytes,
            disable_image=disable_image
        )
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        avg_sim_score = None
        if master_process:
            import random
            total_val_generation_time = 0.0

            # Generate N previews for DIFFERENT prompt+image pairs (controlled by CLI `--num-previews`)
            num_preview = args.num_previews
            semantic_scores = []

            # select candidate indices and collect valid samples
            val_indices = random.sample(range(len(val_dataset)), min(num_preview, len(val_dataset)))
            samples = []
            image_embs = []
            eval_conv = {"messages": [{"role": "user", "content": "Describe this image."}]}

            for idx in val_indices:
                preview_sample = val_dataset[idx]
                preview_image_id = preview_sample.get("image_id", "unknown")
                candidate = os.path.join(embeddings_val_dir, f"{preview_image_id}.pt")
                if not os.path.exists(candidate):
                    print0(f"[Preview] Embedding not found for image_id={preview_image_id} in {embeddings_val_dir}, skipping.")
                    continue
                emb = torch.load(candidate, map_location="cpu")
                if emb.dim() == 3 and emb.shape[0] == 1:
                    emb = emb.squeeze(0)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                # emb is (num_img_tokens, dim)
                image_embs.append(emb)
                samples.append(preview_sample)

            if len(samples) == 0:
                avg_val_generation_time = 0.0
            else:
                # Pad image embeddings to common num_img_tokens
                max_img_tokens = max(e.shape[0] for e in image_embs)
                dim = image_embs[0].shape[1]
                padded_embs = []
                for e in image_embs:
                    if e.shape[0] < max_img_tokens:
                        pad = torch.zeros((max_img_tokens - e.shape[0], dim), dtype=e.dtype)
                        e = torch.cat([e, pad], dim=0)
                    padded_embs.append(e)
                # Stack and move to device later (engine will also ensure device)
                batched_img_emb = torch.stack(padded_embs, dim=0).to(dtype=torch.float32)

                # Build a single prompt token list with placeholders matching max_img_tokens.
                bos_id = tokenizer.get_bos_token_id()
                prompt_ids, _ = tokenizer.render_conversation(eval_conv)
                b = prompt_ids.pop(0)
                prompt_template = [b] + ([b] * max_img_tokens) + prompt_ids

                # Run batched generation in smaller chunks to limit GPU memory
                engine = Engine(model, tokenizer)
                stop_token_id = tokenizer.encode_special("<|assistant_end|>")
                generated_ids_lists = [[] for _ in range(len(samples))]
                finished = [False] * len(samples)

                preview_batch = max(1, int(getattr(args, 'preview_batch_size', 8)))
                # iterate in chunks so we never prefill more than `preview_batch` rows at once
                for start in range(0, len(samples), preview_batch):
                    end = min(start + preview_batch, len(samples))
                    # reuse the same prompt template for each sub-batch
                    sub_prompts = [prompt_template] * (end - start)
                    sub_embs = batched_img_emb[start:end]  # CPU tensor slice
                    # Move this chunk of image embeddings to device for generation
                    if device.type == 'cuda':
                        sub_embs_device = sub_embs.to(device=device, dtype=torch.float32, non_blocking=args.pin_memory)
                    else:
                        sub_embs_device = sub_embs.to(device=device, dtype=torch.float32)

                    val_start_time = time.time()
                    # sub_prompts should all be the same: 50 tokens of image embedding placeholders followed by the "Describe the image" tokens
                    # sub_embs_device is a tensor of shape (sub_batch_size, max_img_tokens, dim) containing the image embeddings for this chunk
                    for token_column, token_masks in engine.generate(
                        sub_prompts,
                        num_samples=1,
                        max_tokens=128,
                        temperature=1.0,
                        top_k=None,
                        seed=42,
                        image_embeddings=sub_embs_device,
                        use_kv_cache=getattr(args, 'kv_cache', False),
                    ):
                        # token_column is a list of tokens, one per sub-sample
                        for local_i in range(len(sub_prompts)):
                            global_i = start + local_i
                            if finished[global_i]:
                                continue
                            token = token_column[local_i]
                            if token == stop_token_id or token == bos_id:
                                finished[global_i] = True
                                continue
                            generated_ids_lists[global_i].append(token)
                        if all(finished):
                            break
                    total_val_generation_time += time.time() - val_start_time
                    # free chunk device memory
                    try:
                        del sub_embs_device
                    except Exception:
                        pass
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                # Compute similarity and print a few
                for i, preview_sample in enumerate(samples):
                    gen_ids = generated_ids_lists[i]
                    # Filter out any special/control tokens before decoding
                    if len(gen_ids) > 0:
                        special_names = [
                            "<|assistant_start|>", "<|assistant_end|>",
                            "<|user_start|>", "<|user_end|>",
                            "<|output_start|>", "<|output_end|>",
                            "<|python_start|>", "<|python_end|>",
                            "<|bos|>",
                        ]
                        special_ids = set()
                        for nm in special_names:
                            try:
                                special_ids.add(tokenizer.encode_special(nm))
                            except Exception:
                                pass
                        filtered = [t for t in gen_ids if t not in special_ids]
                        generated_caption = tokenizer.decode(filtered).strip() if len(filtered) > 0 else ""
                    else:
                        generated_caption = ""
                    target_caption = ""
                    for msg in preview_sample.get("messages", []):
                        if msg.get("role") == "assistant":
                            target_caption = msg.get("content", "")
                            break
                    if target_caption:
                        emb_generated_caption = semantic_model.encode(generated_caption, convert_to_tensor=True, device='cpu', show_progress_bar=False)
                        emb_target_caption = semantic_model.encode(target_caption, convert_to_tensor=True, device='cpu', show_progress_bar=False)
                        sim_score = util.cos_sim(emb_generated_caption, emb_target_caption).item()
                        semantic_scores.append(sim_score)
                    if i < 5:
                        preview_image_id = samples[i].get("image_id", "unknown")
                        print("----------------------------------")
                        print0(f"Step {step:05d} | Validation image_id: ~as7629/multimodal-nanochat/COCO_data/val2017/000000{preview_image_id}.jpg")
                        print0(f"  Generated [{i}]: {generated_caption}")
                        print0(f"  Target:    {target_caption}")
                        if semantic_scores:
                            print0(f"  Sim Score: {semantic_scores[-1]:.4f}")

                avg_val_generation_time = total_val_generation_time / max(1, len(samples))
                
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        
        wandb_run.log({
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
            "val/avg_generation_time_seconds": avg_val_generation_time,
        }, step=step)
        
        if semantic_scores:
            avg_sim_score = sum(semantic_scores) / len(semantic_scores)
            print0(f"Average semantic similarity score across {num_preview} previews: {avg_sim_score:.4f}")
            wandb_run.log({
                "val/avg_semantic_similarity": avg_sim_score
            }, step=step)
        
        model.train()

    # save checkpoint at the end of the run (all ranks participate so each saves its optimizer shard)
    if last_step:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config, # inputs to the training script
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    dataloader_wait_ms = 0.0
    h2d_events = []

    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with torch_profiler.record_function("H2D Transfer"):
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            h2d_start.record()
            
            # manually move tensors so I can time the host->device transfer separately
            x = x.to(device=device, non_blocking=args.pin_memory)
            y = y.to(device=device, non_blocking=args.pin_memory)
            img_feats = img_feats.to(device=device, dtype=torch.float32, non_blocking=args.pin_memory)
            
            h2d_end.record()
            h2d_events.append((h2d_start, h2d_end))

        # loss = model(x, y)
        with torch_profiler.record_function("Forward Pass"):
            if disable_image:
                loss = model(x, y, loss_reduction='mean')
            else:
                # Use the eager model for image-conditioned forwards.
                loss = model(x, y, loss_reduction='mean', image_embeddings=img_feats)
                train_loss = loss.detach() # for logging
                loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        
        with torch_profiler.record_function("Backward Pass"):
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
        with torch_profiler.record_function("DataLoader Prefetch"):
            dataloader_wait_start = time.perf_counter()
            x, y, img_feats = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
            dataloader_wait_ms += (time.perf_counter() - dataloader_wait_start) * 1000
        
        progress = max(progress, approx_progress) # only increase progress monotonically
    h2d_transfer_ms = sum(start.elapsed_time(end) for start, end in h2d_events)
    # step the optimizer
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    
    with torch_profiler.record_function("Optimizer Step"):
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
        if scaler is not None:
            scaler.unscale_(optimizer)
            if is_ddp_initialized():
                for v in scaler._found_inf_per_device(optimizer).values():
                    dist.all_reduce(v, op=dist.ReduceOp.MAX)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # Honor --num-iterations to allow quick smoke tests
    if args.num_iterations > 0 and step >= args.num_iterations:
        last_step = True

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    total_training_time += dt
    if step <= 10:
        warmup_time += dt
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")
    print0(f"  dataloader wait: {dataloader_wait_ms:.2f}ms | h2d transfer: {h2d_transfer_ms:.2f}ms")
    dataloader_fraction = dataloader_wait_ms / (dt*1000) if dt>0 else 0
    wandb_run.log({
        "total_training_flops": flops_so_far,
        "total_training_time": total_training_time,
        "train/loss": debiased_smooth_loss,
        "train/dt": dt,
        "train/tok_per_sec": tok_per_sec,
        "train/mfu": mfu,
        "train/epoch": current_epoch,
        "train/dataloader_wait_ms": dataloader_wait_ms,
        "train/h2d_transfer_ms": h2d_transfer_ms,
        "train/dataloader_fraction": dataloader_fraction
    }, step=step)
    if args.profile:
        try:
            profiler.step()
        except Exception:
            pass

    # The garbage collector spends ~500ms scanning for cycles quite frequently.
    # We manually manage it to avoid these pauses during training.
    if step == 1:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # freeze all currently surviving objects and exclude them from GC
        gc.disable() # disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very long runs

# print a few more stats

# print a few more stats
peak_memory_mib = get_max_memory() / 1024 / 1024
print0(f"Peak memory usage: {peak_memory_mib:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log summary to wandb 
if wandb_run is not None:
    wandb_run.summary["peak_memory_mib"] = peak_memory_mib
    wandb_run.summary["total_training_time"] = total_training_time
    wandb_run.summary["warmup_time"] = warmup_time
    wandb_run.summary["total_hot_time"] = total_training_time - warmup_time
    wandb_run.summary["min_val_bpb"] = min_val_bpb

if args.profile:
    try:
        profiler.stop()
    except Exception:
        pass

# Log to report
from nanochat.report import get_report
get_report().log(section="SFT", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of iterations": step,
        "DDP world size": ddp_world_size,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()

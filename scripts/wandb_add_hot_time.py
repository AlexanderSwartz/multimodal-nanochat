#!/usr/bin/env python3
"""
Update a completed WandB run's summary by adding `total_hot_time` =
`total_training_time - warmup_time`.
"""

import argparse
import sys
import wandb

def get_step_val(row):
    # Prefer explicit step keys over wandb's internal _step
    for k in ["global_step", "step", "train/global_step", "train/step"]:
        if k in row and row[k] is not None:
            return row[k], k
    # Substring match (avoiding internal _step unless necessary)
    for k in row.keys():
        if "step" in k.lower() and k != "_step" and row[k] is not None:
            return row[k], k
    # Fallback to internal _step
    if "_step" in row and row["_step"] is not None:
        return row["_step"], "_step"
    return None, None

def get_dt_val(row):
    exact_keys = ["dt", "duration", "time", "elapsed", "train/dt"]
    for k in exact_keys:
        if k in row and row[k] is not None:
            return row[k], k
    # Substring match
    for k in row.keys():
        k_low = k.lower()
        if any(sub in k_low for sub in ["dt", "duration", "elapsed", "time"]) and row[k] is not None:
            # Prevent catching timestamps or total times
            if "total" not in k_low and "stamp" not in k_low:
                return row[k], k
    return None, None

def main():
    ii = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path", type=str, default=None)
    parser.add_argument("--entity", type=str, default="as7629-columbia-university")
    parser.add_argument("--project", type=str, default="Multimodal-Nanochat")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    run = None

    try:
        if args.run_path:
            run = api.run(args.run_path)
        elif args.run_id:
            run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
        elif args.run_name:
            runs = api.runs(f"{args.entity}/{args.project}")
            run = next((r for r in runs if r.name == args.run_name), None)
            if not run:
                print(f"No run named '{args.run_name}' found.", file=sys.stderr)
                sys.exit(1)
        else:
            parser.error("Provide --run-path, or --entity with either --run-id or --run-name")
    except Exception as e:
        print(f"Failed to load run: {e}", file=sys.stderr)
        sys.exit(1)

    summary = run.summary
    if "total_training_time" not in summary:
        print(f"No 'total_training_time' in summary. Keys: {list(summary.keys())}", file=sys.stderr)
        sys.exit(1)

    total_training_time = float(summary["total_training_time"])

    # Collect dt values from history (ignore step); use the first 21 dt values
    dt_values = []
    seen_dt_keys = set()
    scanned_rows = 0

    try:
        for row in run.scan_history():
            scanned_rows += 1
            for k in row.keys():
                if k:
                    seen_dt_keys.add(k)
            dt_val, dt_key = get_dt_val(row)
            if dt_val is None:
                continue
            try:
                dt_values.append(float(dt_val))
            except Exception:
                # ignore malformed dt
                pass
    except Exception as e:
        print(f"Warning: failed to scan run history: {e}", file=sys.stderr)

    # Fallback to run.history if nothing found via scan_history
    if len(dt_values) == 0:
        try:
            for row in run.history(stream=False):
                for k in getattr(row, 'keys', lambda: [])():
                    if k:
                        seen_dt_keys.add(k)
                dt_val, dt_key = get_dt_val(row)
                if dt_val is None:
                    continue
                try:
                    dt_values.append(float(dt_val))
                except Exception:
                    pass
        except Exception:
            pass

    used_count = min(len(dt_values), 21)
    history_warmup = sum(dt_values[:used_count]) if used_count > 0 else 0.0
    matched_rows = used_count

    total_hot_time = total_training_time - history_warmup

    print(f"Run: {run.path}")
    print(f"  total_training_time = {total_training_time:.4f}")
    
    if matched_rows > 0:
        print(f"  Keys seen for time: {list(sorted(seen_dt_keys))[:20]}")
        print(f"  Matched {matched_rows} dt rows (used {used_count}).")
    else:
        print("  WARNING: Could not find any valid dt rows in steps 0-20. Check your history keys.")
        
    print(f"  warmup_time (Steps 0-20) = {history_warmup:.4f}")
    print(f"  computed total_hot_time  = {total_hot_time:.4f}")

    if args.dry_run:
        print("Dry run; not updating WandB.")
        return

    try:
        run.summary.update({
            "warmup_time": history_warmup, 
            "total_hot_time": total_hot_time
        })
        print("Updated run.summary successfully.")
    except Exception as e:
        print(f"Failed to update run summary: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
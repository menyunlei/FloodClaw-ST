#!/usr/bin/env python3
"""run_sequential_analysis.py — PHASE 5: Trajectory + revision analysis from raw predictions."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FLOOD_DIR = Path(__file__).parent.parent
RESULTS_DIR = FLOOD_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw_predictions"
METRICS_DIR = RESULTS_DIR / "metrics"


def load_raw_predictions(raw_dir: Path) -> list:
    """Load all raw prediction JSON files."""
    rows = []
    for f in sorted(raw_dir.glob("*.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")
    return rows


def write_stepwise_trajectory(rows: list, suffix: str = "") -> None:
    """stepwise_trajectory.csv: event_id, step_index, method, hit, semantic_score"""
    out_path = METRICS_DIR / f"stepwise_trajectory{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["event_id,step_index,method,hit,semantic_score"]
    for r in rows:
        lines.append(
            f"{r['event_id']},{r['step_index']},{r['method']},"
            f"{r['hit']},{r['semantic_score']:.4f}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({len(rows)} rows)")


def write_revision_effect(rows: list, suffix: str = "") -> None:
    """revision_effect.csv: compare full_floodclaw vs no_logic per step."""
    out_path = METRICS_DIR / f"revision_effect{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Index by (event_id, step_index, method)
    idx: dict = {}
    for r in rows:
        key = (r["event_id"], r["step_index"], r["method"])
        idx[key] = r

    lines = ["event_id,step_index,method,pre_revision_score,post_revision_score,delta"]
    seen = set()
    for r in rows:
        eid = r["event_id"]
        sidx = r["step_index"]
        pair_key = (eid, sidx)
        if pair_key in seen:
            continue
        seen.add(pair_key)

        no_logic = idx.get((eid, sidx, "no_logic"))
        full = idx.get((eid, sidx, "full_floodclaw"))
        if no_logic and full:
            pre = no_logic["semantic_score"]
            post = full["semantic_score"]
            delta = post - pre
            lines.append(f"{eid},{sidx},full_vs_no_logic,{pre:.4f},{post:.4f},{delta:+.4f}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def write_error_alignment_sample(rows: list, n: int = 10, suffix: str = "") -> None:
    """error_alignment_sample.csv: 10 sampled rows with full details."""
    out_path = METRICS_DIR / f"error_alignment_sample{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    sample = rng.sample(rows, min(n, len(rows)))

    lines = ["event_id,step_index,method,prediction,gold,hit,semantic_score"]
    for r in sample:
        pred = str(r.get("prediction", "")).replace(",", ";").replace("\n", " ")
        gold = str(r.get("gold", "")).replace(",", ";").replace("\n", " ")
        lines.append(
            f"{r['event_id']},{r['step_index']},{r['method']},"
            f"\"{pred}\",\"{gold}\","
            f"{r['hit']},{r['semantic_score']:.4f}"
        )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({len(sample)} sampled rows)")


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=None, help="Directory of raw prediction JSONs")
    p.add_argument("--suffix", default="", help="Output filename suffix")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DIR
    rows = load_raw_predictions(raw_dir)
    if not rows:
        print(f"No raw predictions found in {raw_dir}. Run run_ablation.py first.")
        return 1

    print(f"Loaded {len(rows)} raw prediction records")
    write_stepwise_trajectory(rows, args.suffix)
    write_revision_effect(rows, args.suffix)
    write_error_alignment_sample(rows, suffix=args.suffix)
    print("Sequential analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

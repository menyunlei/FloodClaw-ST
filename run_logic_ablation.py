#!/usr/bin/env python3
"""run_logic_ablation.py — PHASE 4: Logic revision ablation across 6 logic settings."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loader import load_replay_subset, iter_steps, format_prior_steps, format_summary_memory
from predictor import predict
from evaluator import evaluate
from metrics import aggregate_method_results

FLOOD_DIR = Path(__file__).parent.parent
DATA_PATH = FLOOD_DIR / "data" / "replay_subset.jsonl"  # default; overridden by --data-path
RULES_PATH = FLOOD_DIR / "optimized_flood_logic_rules.jsonl"
CONFIGS_DIR = FLOOD_DIR / "configs"
RESULTS_DIR = FLOOD_DIR / "results"

# Logic ablation settings: subset of methods focused on logic variants
LOGIC_METHOD_IDS = [
    "full_floodclaw",
    "no_logic",
    "fixed_macro_logic",
    "summary_memory",
    "explicit_logic_only",
    "revision_without_evaluator",
]


def load_rules() -> list:
    rules = []
    if RULES_PATH.exists():
        with RULES_PATH.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rules.append(json.loads(line))
    return rules


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default=None)
    p.add_argument("--suffix", default="")
    args = p.parse_args()

    endpoints = json.loads((CONFIGS_DIR / "endpoints.json").read_text())
    methods_cfg = json.loads((CONFIGS_DIR / "methods.json").read_text())["methods"]
    rules = load_rules()
    data_path = Path(args.data_path) if args.data_path else DATA_PATH
    events = load_replay_subset(data_path)

    predictor_cfg = endpoints["predictor"]
    evaluator_cfg = endpoints["evaluator"]

    methods = [m for m in methods_cfg if m["id"] in LOGIC_METHOD_IDS]
    all_results_by_method: dict = {}

    for method in methods:
        print(f"\n=== Logic Ablation: {method['name']} ===")
        method_results = []

        for event in events:
            print(f"  Event: {event['event_id']}")
            for step_index, observed_state, gold in iter_steps(event):
                if method["id"] == "summary_memory":
                    prior_text = format_summary_memory(event, step_index)
                else:
                    prior_text = format_prior_steps(event, step_index)

                logic_rules = rules if method.get("use_logic") else None

                print(f"    step {step_index} ... ", end="", flush=True)
                try:
                    prediction = predict(
                        observed_state=observed_state,
                        method_config=method,
                        prior_steps_text=prior_text,
                        step_index=step_index,
                        logic_rules=logic_rules,
                        predictor_config=predictor_cfg,
                    )
                except Exception as e:
                    print(f"PREDICT ERROR: {e}")
                    prediction = f"[ERROR: {e}]"

                try:
                    eval_result = evaluate(prediction, gold, evaluator_cfg)
                except Exception as e:
                    print(f"EVAL ERROR: {e}")
                    eval_result = {"hit": 0, "semantic_score": 0.0, "keyword_overlap": 0.0}

                row = {
                    "event_id": event["event_id"],
                    "method": method["id"],
                    "step_index": step_index,
                    "hit": eval_result["hit"],
                    "semantic_score": eval_result["semantic_score"],
                    "keyword_overlap": eval_result.get("keyword_overlap", 0.0),
                }
                method_results.append(row)
                print(f"hit={eval_result['hit']} sem={eval_result['semantic_score']:.3f}")

        all_results_by_method[method["id"]] = method_results

    # Write logic ablation CSV
    out_rows = ["method,hit_rate,semantic_score,event_avg_hit_rate,late_stage_hit_rate,n_steps,n_hits"]
    for method in methods:
        mid = method["id"]
        if mid not in all_results_by_method:
            continue
        agg = aggregate_method_results(all_results_by_method[mid])
        out_rows.append(
            f"{mid},{agg['hit_rate']:.4f},{agg['semantic_score']:.4f},"
            f"{agg['event_avg_hit_rate']:.4f},{agg['late_stage_hit_rate']:.4f},"
            f"{agg['n_steps']},{agg['n_hits']}"
        )

    out_path = RESULTS_DIR / "tables" / f"logic_revision_ablation{args.suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_rows) + "\n")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

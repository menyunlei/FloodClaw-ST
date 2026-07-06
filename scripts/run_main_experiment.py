#!/usr/bin/env python3
"""Main experiment runner for Nebraska 2019 key-event reasoning.

- Reasoner (test-time): Qwen endpoint
- Judge/feedback: Deepseek endpoint
- Core protocol: case-based, feedback-driven logic update
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List
from urllib import request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run main flood reasoning experiment")
    parser.add_argument(
        "--data-jsonl",
        default="/hpc2hdd/home/xhe989/2541/model/Nebraska_Flood_2019_Key_Events.jsonl",
        help="Path to 12-step key events jsonl",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeated runs",
    )
    parser.add_argument(
        "--output-json",
        default="/hpc2hdd/home/xhe989/2541/model/main_experiment_results.json",
        help="Detailed JSON output",
    )
    parser.add_argument(
        "--output-report",
        default="/hpc2hdd/home/xhe989/2541/model/main_experiment_report.txt",
        help="Text summary report",
    )
    parser.add_argument(
        "--qwen-base",
        default="http://localhost:8072/v1",
    )
    parser.add_argument(
        "--qwen-model",
        default="/home/yunlei/flood/model/Qwen3.5-27B-FP8",
    )
    parser.add_argument(
        "--deepseek-base",
        default="http://10.120.20.222:27007/v1",
    )
    parser.add_argument(
        "--deepseek-model",
        default="/hpc2hdd/home/xhe989/Documents/ExGRPO/qwen/model/flood/Llama-3.1-70B-Instruct",
    )
    parser.add_argument(
        "--deepseek-key",
        default="",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda x: int(x.get("event_id", 0)))
    return rows


def call_chat(
    base: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: str | None = None,
    temperature: float = 0.0,
    retries: int = 6,
) -> str:
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ctx = ssl._create_unverified_context() if base.startswith("https") else None
    last = None
    for i in range(retries):
        try:
            req = request.Request(url, data=data, headers=headers, method="POST")
            with request.urlopen(req, context=ctx, timeout=180) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            return obj["choices"][0]["message"]["content"]
        except Exception as exc:
            last = exc
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f"chat failed: {last}")


def parse_json_obj(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}


def canonical_category(cat: str) -> str:
    c = (cat or "").strip().lower()
    mapping = {
        "natural trigger": "Nature Events",
        "natural response": "Natural Responses",
        "infrastructure": "Infrastructure",
        "intervention": "Interventions",
        "trajectory / outcome": "Trajectories",
        "trajectory outcome": "Trajectories",
        "nature events": "Nature Events",
        "natural responses": "Natural Responses",
        "interventions": "Interventions",
        "trajectories": "Trajectories",
    }
    return mapping.get(c, cat)


def reasoning_predict_next(
    qwen_base: str,
    qwen_model: str,
    current_event: Dict[str, Any],
    logic_memory: List[str],
    temperature: float,
) -> Dict[str, Any]:
    logic_text = "\n".join([f"- {x}" for x in logic_memory[-8:]]) if logic_memory else "- (empty)"
    prompt = (
        "You are a flood reasoning model. Given current state, predict the NEXT key event.\n"
        "Use concise causal reasoning. Return strict JSON keys:\n"
        "category, trigger, response, infrastructure, intervention, outcome, explanation.\n\n"
        f"Current event:\n{json.dumps(current_event, ensure_ascii=False)}\n\n"
        f"Current learned logic:\n{logic_text}\n"
    )
    raw = call_chat(
        base=qwen_base,
        model=qwen_model,
        messages=[
            {"role": "system", "content": "You are concise and deterministic in JSON."},
            {"role": "user", "content": prompt},
        ],
        api_key=None,
        temperature=temperature,
    )
    return parse_json_obj(raw)


def reasoning_predict_previous(
    qwen_base: str,
    qwen_model: str,
    current_event: Dict[str, Any],
    learned_logic: List[str],
) -> Dict[str, Any]:
    logic_text = "\n".join([f"- {x}" for x in learned_logic[-8:]]) if learned_logic else "- (empty)"
    prompt = (
        "You are a flood reasoning model in BACKWARD mode.\n"
        "Given current event, infer the IMMEDIATE PREVIOUS key event. Return strict JSON keys:\n"
        "category, trigger, response, infrastructure, intervention, outcome, explanation.\n\n"
        f"Current event:\n{json.dumps(current_event, ensure_ascii=False)}\n\n"
        f"Learned logic:\n{logic_text}\n"
    )
    raw = call_chat(
        base=qwen_base,
        model=qwen_model,
        messages=[
            {"role": "system", "content": "You are concise and deterministic in JSON."},
            {"role": "user", "content": prompt},
        ],
        api_key=None,
        temperature=0.0,
    )
    return parse_json_obj(raw)


def deepseek_judge(
    deepseek_base: str,
    deepseek_model: str,
    deepseek_key: str,
    predicted_event: Dict[str, Any],
    reference_event: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = (
        "Evaluate prediction vs reference for one flood key event.\n"
        "Rules:\n"
        "1) category_match = 1 only if categories are semantically equivalent.\n"
        "2) trigger_match/response_match/infrastructure_match/intervention_match/outcome_match = 1 only if both fields non-empty and semantically matching.\n"
        "3) event_correct = 1 if category_match=1 and at least one among trigger/response/infrastructure/intervention/outcome matches.\n"
        "4) If no non-category fields match, event_correct must be 0.\n"
        "Return strict JSON keys: category_match, trigger_match, response_match, infrastructure_match, intervention_match, outcome_match, event_correct, feedback_note.\n\n"
        f"Prediction:\n{json.dumps(predicted_event, ensure_ascii=False)}\n\n"
        f"Reference:\n{json.dumps(reference_event, ensure_ascii=False)}\n"
    )
    raw = call_chat(
        base=deepseek_base,
        model=deepseek_model,
        messages=[
            {"role": "system", "content": "You are a strict evaluator."},
            {"role": "user", "content": prompt},
        ],
        api_key=deepseek_key,
        temperature=0.0,
    )
    obj = parse_json_obj(raw)

    def bit(name: str) -> int:
        try:
            return 1 if int(obj.get(name, 0)) == 1 else 0
        except Exception:
            return 0

    result = {
        "category_match": bit("category_match"),
        "trigger_match": bit("trigger_match"),
        "response_match": bit("response_match"),
        "infrastructure_match": bit("infrastructure_match"),
        "intervention_match": bit("intervention_match"),
        "outcome_match": bit("outcome_match"),
        "event_correct": bit("event_correct"),
        "feedback_note": (obj.get("feedback_note", "") or "").strip(),
    }

    # Hard enforce event_correct rule to reduce judge drift.
    non_cat = (
        result["trigger_match"]
        + result["response_match"]
        + result["infrastructure_match"]
        + result["intervention_match"]
        + result["outcome_match"]
    )
    result["event_correct"] = 1 if (result["category_match"] == 1 and non_cat >= 1) else 0
    return result


def to_reference_schema(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "category": canonical_category(str(event.get("category", ""))),
        "trigger": str(event.get("trigger", "") or ""),
        "response": str(event.get("response", "") or ""),
        "infrastructure": str(event.get("infrastructure", "") or ""),
        "intervention": str(event.get("intervention", "") or ""),
        "outcome": str(event.get("outcome", "") or ""),
    }


def run_single_repeat(events: List[Dict[str, Any]], args: argparse.Namespace, repeat_id: int) -> Dict[str, Any]:
    rng = random.Random(1000 + repeat_id)

    baseline_correct = 0
    baseline_total = 0

    feedback_correct = 0
    feedback_total = 0
    logic_memory: List[str] = []

    # Forward sequence prediction: event_t -> predict event_{t+1}
    for i in range(len(events) - 1):
        current = to_reference_schema(events[i])
        target_next = to_reference_schema(events[i + 1])

        # Baseline: no logic memory, no feedback update.
        pred_base = reasoning_predict_next(
            qwen_base=args.qwen_base,
            qwen_model=args.qwen_model,
            current_event=current,
            logic_memory=[],
            temperature=0.2 + 0.05 * rng.random(),
        )
        judge_base = deepseek_judge(
            deepseek_base=args.deepseek_base,
            deepseek_model=args.deepseek_model,
            deepseek_key=args.deepseek_key,
            predicted_event=to_reference_schema(pred_base),
            reference_event=target_next,
        )
        baseline_correct += judge_base["event_correct"]
        baseline_total += 1

        # Feedback-driven: uses logic memory and updates it with judge feedback.
        pred_fb = reasoning_predict_next(
            qwen_base=args.qwen_base,
            qwen_model=args.qwen_model,
            current_event=current,
            logic_memory=logic_memory,
            temperature=0.2 + 0.05 * rng.random(),
        )
        judge_fb = deepseek_judge(
            deepseek_base=args.deepseek_base,
            deepseek_model=args.deepseek_model,
            deepseek_key=args.deepseek_key,
            predicted_event=to_reference_schema(pred_fb),
            reference_event=target_next,
        )
        feedback_correct += judge_fb["event_correct"]
        feedback_total += 1

        fb_note = judge_fb.get("feedback_note", "")
        if fb_note:
            logic_memory.append(f"Step {i+1}: {fb_note}")

    # Backward reasoning with learned logic (from feedback branch).
    backward_correct = 0
    backward_total = 0
    for j in range(len(events) - 1, 0, -1):
        current = to_reference_schema(events[j])
        target_prev = to_reference_schema(events[j - 1])

        pred_prev = reasoning_predict_previous(
            qwen_base=args.qwen_base,
            qwen_model=args.qwen_model,
            current_event=current,
            learned_logic=logic_memory,
        )
        judge_prev = deepseek_judge(
            deepseek_base=args.deepseek_base,
            deepseek_model=args.deepseek_model,
            deepseek_key=args.deepseek_key,
            predicted_event=to_reference_schema(pred_prev),
            reference_event=target_prev,
        )
        backward_correct += judge_prev["event_correct"]
        backward_total += 1

    baseline_acc = baseline_correct / baseline_total if baseline_total else 0.0
    feedback_acc = feedback_correct / feedback_total if feedback_total else 0.0
    backward_acc = backward_correct / backward_total if backward_total else 0.0
    delta = (feedback_acc - baseline_acc) / baseline_acc if baseline_acc > 0 else 0.0

    return {
        "repeat_id": repeat_id,
        "baseline_acc": baseline_acc,
        "feedback_acc": feedback_acc,
        "backward_acc": backward_acc,
        "delta_acc": delta,
        "logic_size": len(logic_memory),
        "logic_tail": logic_memory[-3:],
    }


def main() -> int:
    args = parse_args()
    events = load_jsonl(Path(args.data_jsonl))
    if len(events) < 2:
        raise RuntimeError("Need at least 2 events in data-jsonl")

    all_runs = []
    for r in range(1, args.repeats + 1):
        run = run_single_repeat(events, args, r)
        all_runs.append(run)
        print(
            f"[repeat {r}] baseline={run['baseline_acc']:.4f} feedback={run['feedback_acc']:.4f} "
            f"backward={run['backward_acc']:.4f} delta={run['delta_acc']:.4f} logic={run['logic_size']}"
        )

    baseline_list = [x["baseline_acc"] for x in all_runs]
    feedback_list = [x["feedback_acc"] for x in all_runs]
    backward_list = [x["backward_acc"] for x in all_runs]
    delta_list = [x["delta_acc"] for x in all_runs]

    summary = {
        "setup": {
            "source_event": "Nebraska Flood 2019 (12 key events)",
            "reasoner": "Llama-3.1-70B-Instruct (10.120.20.222:27007)",
            "judge": "Deepseek (hpc4login endpoint)",
            "repeats": args.repeats,
            "notes": [
                "Backward reasoning executed on reverse sequence using learned logic.",
                "Forward transfer (2024 Asheville) skipped: dataset not found in workspace.",
                "Cross-lingual portability (100 Arabic cases/Jais) skipped: dataset/model set not found in workspace.",
            ],
        },
        "metrics": {
            "baseline_avg_acc": mean(baseline_list),
            "feedback_avg_acc": mean(feedback_list),
            "backward_avg_acc": mean(backward_list),
            "avg_delta_acc": mean(delta_list),
        },
        "runs": all_runs,
    }

    output_json = Path(args.output_json)
    output_report = Path(args.output_report)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = []
    lines.append("Main Experiment Report")
    lines.append("")
    lines.append("Setup")
    lines.append(f"- Source event: {summary['setup']['source_event']}")
    lines.append(f"- Reasoner: {summary['setup']['reasoner']}")
    lines.append(f"- Judge: {summary['setup']['judge']}")
    lines.append(f"- Repeats: {summary['setup']['repeats']}")
    lines.append("")
    lines.append("Average Metrics")
    lines.append(f"- Baseline Accuracy: {summary['metrics']['baseline_avg_acc']:.4f}")
    lines.append(f"- Feedback Accuracy: {summary['metrics']['feedback_avg_acc']:.4f}")
    lines.append(f"- Backward Accuracy: {summary['metrics']['backward_avg_acc']:.4f}")
    lines.append(f"- Delta vs Baseline: {summary['metrics']['avg_delta_acc']:.4f}")
    lines.append("")
    lines.append("Per-Run Metrics")
    for r in all_runs:
        lines.append(
            f"- Run {r['repeat_id']}: baseline={r['baseline_acc']:.4f}, feedback={r['feedback_acc']:.4f}, "
            f"backward={r['backward_acc']:.4f}, delta={r['delta_acc']:.4f}, logic_size={r['logic_size']}"
        )
    lines.append("")
    lines.append("Data Availability Notes")
    for note in summary["setup"]["notes"]:
        lines.append(f"- {note}")

    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"WROTE {output_json}")
    print(f"WROTE {output_report}")
    print(
        "SUMMARY "
        f"baseline={summary['metrics']['baseline_avg_acc']:.4f} "
        f"feedback={summary['metrics']['feedback_avg_acc']:.4f} "
        f"backward={summary['metrics']['backward_avg_acc']:.4f} "
        f"delta={summary['metrics']['avg_delta_acc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

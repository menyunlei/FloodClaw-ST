#!/usr/bin/env python3
"""
并行评估 - 128并发，包含完整指标
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import ssl
from urllib import request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

EVALUATOR_URL = "http://10.120.20.222:27007/v1/chat/completions"
EVALUATOR_MODEL = "/hpc2hdd/home/xhe989/Documents/ExGRPO/qwen/model/flood/Llama-3.3-70B-Instruct"
PREDICTOR_URL = "https://hpc3login.hpc.hkust-gz.edu.cn/endpoints/mindinsight/eyJwb3J0Ijo4MDgwLCJzZXJ2aWNlIjoieHgtZGV2LTBhYjMwMDcxLTczYzctNDM2YS1hMzQxLTU1ZGQ1YTQ5ZjI1Zi5hcHVsaXMuc3ZjLmNsdXN0ZXIubG9jYWwifQ==/v1/chat/completions"
PREDICTOR_MODEL = "/home/apulis-dev/userdata/ExGRPO/model/Qwen2.5-14B-Instruct"

lock = threading.Lock()
counter = {"done": 0, "total": 0}


def call_llama_evaluator(prediction: Dict[str, str], reference: Dict[str, str]) -> Dict[str, Any]:
    """使用Llama模型进行语义评估"""
    prompt = (
        "Evaluate semantic equivalence between predicted and reference flood events.\n"
        "Aspects: trigger, response, infrastructure, intervention, outcome.\n"
        "Return JSON keys: semantic_score(0..1), key_event_match(0/1), matched_aspects(array).\n"
        "Rule: key_event_match=1 if semantic_score>=0.6 OR at least 3 matched aspects.\n\n"
        f"Prediction: {json.dumps({k: prediction[k] for k in ['trigger','response','infrastructure','intervention','outcome']}, ensure_ascii=False)}\n"
        f"Reference: {json.dumps({k: reference[k] for k in ['trigger','response','infrastructure','intervention','outcome']}, ensure_ascii=False)}\n"
    )

    payload = {
        "model": EVALUATOR_MODEL,
        "messages": [
            {"role": "system", "content": "You are a strict semantic evaluator for flood events."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    ctx = ssl._create_unverified_context()

    try:
        req = request.Request(EVALUATOR_URL, data=data, headers=headers, method="POST")
        with request.urlopen(req, context=ctx, timeout=180) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(raw)
        content = obj["choices"][0]["message"]["content"]

        try:
            result = json.loads(content)
        except:
            s = content.find("{")
            e = content.rfind("}")
            if s >= 0 and e > s:
                result = json.loads(content[s:e+1])
            else:
                result = {"semantic_score": 0.0, "key_event_match": 0}

        return {
            "semantic_score": float(result.get("semantic_score", 0.0)),
            "key_event_match": int(result.get("key_event_match", 0)),
            "matched_aspects": result.get("matched_aspects", []),
        }
    except Exception as e:
        return {
            "semantic_score": 0.0,
            "key_event_match": 0,
            "matched_aspects": [],
            "error": str(e)
        }


def call_qwen_predictor(event: Dict[str, str], applicable_logic: List[Dict] = None) -> Dict[str, str]:
    """调用Qwen模型进行预测"""
    prompt = "Predict the immediate NEXT flood key event.\n"
    prompt += "Return JSON keys: trigger, response, infrastructure, intervention, outcome, confidence, reasoning.\n"
    prompt += "Do not output analysis text outside JSON.\n\n"

    if applicable_logic:
        prompt += "=== APPLICABLE LOGIC RULES ===\n"
        for i, logic in enumerate(applicable_logic, 1):
            prompt += f"\nRule {i}:\n"
            prompt += f"  Trigger: {logic.get('trigger', '')}\n"
            prompt += f"  Intermediate State: {logic.get('intermediate_state', '')}\n"
            prompt += f"  Potential Outcome: {logic.get('potential_outcome', '')}\n"
            prompt += f"  Intervention: {logic.get('intervention', '')}\n"
            prompt += f"  Principle: {logic.get('principle', '')}\n"
        prompt += "\n=== END LOGIC RULES ===\n\n"

    prompt += f"Current event: {json.dumps({k: event[k] for k in ['trigger','response','infrastructure','intervention','outcome']}, ensure_ascii=False)}\n"

    payload = {
        "model": PREDICTOR_MODEL,
        "messages": [
            {"role": "system", "content": "You are a flood progression predictor."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.12,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    ctx = ssl._create_unverified_context()

    try:
        req = request.Request(PREDICTOR_URL, data=data, headers=headers, method="POST")
        with request.urlopen(req, context=ctx, timeout=180) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(raw)
        content = obj["choices"][0]["message"]["content"]

        try:
            pred = json.loads(content)
        except:
            s = content.find("{")
            e = content.rfind("}")
            if s >= 0 and e > s:
                pred = json.loads(content[s:e+1])
            else:
                pred = {}

        return {
            "trigger": str(pred.get("trigger", "") or ""),
            "response": str(pred.get("response", "") or ""),
            "infrastructure": str(pred.get("infrastructure", "") or ""),
            "intervention": str(pred.get("intervention", "") or ""),
            "outcome": str(pred.get("outcome", "") or ""),
            "confidence": pred.get("confidence", 0.5),
        }
    except Exception as e:
        return {
            "trigger": "Unknown",
            "response": "Unknown",
            "infrastructure": "",
            "intervention": "",
            "outcome": "Unknown",
            "confidence": 0.0,
            "error": str(e)
        }


def process_one(i: int, events: List[Dict], applicable_logics: List) -> Dict:
    """处理一个预测+评估"""
    pred = call_qwen_predictor(events[i], applicable_logic=applicable_logics[i])
    eval_result = call_llama_evaluator(pred, events[i + 1])

    with lock:
        counter["done"] += 1
        if counter["done"] % 10 == 0:
            print(f"  进度: {counter['done']}/{counter['total']}", flush=True)

    return {
        "semantic_score": eval_result["semantic_score"],
        "key_event_match": eval_result["key_event_match"],
        "confidence": float(pred.get("confidence", 0.5)) if isinstance(pred.get("confidence"), (int, float)) else 0.5,
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_event(item: Dict[str, Any]) -> Dict[str, str]:
    if "event" in item:
        return item["event"]
    return item


def compute_metrics(predictions: List[int], references: List[int]) -> Dict[str, float]:
    """计算 Precision, Recall, F1, Raw Agreement, Cohen's Kappa"""
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        references, predictions, average='binary', zero_division=0
    )

    # Raw Agreement
    raw_agreement = np.mean(np.array(predictions) == np.array(references))

    # Cohen's Kappa
    kappa = cohen_kappa_score(references, predictions)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "raw_agreement": float(raw_agreement),
        "cohens_kappa": float(kappa),
    }


def run_parallel_evaluation(target_file: str, repeats: int = 3, workers: int = 128):
    print("="*80)
    print(f"并行评估 - {workers} 并发")
    print("="*80)

    print(f"\n📁 加载数据: {target_file}")
    target_data = read_jsonl(Path(target_file))
    print(f"   加载了 {len(target_data)} 个事件")

    events = [extract_event(item) for item in target_data]
    applicable_logics = [item.get("applicable_logic", None) for item in target_data]

    all_results = []

    for repeat in range(repeats):
        print(f"\n【重复 {repeat + 1}/{repeats}】")

        counter["done"] = 0
        counter["total"] = len(events) - 1

        t0 = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_one, i, events, applicable_logics): i
                      for i in range(len(events) - 1)}

            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"  ✗ 任务失败: {e}")
                    results.append({
                        "semantic_score": 0.0,
                        "key_event_match": 0,
                        "confidence": 0.5,
                    })

        elapsed = time.time() - t0

        # 提取预测和参考标签
        predictions = [r["key_event_match"] for r in results]
        # 参考标签应该基于语义分数阈值（>=0.6为正例）
        references = [1 if r["semantic_score"] >= 0.6 else 0 for r in results]

        logic_semantic_scores = [r["semantic_score"] for r in results]
        logic_confidences = [r["confidence"] for r in results]

        hit_rate = np.mean(predictions)
        semantic_mean = np.mean(logic_semantic_scores) if logic_semantic_scores else 0.0

        # 计算额外指标
        metrics = compute_metrics(predictions, references)

        print(f"  ✓ Hit Rate: {hit_rate:.4f}, Semantic Score: {semantic_mean:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
        print(f"    Raw Agreement: {metrics['raw_agreement']:.4f}, Cohen's κ: {metrics['cohens_kappa']:.4f}")
        print(f"    耗时: {elapsed:.1f}s")

        all_results.append({
            'hit_rate': hit_rate,
            'semantic_score': semantic_mean,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'raw_agreement': metrics['raw_agreement'],
            'cohens_kappa': metrics['cohens_kappa'],
            'confidences': logic_confidences,
        })

    print("\n" + "="*80)
    print("📊 评估总结")
    print("="*80)

    hit_rates = [r['hit_rate'] for r in all_results]
    semantic_scores = [r['semantic_score'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    f1_scores = [r['f1_score'] for r in all_results]
    raw_agreements = [r['raw_agreement'] for r in all_results]
    cohens_kappas = [r['cohens_kappa'] for r in all_results]

    print(f"\n  Hit Rate: {np.mean(hit_rates):.4f} ± {np.std(hit_rates):.4f}")
    print(f"  Semantic Score: {np.mean(semantic_scores):.4f} ± {np.std(semantic_scores):.4f}")
    print(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"  Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"  Raw Agreement: {np.mean(raw_agreements):.4f} ± {np.std(raw_agreements):.4f}")
    print(f"  Cohen's κ: {np.mean(cohens_kappas):.4f} ± {np.std(cohens_kappas):.4f}")

    output_file = Path("parallel_evaluation_results.json")
    results_to_save = {
        'summary': {
            'hit_rate_mean': float(np.mean(hit_rates)),
            'hit_rate_std': float(np.std(hit_rates)),
            'semantic_score_mean': float(np.mean(semantic_scores)),
            'semantic_score_std': float(np.std(semantic_scores)),
            'precision_mean': float(np.mean(precisions)),
            'precision_std': float(np.std(precisions)),
            'recall_mean': float(np.mean(recalls)),
            'recall_std': float(np.std(recalls)),
            'f1_score_mean': float(np.mean(f1_scores)),
            'f1_score_std': float(np.std(f1_scores)),
            'raw_agreement_mean': float(np.mean(raw_agreements)),
            'raw_agreement_std': float(np.std(raw_agreements)),
            'cohens_kappa_mean': float(np.mean(cohens_kappas)),
            'cohens_kappa_std': float(np.std(cohens_kappas)),
        },
        'detailed_results': all_results,
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ 结果已保存到: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并行评估")
    parser.add_argument("--target-jsonl", default="data/logic_enriched_926.jsonl")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=128)

    args = parser.parse_args()

    run_parallel_evaluation(args.target_jsonl, args.repeats, args.workers)

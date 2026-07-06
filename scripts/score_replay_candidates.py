#!/usr/bin/env python3
"""score_replay_candidates.py — Score prior_case_pool for sequential replay suitability."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

FLOOD_DIR = Path(__file__).parent.parent
POOL_PATH = FLOOD_DIR / "data" / "prior_case_pool.jsonl"
OUT_CSV   = FLOOD_DIR / "data" / "replay_candidate_ranked.csv"
OUT_JSONL = FLOOD_DIR / "data" / "replay_candidate_top40.jsonl"

# ---------------------------------------------------------------------------
# Keyword signals for scoring
# ---------------------------------------------------------------------------

STAGE_KEYWORDS = [
    # physical progression
    r"\bintensif\w*\b", r"\bstrengthen\w*\b", r"\brapidly\b", r"\bescalat\w*\b",
    r"\bspread\w*\b", r"\bexpand\w*\b", r"\brise\w*\b", r"\boverflow\w*\b",
    r"\bovertopp\w*\b", r"\bbreached?\b", r"\bcollaps\w*\b", r"\bsubsid\w*\b",
    r"\brecede\w*\b", r"\brecovery\b", r"\brestoration\b", r"\bstabiliz\w*\b",
    # temporal markers
    r"\bdays?\b", r"\bweeks?\b", r"\bhours?\b", r"\bafter\b", r"\bfollowing\b",
    r"\binitial\b", r"\bsubsequent\b", r"\beventually\b", r"\bphase\b",
]

INFRA_KEYWORDS = [
    r"\broad\w*\b", r"\bbridge\w*\b", r"\bhighway\w*\b", r"\blevee\w*\b",
    r"\bdam\b", r"\bpower\b", r"\boutage\w*\b", r"\butility\b", r"\butilities\b",
    r"\binfrastructure\b", r"\baccess\b", r"\bclosure\w*\b", r"\bdisrupt\w*\b",
    r"\bdamage\w*\b", r"\bdestroy\w*\b", r"\bwashed out\b", r"\bflooded\b",
    r"\bsubstation\b", r"\bsewer\b", r"\bdrainage\b",
]

INTERVENTION_KEYWORDS = [
    r"\bevacuat\w*\b", r"\brescue\w*\b", r"\bemergency\b", r"\bresponse\b",
    r"\bshelter\w*\b", r"\brelief\b", r"\baid\b", r"\bdeployed?\b",
    r"\bFEMA\b", r"\bmilitary\b", r"\bnational guard\b", r"\bwarning\w*\b",
    r"\bpreparedness\b", r"\bcoordinat\w*\b", r"\bmutual assistance\b",
    r"\bdisaster declaration\b", r"\bstate of emergency\b",
]

RECOVERY_KEYWORDS = [
    r"\brestoration\b", r"\brecovery\b", r"\brebuild\w*\b", r"\brepair\w*\b",
    r"\breopen\w*\b", r"\brehabilitat\w*\b", r"\brestored?\b", r"\bprogress\b",
    r"\bassessment\b", r"\bdamage assessment\b", r"\bfederal assistance\b",
    r"\blong.term\b", r"\bmonths?\b",
]

SOURCE_STRONG = [
    r"\bNWS\b", r"\bFEMA\b", r"\bNOAA\b", r"\bUSACE\b", r"\bDOE\b",
    r"\bEM-DAT\b", r"\bWorld Bank\b", r"\bUN\b", r"\bIFRC\b", r"\bRed Cross\b",
    r"\bgovernment report\b", r"\bofficial\b", r"\bsituation report\b",
]


def count_matches(text: str, patterns: list) -> int:
    t = text.lower()
    return sum(1 for p in patterns if re.search(p, t, re.IGNORECASE))


def all_text(case: dict) -> str:
    s = case.get("case_summary", {})
    parts = (
        s.get("trigger_factors", [])
        + s.get("physical_responses", [])
        + s.get("infrastructure_status", [])
        + s.get("interventions", [])
        + s.get("outcomes", [])
        + case.get("language_sources", [])
    )
    return " ".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_timeline_density(case: dict) -> tuple[int, str]:
    """0-3: how many distinguishable stages can be inferred."""
    s = case.get("case_summary", {})
    n_physical = len(s.get("physical_responses", []))
    n_infra    = len(s.get("infrastructure_status", []))
    n_interv   = len(s.get("interventions", []))
    n_outcome  = len(s.get("outcomes", []))
    text = all_text(case)

    # Count distinct non-empty field types as rough stage proxies
    filled_sections = sum([
        n_physical > 0,
        n_infra > 0,
        n_interv > 0,
        n_outcome > 0,
    ])

    # Temporal/progression keyword density
    stage_hits = count_matches(text, STAGE_KEYWORDS)

    # Score
    if filled_sections >= 4 and stage_hits >= 4:
        return 3, f"4 sections filled, {stage_hits} stage keywords"
    elif filled_sections >= 3 and stage_hits >= 2:
        return 2, f"{filled_sections} sections, {stage_hits} stage keywords"
    elif filled_sections >= 2 or stage_hits >= 2:
        return 1, f"{filled_sections} sections, {stage_hits} stage keywords"
    else:
        return 0, "single summary only"


def score_trigger_clarity(case: dict) -> tuple[int, str]:
    triggers = [t for t in case.get("trigger_type", []) if t != "unknown"]
    tf = case.get("case_summary", {}).get("trigger_factors", [])
    if triggers and tf:
        return 1, f"trigger={triggers[0]}"
    elif triggers or tf:
        return 1, "trigger identifiable"
    return 0, "trigger unclear"


def score_infrastructure(case: dict) -> tuple[int, str]:
    s = case.get("case_summary", {})
    infra_list = s.get("infrastructure_status", [])
    text = all_text(case)
    hits = count_matches(text, INFRA_KEYWORDS)
    infra_ctx = case.get("infrastructure_context", [])

    if len(infra_list) >= 2 or (hits >= 4 and infra_ctx):
        return 2, f"{len(infra_list)} infra items, {hits} keywords"
    elif infra_list or hits >= 2 or infra_ctx:
        return 1, f"{len(infra_list)} infra items, {hits} keywords"
    return 0, "no infrastructure signal"


def score_intervention(case: dict) -> tuple[int, str]:
    s = case.get("case_summary", {})
    interv_list = s.get("interventions", [])
    text = all_text(case)
    hits = count_matches(text, INTERVENTION_KEYWORDS)

    if len(interv_list) >= 2 or hits >= 4:
        return 2, f"{len(interv_list)} interventions, {hits} keywords"
    elif interv_list or hits >= 1:
        return 1, f"{len(interv_list)} interventions, {hits} keywords"
    return 0, "no intervention signal"


def score_source_richness(case: dict) -> tuple[int, str]:
    text = all_text(case)
    hits = count_matches(text, SOURCE_STRONG)
    src = case.get("language_sources", [])
    # 100yr events have structured source hints; real_flood has source_hint field
    if hits >= 1 or len(src) > 1:
        return 1, f"{hits} source keywords"
    # Confidence proxy: if trigger is non-unknown and multiple fields filled
    s = case.get("case_summary", {})
    filled = sum([
        bool(s.get("trigger_factors")),
        bool(s.get("physical_responses")),
        bool(s.get("infrastructure_status")),
        bool(s.get("interventions")),
        bool(s.get("outcomes")),
    ])
    if filled >= 4:
        return 1, "4+ fields filled"
    return 0, "weak source support"


def score_recovery_tail(case: dict) -> tuple[int, str]:
    text = all_text(case)
    hits = count_matches(text, RECOVERY_KEYWORDS)
    outcomes = case.get("case_summary", {}).get("outcomes", [])
    outcome_text = " ".join(outcomes).lower()
    recovery_in_outcome = any(k in outcome_text for k in [
        "recovery", "restoration", "rebuild", "repair", "reopen", "rehabilit",
        "months", "long-term", "assessment", "progress",
    ])
    if hits >= 2 or recovery_in_outcome:
        return 1, f"{hits} recovery keywords"
    return 0, "no recovery tail"


def recommended_status(score: int) -> str:
    if score >= 7:
        return "replay_top"
    elif score >= 5:
        return "replay_possible"
    elif score >= 3:
        return "prior_only"
    return "too_sparse"


def build_notes(tl, tr, inf, inv, src, rec) -> str:
    parts = []
    if tl[0] >= 2: parts.append(f"timeline:{tl[1]}")
    if inf[0] >= 1: parts.append(f"infra:{inf[1]}")
    if inv[0] >= 1: parts.append(f"interv:{inv[1]}")
    if rec[0] == 1: parts.append("recovery_tail")
    if not parts: parts.append("sparse")
    return "; ".join(parts)[:120]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cases = []
    with POOL_PATH.open() as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"Processing {len(cases)} events...")

    scored = []
    for case in cases:
        tl  = score_timeline_density(case)
        tr  = score_trigger_clarity(case)
        inf = score_infrastructure(case)
        inv = score_intervention(case)
        src = score_source_richness(case)
        rec = score_recovery_tail(case)

        total = tl[0] + tr[0] + inf[0] + inv[0] + src[0] + rec[0]
        status = recommended_status(total)
        notes = build_notes(tl, tr, inf, inv, src, rec)

        scored.append({
            "event_id": case["event_id"],
            "event_name": case["event_name"],
            "region_group": case["region_group"],
            "country": case["country"],
            "year": case["year"],
            "trigger_type": "|".join(case.get("trigger_type", [])),
            "flood_type": "|".join(case.get("flood_type", [])),
            "replay_score": total,
            "timeline_score": tl[0],
            "trigger_score": tr[0],
            "infrastructure_score": inf[0],
            "intervention_score": inv[0],
            "source_score": src[0],
            "recovery_score": rec[0],
            "recommended_status": status,
            "notes": notes,
            "_case": case,  # keep for JSONL export
        })

    scored.sort(key=lambda x: x["replay_score"], reverse=True)

    # Print top 10
    print(f"\nTop 10 replay candidates:")
    print(f"{'event_id':<45} {'score':>5}  {'status':<16}  notes")
    print("-" * 100)
    for r in scored[:10]:
        print(f"{r['event_id']:<45} {r['replay_score']:>5}  {r['recommended_status']:<16}  {r['notes']}")

    # Score distribution
    from collections import Counter
    dist = Counter(r["replay_score"] for r in scored)
    status_dist = Counter(r["recommended_status"] for r in scored)
    print(f"\nScore distribution: {dict(sorted(dist.items(), reverse=True))}")
    print(f"Status distribution: {dict(status_dist)}")

    # Write CSV
    fieldnames = [
        "event_id","event_name","region_group","country","year",
        "trigger_type","flood_type","replay_score",
        "timeline_score","trigger_score","infrastructure_score",
        "intervention_score","source_score","recovery_score",
        "recommended_status","notes",
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in scored:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nWrote {OUT_CSV} ({len(scored)} rows)")

    # Write top-40 JSONL (original case + scores)
    top40 = scored[:40]
    with OUT_JSONL.open("w") as f:
        for r in top40:
            out = {k: r[k] for k in fieldnames}
            out["case_detail"] = r["_case"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_JSONL} ({len(top40)} entries)")


if __name__ == "__main__":
    main()

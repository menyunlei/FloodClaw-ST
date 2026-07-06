# FloodClaw-ST

**Leveraging LLM Test-Time Spatial Reasoning for Urban Flood Preparedness**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FloodClaw-ST is a case-based test-time reasoning framework for sequential urban flood preparedness. It studies how large language models can adapt a macro-level flood progression prior to a local event trajectory through stepwise prediction, independent evaluation, and feedback-grounded logic revision.

The project focuses on sequential flood-event reasoning: given the current flood state and previous context, predict the next key development such as physical flood response, infrastructure disruption, emergency intervention, or recovery outcome.

## Highlights

- **Macro-to-local reasoning:** derives reusable flood progression logic from historical cases, then adapts it at test time to a specific event sequence.
- **Two-agent protocol:** separates the predictor from an independent evaluator/judge to reduce self-confirming feedback.
- **Sequential benchmark:** includes Nebraska 2019 prompt/evaluation artifacts and a 622-transition benchmark with method predictions.
- **Ablation-ready prompts:** provides prompt templates for full FloodClaw-ST, no-logic, fixed-macro, summary-memory, self-revision, reflection, and evaluator-feedback variants.
- **Reproducibility artifacts:** includes final 623-event data and paper-style figures for hit rate, regional robustness, temporal robustness, calibration, critical difference, and cost-performance analysis.

## Repository Layout

```text
FloodClaw-ST-main/
├── data/
│   ├── Nebraska_Flood_2019_Key_Events__Processed_.csv
│   ├── Nebraska_Flood_2019_Key_Events_PromptOnly.csv
│   ├── Nebraska_Flood_2019_PromptOnly_Eval_*.csv
│   └── true_samples_622_transitions_with_predictions.csv
├── final_623_only/
│   ├── true_samples_623_events.csv
│   └── figures_623_only/
├── prompts/
│   ├── evaluate.txt
│   ├── predict_full_floodclaw*.txt
│   ├── predict_no_logic.txt
│   ├── predict_fixed_macro.txt
│   └── other ablation prompts
├── scripts/
│   ├── run_main_experiment.py
│   ├── run_parallel_evaluation.py
│   ├── run_logic_ablation.py
│   ├── run_sequential_analysis.py
│   ├── score_replay_candidates.py
│   └── which_logic_used.py
├── data.md
├── LICENSE
└── README.md
```

## Data Artifacts

| Path | Description |
| --- | --- |
| `data/Nebraska_Flood_2019_Key_Events__Processed_.csv` | Processed Nebraska 2019 flood event material and weather/context columns. |
| `data/Nebraska_Flood_2019_Key_Events_PromptOnly.csv` | Prompt-only Nebraska key-event dataset. |
| `data/Nebraska_Flood_2019_PromptOnly_Eval_ByEvent.csv` | Event-level prompt-only evaluation summary. |
| `data/Nebraska_Flood_2019_PromptOnly_Eval_DeepseekStrict.csv` | Strict evaluator results. |
| `data/Nebraska_Flood_2019_PromptOnly_Eval_LLMJudge.csv` | LLM-judge evaluation results. |
| `data/Nebraska_Flood_2019_PromptOnly_Eval_LLMJudge_FullLog.csv` | Full LLM-judge evaluation log. |
| `data/true_samples_622_transitions_with_predictions.csv` | 622 transition-level samples with method confidence and hit columns. |
| `final_623_only/true_samples_623_events.csv` | 623 final event records used by the final analysis package. |
| `final_623_only/figures_623_only/` | Generated paper figures and the critical-difference table. |

## Environment

This repository currently uses plain Python scripts. A minimal environment is:

```bash
cd /hpc2hdd/home/xhe989/Documents/2541/project2/DreamX-World-master/Wan-AI/Wan2.2-I2V-A14B/FloodClaw-ST-main

conda create -n floodclaw-st python=3.10 -y
conda activate floodclaw-st
pip install -r requirements.txt
```

The experiment scripts call OpenAI-compatible chat-completion endpoints through HTTP. Before running experiments, make sure your predictor and evaluator services are reachable from the machine where you run the scripts.

## Quick Start

### 1. Inspect the packaged benchmark artifacts

```bash
cd /hpc2hdd/home/xhe989/Documents/2541/project2/DreamX-World-master/Wan-AI/Wan2.2-I2V-A14B/FloodClaw-ST-main

wc -l data/*.csv final_623_only/*.csv
ls final_623_only/figures_623_only
```

### 2. Run the main sequential experiment

`scripts/run_main_experiment.py` compares a no-feedback baseline with a feedback-driven logic-memory setting, then runs backward reasoning over the learned logic memory.

The script expects a JSONL file containing ordered key events. Each row should include fields such as:

```json
{
  "event_id": 1,
  "category": "Nature Events",
  "trigger": "...",
  "response": "...",
  "infrastructure": "...",
  "intervention": "...",
  "outcome": "..."
}
```

Example command:

```bash
python scripts/run_main_experiment.py \
  --data-jsonl /path/to/Nebraska_Flood_2019_Key_Events.jsonl \
  --repeats 10 \
  --output-json results/main_experiment_results.json \
  --output-report results/main_experiment_report.txt \
  --qwen-base http://localhost:8072/v1 \
  --qwen-model /path/to/predictor-model \
  --deepseek-base http://localhost:27007/v1 \
  --deepseek-model /path/to/evaluator-model
```

Outputs:

- `results/main_experiment_results.json`: detailed per-repeat metrics and logic-memory tail.
- `results/main_experiment_report.txt`: readable summary report.

### 3. Run parallel transition evaluation

`scripts/run_parallel_evaluation.py` evaluates next-event predictions in parallel and reports hit rate, semantic score, precision, recall, F1, raw agreement, and Cohen's kappa.

```bash
python scripts/run_parallel_evaluation.py \
  --target-jsonl /path/to/logic_enriched_events.jsonl \
  --repeats 3 \
  --workers 128
```

The default endpoint constants are defined at the top of the script:

- `PREDICTOR_URL`
- `PREDICTOR_MODEL`
- `EVALUATOR_URL`
- `EVALUATOR_MODEL`

Edit these values or adapt the script before running on a different cluster or model server.

### 4. Generate sequential analysis tables

If raw prediction JSON files are available under `results/raw_predictions/`, run:

```bash
python scripts/run_sequential_analysis.py \
  --raw-dir results/raw_predictions \
  --suffix _final
```

This writes:

- `results/metrics/stepwise_trajectory_final.csv`
- `results/metrics/revision_effect_final.csv`
- `results/metrics/error_alignment_sample_final.csv`

## Prompt Templates

The `prompts/` directory contains reusable templates for the main method and ablations:

| Prompt | Purpose |
| --- | --- |
| `predict_full_floodclaw.txt` | Main local-adaptive FloodClaw-ST prediction format. |
| `predict_full_floodclaw_v1.txt`, `predict_full_floodclaw_v2.txt` | Earlier and more detailed full-method prompt variants. |
| `predict_no_logic.txt` | Baseline without explicit logic rules. |
| `predict_fixed_macro.txt` | Uses macro flood rules without event-specific revision. |
| `predict_summary_memory.txt` | Uses summarized prior steps as memory. |
| `predict_no_evaluator*.txt` | Self-revision variants without independent evaluator feedback. |
| `predict_eval_feedback.txt` | Prediction with evaluator feedback from previous step. |
| `predict_explicit_logic.txt` | Logic-only prediction without prior case context. |
| `predict_reflection_baseline.txt` | Structured reflection baseline. |
| `evaluate.txt` | Semantic evaluation template. |

## Script Status

| Script | Status | Notes |
| --- | --- | --- |
| `scripts/run_main_experiment.py` | Runnable with external JSONL data and reachable predictor/evaluator endpoints. | Uses only standard-library HTTP calls. |
| `scripts/run_parallel_evaluation.py` | Runnable with a compatible JSONL input and configured endpoints. | Requires `numpy` and `scikit-learn`. |
| `scripts/run_sequential_analysis.py` | Runnable when `results/raw_predictions/*.json` exists. | Produces CSV analysis tables. |
| `scripts/score_replay_candidates.py` | Utility script for ranking `data/prior_case_pool.jsonl`. | The required `prior_case_pool.jsonl` is not included in this snapshot. |
| `scripts/run_logic_ablation.py` | Experimental/legacy ablation runner. | References helper modules and config files not included in this snapshot (`loader.py`, `predictor.py`, `evaluator.py`, `metrics.py`, `configs/`). |
| `scripts/which_logic_used.py` | Documentation/debug helper. | Prints the historical 7-rule logic explanation and contains old absolute paths. |

## Method Summary

FloodClaw-ST treats flood reasoning as a sequential adaptation problem:

1. **Current state encoding:** represent the observed flood state using trigger, response, infrastructure, intervention, and outcome fields.
2. **Macro prior injection:** provide general flood progression logic as a weak prior.
3. **Local prediction:** predict the immediate next event while grounding the answer in the current case.
4. **Independent evaluation:** compare the prediction with the reference next event using a separate evaluator.
5. **Logic update:** store feedback as local logic memory for later steps.
6. **Trajectory-level analysis:** evaluate stepwise hit rate, semantic score, revision gain, and robustness across event groups.

## Figures

The final analysis figures are stored in `final_623_only/figures_623_only/`:

- `main_time_hitrate_with_ci.png`
- `main_delta_heatmap_vs_sc.png`
- `main_cost_performance_frontier.png`
- `app_region_hitrate_with_ci.png`
- `app_temporal_robustness_drop.png`
- `additional_calibration_curve.png`
- `additional_critical_difference_diagram.png`
- `additional_critical_difference_table.csv`

## Notes for Reproduction

- Many commands require external model endpoints; endpoint URLs and model paths should be updated for your environment.
- The packaged CSV files are sufficient for inspecting final artifacts, but some runnable scripts expect JSONL inputs that are not bundled in this snapshot.
- Create a `results/` directory before writing experiment outputs if it does not already exist:

```bash
mkdir -p results/metrics results/raw_predictions results/tables
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

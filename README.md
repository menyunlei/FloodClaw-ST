# FloodClaw-ST: Leveraging LLM Test-Time Spatial Reasoning for Urban Flood Preparedness


## Overview
FloodClaw-ST is a case-based test-time reasoning framework designed for sequential urban flood preparedness[cite: 1]. Traditional Large Language Models (LLMs) struggle with the dynamic, spatial, and sequential nature of disasters[cite: 1]. This framework reframes flood reasoning as an active adaptation process, where a macro-level flood logic prior is derived from historical cases and subsequently updated through evaluator-grounded feedback at the local, event-specific level[cite: 1]. 

## Key Contributions
* **Dynamic Logic Adaptation:** Employs a macro-to-local adaptation process that extracts reusable flood logic from a pool of 1,048 historical flood events[cite: 1].
* **Two-Agent Reasoning Protocol:** Utilizes a predictor and an independent evaluator to sequentially update the event-specific logic state via structured feedback (+1, 0, -1)[cite: 1].
* **Comprehensive Benchmark:** Introduces a large-scale sequential benchmark of 622 complete flood trajectories, where FloodClaw-ST achieves a 62.06% Hit Rate and 49.21% Semantic Score, outperforming representative reasoning baselines (SC, MoT, GoT, TTA)[cite: 1].
* **Cross-Regional & Cross-Lingual Portability:** Validated through a Nebraska-to-Asheville case probe and an evaluation of Arabic flood events to diagnose transferability across distinct hydrological mechanisms and linguistic contexts[cite: 1].

## Framework Architecture
1. **Macro-Scale Case Logic Derivation (The Prior):** Synthesizes knowledge from a macro pool of historical flood events to induce a general spatial and physical reasoning schema[cite: 1].
2. **Micro-Scale Test-Time Case Adaptation (Daily Loop):** Given a target event, the framework applies the prior and updates it step-by-step using daily observations and evaluator feedback, maintaining causal continuity across meteorological triggers, infrastructure states, interventions, and downstream impacts[cite: 1].

## Evaluation Settings
* **Backward Reasoning:** Retrodiction on the time-shifted 2019 Nebraska anchor case[cite: 1].
* **Forward Transfer:** Stepwise prediction on the unseen Asheville flash flood case[cite: 1].
* **Large-Scale Validation:** Sequential hidden-next-event prediction on 622 trajectories[cite: 1].
* **Cross-Lingual Portability:** Deployment on 100 Arabic flood cases utilizing localized models (e.g., Jais) to test factual and cultural grounding[cite: 1].




## 🚀 快速开始

### 1. 运行评估
```bash
cd /home/yunlei/flood
python compare_logic_variants.py
```

### 2. 查看结果
```bash
cat eval_with_logic_report.txt
cat eval_without_logic_report.txt
```

### 3. 修改规则
```bash
vim optimized_flood_logic_rules.jsonl
```

## 📁 重要文件

| 文件 | 说明 |
|------|------|
| `optimized_flood_logic_rules.jsonl` | 7条优化规则 |
| `run_logic_gain_eval.py` | 主评估脚本 |
| `compare_logic_variants.py` | 对比脚本 |
| `PROJECT_SUMMARY.md` | 项目完整总结 |
| `EVALUATION_IMPROVEMENTS.md` | 改进建议 |
| `QUICK_COMMAND_REFERENCE.py` | 快速命令参考 |

## 📊 规则流程

```
Nature Events (飓风/极端降雨)
    ↓ Rule 1
Natural Responses (河流快速上升)
    ↓ Rule 2
Infrastructure (基础设施受损)
    ↓ Rule 3
Interventions (紧急疏散)
    ↓ Rule 4 (循环)
Natural Responses (水位仍然很高)
    ↓ Rule 5
Infrastructure (服务系统故障)
    ↓ Rule 6
Interventions (应急响应)
    ↓ Rule 7
Trajectories (恢复和稳定)
```

## 💡 逻辑如何帮助LLM

**没有逻辑：** 当前事件 → [LLM猜测] → 预测 (准确率 ~26-30%)

**有逻辑：** 当前事件 + 规则 → [LLM应用规则] → 预测 (准确率 ~40-48%)

关键机制：
1. 搜索空间约束 - 减少离题预测
2. 因果链编码 - 明确的事件序列
3. 域知识注入 - 洪水特定的关键词
4. 推理可解释性 - 基于规则的推理

## 📚 文档导航

- **[项目总结](PROJECT_SUMMARY.md)** - 完整项目概览
- **[改进建议](EVALUATION_IMPROVEMENTS.md)** - 基线、指标、鲁棒性
- **[快速参考](QUICK_COMMAND_REFERENCE.py)** - 常用命令速查
- **[规则总结](RULES_STORAGE_SUMMARY.md)** - 规则存储方案

## 🔧 常用命令

```bash
# 运行完整对比
python compare_logic_variants.py

# 查看规则
cat optimized_flood_logic_rules.jsonl

# 编辑规则
vim optimized_flood_logic_rules.jsonl

# 分析规则
python analyze_rules.py

# 查看快速参考
python QUICK_COMMAND_REFERENCE.py
```

## 🌐 ReliefWeb 自动下载与 2000+ 样本构建

以下流程用于自动抓取 ReliefWeb 报告 PDF，并生成可用于后续 replay 升级的数据。脚本会基于 `download_manifest.tsv` 去重，重复运行不会重复下载同一 URL。

### 1. 自动下载到 2000+ PDF（建议目标 2200）

```bash
cd /home/yunlei/flood && /home/yunlei/miniconda3/envs/worldplay/bin/python - <<'PY'
import re, hashlib
from pathlib import Path
from urllib.parse import urljoin, quote_plus
import requests

base = 'https://reliefweb.int'
out_dir = Path('/home/yunlei/flood/report/reliefweb_auto_20260405')
out_dir.mkdir(parents=True, exist_ok=True)
manifest = out_dir / 'download_manifest.tsv'
headers = {'User-Agent': 'Mozilla/5.0'}

seen_pdf_urls = set()
if manifest.exists():
    for line in manifest.read_text(encoding='utf-8', errors='ignore').splitlines():
        parts = line.split('\t')
        if len(parts) >= 2:
            seen_pdf_urls.add(parts[1].strip())

existing_files = len(list(out_dir.glob('*.pdf')))
print('EXISTING_PDFS', existing_files)

queries = [
    'flood', 'floods', 'flash flood', 'river flood',
    'inundation', 'heavy rain flood', 'monsoon flood', 'storm surge flood'
]

report_paths = set()
for q in queries:
    qv = quote_plus(q)
    for page in range(0, 220):
        u = f'https://reliefweb.int/updates?search={qv}&page={page}'
        try:
            h = requests.get(u, headers=headers, timeout=25).text
        except Exception:
            continue
        hits = re.findall(r'/report/[^"\'\s>]+', h)
        if not hits and page > 12:
            break
        report_paths.update(hits)

report_urls = [urljoin(base, p) for p in sorted(report_paths)]
print('REPORT_URLS_FOUND', len(report_urls))

pdf_urls = []
seen_batch = set()
for i, ru in enumerate(report_urls, start=1):
    try:
        t = requests.get(ru, headers=headers, timeout=25).text
    except Exception:
        continue
    links = set(re.findall(r'https?://[^"\'\s>]+\.pdf(?:\?[^"\'\s>]*)?', t, re.I))
    rel = set(re.findall(r'"(/[^"\']+\.pdf(?:\?[^"\']*)?)"', t, re.I))
    links.update(urljoin(base, x) for x in rel)
    for l in links:
        if l in seen_pdf_urls or l in seen_batch:
            continue
        seen_batch.add(l)
        pdf_urls.append((ru, l))
    if i % 500 == 0:
        print('SCANNED', i, 'NEW_CANDIDATES', len(pdf_urls))

print('TOTAL_NEW_PDF_CANDIDATES', len(pdf_urls))

TARGET_TOTAL = 2200
need = max(0, TARGET_TOTAL - existing_files)
print('NEED_TO_DOWNLOAD', need)

ok = 0
skip = 0
err = 0
for src, pdf in pdf_urls:
    if ok >= need:
        break
    try:
        r = requests.get(pdf, headers=headers, timeout=50)
        ct = (r.headers.get('content-type') or '').lower()
        if r.status_code != 200:
            skip += 1
            continue
        if ('pdf' not in ct) and (not pdf.lower().endswith('.pdf')):
            skip += 1
            continue
        name = pdf.split('/')[-1].split('?')[0]
        if not name.lower().endswith('.pdf'):
            name += '.pdf'
        name = name[:140]
        h = hashlib.sha1(pdf.encode()).hexdigest()[:8]
        fpath = out_dir / f'{h}_{name}'
        fpath.write_bytes(r.content)
        with manifest.open('a', encoding='utf-8') as mf:
            mf.write(f'{fpath.name}\t{pdf}\t{src}\n')
        ok += 1
        if ok % 50 == 0:
            print('DOWNLOADED', ok)
    except Exception:
        err += 1

print('DOWNLOAD_DONE')
print('DOWNLOADED_OK', ok)
print('SKIPPED', skip)
print('ERRORS', err)
print('TOTAL_PDFS_NOW', len(list(out_dir.glob('*.pdf'))))
print('OUT_DIR', out_dir)
PY
```

### 2. 解析为 2000+ JSONL 样本

```bash
cd /home/yunlei/flood
/home/yunlei/miniconda3/envs/worldplay/bin/python pipelines/ingest_local_reports_to_rag_jsonl.py \
  --input-dir /home/yunlei/flood/report \
  --output-doc-jsonl data/local_report_docs_2000plus.jsonl \
  --output-replay-seed data/replay_from_local_reports_seed_2000plus.jsonl \
  --min-chars 600

wc -l data/local_report_docs_2000plus.jsonl data/replay_from_local_reports_seed_2000plus.jsonl
```

### 3. 可选：升级为 multi-step + logic-rich 数据

```bash
cd /home/yunlei/flood
PYTHONUNBUFFERED=1 /home/yunlei/miniconda3/envs/worldplay/bin/python -u pipelines/upgrade_replay_rag_dataset.py \
  --input data/replay_from_local_reports_seed_2000plus.jsonl \
  --output data/replay_from_local_reports_upgraded_2000plus.jsonl \
  --failed-output data/replay_from_local_reports_upgraded_2000plus_failed.jsonl \
  --llm-url http://10.120.20.222:27007/v1/chat/completions \
  --llm-model /hpc2hdd/home/xhe989/Documents/ExGRPO/qwen/model/flood/Llama-3.3-70B-Instruct \
  --timeout 120 --min-steps 3 --max-steps 4 --rules-per-event 3 --max-retries 2 --workers 32
```

### 4. 产物位置

- PDF 下载目录: `report/reliefweb_auto_20260405/`
- 下载映射清单: `report/reliefweb_auto_20260405/download_manifest.tsv`
- 文档级 JSONL: `data/local_report_docs_2000plus.jsonl`
- Replay seed JSONL: `data/replay_from_local_reports_seed_2000plus.jsonl`
- 升级后 JSONL: `data/replay_from_local_reports_upgraded_2000plus.jsonl`

### 5. 公开发布前注意

- 建议仅公开代码与处理流程，不直接公开大体量原始 PDF（体积大且可能涉及来源站点条款）。
- 保留 `download_manifest.tsv`，用于复现实验与来源追踪。
- 对外发布数据时附带来源与时间戳说明。

## 📈 性能指标

### Without Logic 数据集
- no_logic: 25.93% hit rate
- learned_logic: 48.15% hit rate ⬆️ +30%
- 改进: +85.71%

### With Logic 数据集
- no_logic: 33.33% hit rate
- learned_logic: 37.04% hit rate
- 改进: +11.11%

## 🎓 改进建议

查看 `EVALUATION_IMPROVEMENTS.md` 了解：
- 基线对比 (RAG, Reflexion, Test-time training, CoT)
- 时间序列指标 (Kendall Tau, Sequence F1, Edit Distance)
- 校准指标 (ECE, MCE, Brier Score)
- 鲁棒性分析 (Prompt variation, Adversarial perturbations, OOD)

## ❓ 常见问题

**Q: 评估需要多长时间?**
A: 3次重复约15-30分钟

**Q: 如何修改规则?**
A: 直接编辑 `optimized_flood_logic_rules.jsonl`

**Q: 性能波动很大怎么办?**
A: 增加重复次数 (`--repeats 5` 或更多)

**Q: 如何使用自定义数据集?**
A: 修改 `--target-jsonl` 参数

## 📞 支持

如有问题，请查看：
1. `PROJECT_SUMMARY.md` - 完整项目信息
2. `EVALUATION_IMPROVEMENTS.md` - 改进建议
3. `QUICK_COMMAND_REFERENCE.py` - 命令参考

---

**项目状态：** ✅ 完成并可用
**最后更新：** 2026-03-20

## Citation
If you find this framework or dataset helpful for your research, please consider citing:
```bibtex
@article{men2026floodclaw,
  title={FloodClaw-ST: Leveraging LLM Test-Time Spatial Reasoning for Urban Flood Preparedness},
  author={Men, Yunlei and He, Xiaoli and Mohamed, Wiam Osman Hassan and Liu, Xiaoli and KAN, Ge Lin and Li, Hao},
  journal={GIScience & Remote Sensing},
  year={2026}
}

# -*- coding: utf-8 -*-
"""把 data/runs_v2/<组>/*.json 汇总成 docs/实验记录-v2.md（v2 判定口径）。

用法：
    python experiments/build_v2_table.py

口径（与 docs/反思判定状态机-改动清单.md 第 6 节一致）：
- 合格率 = verdict == 通过 / 样本数（降级通过不计入分子）
- 降级率 = verdict == 降级通过 / 样本数（缺资料 ≠ 失败，单独成列）
- 否决率 = 其余终态 / 样本数
- 空转样本 = replan_count > 0 且终态 != 通过 的运行数
v1 的 20 份记录在 data/runs/，判定语义不同，**本脚本不读、也不与之混表**。
"""
import glob
import json
import os
import statistics
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "data", "runs_v2")
OUT_PATH = os.path.join(ROOT, "docs", "实验记录-v2.md")

GROUP_DESC = {
    "E1": "基线：ENABLE_REFLECTION=0（无反思）",
    "E2": "纯 LLM 反思：DISABLE_CONTENT_GATE=1 + 关覆盖率/重复规则",
    "E3": "本文方法：全默认",
    "E4": "消融·覆盖率：ENABLE_COVERAGE_RULE=0",
    "E5": "消融·重复检测：ENABLE_REPEAT_RULE=0",
    "E6": "故障注入：INJECT_LLM_FAIL=3",
}


def _load() -> dict:
    """{组: [record, ...]}；组名取 JSON 的 group 字段，缺失时用目录名。"""
    out = {}
    for path in sorted(glob.glob(os.path.join(RUN_DIR, "*", "*.json"))):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        group = rec.get("group") or os.path.basename(os.path.dirname(path))
        rec["_path"] = os.path.relpath(path, ROOT)
        out.setdefault(group, []).append(rec)
    return out


def _verdict(rec: dict) -> str:
    return (rec.get("reflection") or {}).get("verdict") or "（未记录）"


def _rate(part: int, total: int) -> str:
    return f"{part / total:.0%}" if total else "-"


def _avg(values) -> str:
    values = [v for v in values if isinstance(v, (int, float))]
    return f"{statistics.mean(values):.1f}" if values else "-"


def _summarize(recs) -> dict:
    n = len(recs)
    verdicts = [_verdict(r) for r in recs]
    strict = sum(1 for v in verdicts if v == "通过")
    degraded = sum(1 for v in verdicts if v == "降级通过")
    replans = [r.get("replan_count", 0) or 0 for r in recs]
    idle = sum(1 for r, v in zip(recs, verdicts)
               if (r.get("replan_count", 0) or 0) > 0 and v != "通过")
    retries = sum(1 for r in recs if (r.get("search") or {}).get("retry_query"))
    return {
        "n": n,
        "strict": strict,
        "degraded": degraded,
        "rejected": n - strict - degraded,
        "replan_avg": _avg(replans),
        "idle": idle,
        "retry": retries,
        "elapsed": _avg([r.get("elapsed_seconds") for r in recs]),
        "llm_calls": _avg([(r.get("reflection") or {}).get("llm_calls") for r in recs]),
        "cited": _avg([(r.get("reflection") or {}).get("cited_count") for r in recs]),
    }



def _summary_table(data: dict) -> str:
    lines = [
        "| 组 | 配置 | 样本 | 合格率（通过） | 降级率 | 否决率 | 平均重规划 | 空转样本 | 补检索 | 平均耗时(s) | 平均 LLM 调用 | 平均引用数 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group in sorted(GROUP_DESC):
        recs = data.get(group) or []
        if not recs:
            lines.append(f"| {group} | {GROUP_DESC[group]} | 0 | 待跑 | - | - | - | - | - | - | - | - |")
            continue
        s = _summarize(recs)
        lines.append(
            f"| {group} | {GROUP_DESC[group]} | {s['n']} | "
            f"{s['strict']}（{_rate(s['strict'], s['n'])}） | "
            f"{s['degraded']}（{_rate(s['degraded'], s['n'])}） | "
            f"{s['rejected']}（{_rate(s['rejected'], s['n'])}） | "
            f"{s['replan_avg']} | {s['idle']} | {s['retry']} | "
            f"{s['elapsed']} | {s['llm_calls']} | {s['cited']} |")
    return "\n".join(lines)


def _detail_table(data: dict) -> str:
    lines = [
        "| 组 | 主题 | 终态 | 判定来源 | pass_kind | 证据/引用 | replan | 补检索式 | 耗时(s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group in sorted(data):
        for rec in sorted(data[group], key=lambda r: r.get("ts") or ""):
            refl = rec.get("reflection") or {}
            search = rec.get("search") or {}
            cited = refl.get("cited_count")
            ratio = refl.get("citation_ratio")
            cite_str = f"{refl.get('evidence_count', 0)}/{cited if cited is not None else '-'}"
            if isinstance(ratio, (int, float)):
                cite_str += f"（{ratio:.0%}）"
            retry = search.get("retry_query") or "-"
            if len(retry) > 40:
                retry = retry[:40] + "…"
            lines.append(
                f"| {group} | {rec.get('topic', '')} | {_verdict(rec)} | "
                f"{refl.get('decision_source') or refl.get('gated_by') or '-'} | "
                f"{refl.get('pass_kind') or '-'} | {cite_str} | "
                f"{rec.get('replan_count', 0)} | {retry} | "
                f"{rec.get('elapsed_seconds', '-')} |")
    return "\n".join(lines)


def _config_note(data: dict) -> str:
    seen = {}
    for group, recs in data.items():
        for rec in recs:
            cfg = rec.get("config") or {}
            if cfg:
                seen[group] = cfg
                break
    if not seen:
        return "- （尚无记录，跑完 E1~E6 后重跑本脚本即可）"
    lines = []
    for group in sorted(seen):
        cfg = seen[group]
        lines.append(
            f"- **{group}**：`enable_reflection={cfg.get('enable_reflection')}`、"
            f"`enable_coverage_rule={cfg.get('enable_coverage_rule')}`、"
            f"`enable_repeat_rule={cfg.get('enable_repeat_rule')}`、"
            f"`disable_content_gate={cfg.get('disable_content_gate')}`、"
            f"`inject_llm_fail={cfg.get('inject_llm_fail')}`、"
            f"`pass_min_cited={cfg.get('pass_min_cited')}`、"
            f"`pass_min_chars={cfg.get('pass_min_chars')}`、"
            f"`evidence_top_k={cfg.get('evidence_top_k')}`、"
            f"`retry_query_mode={cfg.get('retry_query_mode')}`、"
            f"`generator={cfg.get('generator_model')}` / `critic={cfg.get('critic_model')}`")
    return "\n".join(lines)



def build() -> str:
    data = _load()
    total = sum(len(v) for v in data.values())
    parts = [
        "# 实验记录 v2（E1~E6）",
        "",
        f"> **自动生成**：`python experiments/build_v2_table.py`（{datetime.now():%Y-%m-%d %H:%M}）",
        f"> **数据源**：`data/runs_v2/<组>/*.json`，共 **{total}** 份；判定语义 = 反思判定状态机 v2（统一门控 + P1 规则）。",
        "> **不可混表**：`data/runs/` 里的 20 份是 **v1 口径**（无 `cited_count`、无 `repeat_hit`），"
        "只作历史对照，任何表格都不得与本表合并统计。",
        "",
        "## 1. 指标汇总",
        "",
        _summary_table(data),
        "",
        "> 口径：合格率只数 `verdict == 通过`；`降级通过`（可读但无可核验引用）单独成列；"
        "空转样本 = `replan_count > 0` 且终态未回到 `通过`，用来量化「重规划没救回来」。",
        "",
        "## 2. 逐条明细",
        "",
        _detail_table(data),
        "",
        "## 3. 各组实际配置（读自 JSON 的 config 段）",
        "",
        _config_note(data),
        "",
        "## 4. 复现命令",
        "",
        "```powershell",
        "# 全部 6 组 × 默认 4 主题（结果写入 data/runs_v2/<组>/）",
        "python experiments/run_matrix.py",
        "",
        "# 单组 / 单主题",
        "python experiments/run_matrix.py --groups E3",
        "python experiments/run_matrix.py --groups E3 --topics \"生物细胞的结构\"",
        "",
        "# 重新生成本表",
        "python experiments/build_v2_table.py",
        "```",
        "",
    ]
    text = "\n".join(parts)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"已生成 {os.path.relpath(OUT_PATH, ROOT)}（{total} 份记录）")
    return text


if __name__ == "__main__":
    build()

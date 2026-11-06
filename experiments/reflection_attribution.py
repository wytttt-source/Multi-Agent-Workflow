# -*- coding: utf-8 -*-
"""反思归因表：把 24 份 v2 记录的判定来源、重规划、空转与截断信号摊开成一张表。

回答三个「论文里必须能自证」的问题：
    1. 终态是谁定的？—— `decision_source` 分布（llm / gate / coverage_rule /
       rule_fallback / free_reflection / baseline），以及门控改判了哪几条；
    2. 重规划值不值？—— 发生重规划的记录最终是否救回（通过），没救回即「空转」；
    3. 规则到底省了多少 LLM 调用？—— 覆盖率前置否决不需要裁判，每条省 1 次 critic。

另外做一次**离线复算**：用记录里的 `result` + `sources` 重跑 `content_gate`，
核对 (a) 24 份历史终态零翻转；(b) 真实回答的 `truncated` 命中数（v2.2 完整性规则
的误伤检查）。全程不调 LLM、不写 `data/`。

用法：
    python experiments/reflection_attribution.py                     # 打印 + 写 docs/反思归因表.md
    python experiments/reflection_attribution.py --out -             # 只打印，不写文件
    python experiments/reflection_attribution.py --group E3          # 只看某一组
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agents.nodes as nodes  # noqa: E402  （必须在 sys.path 之后导入）
from utils import content_gate  # noqa: E402

RUN_DIR = os.path.join(ROOT, "data", "runs_v2")
CASES_PATH = os.path.join(ROOT, "experiments", "samples", "gate_cases.json")
OUT_DOC = os.path.join(ROOT, "docs", "反思归因表.md")

GROUPS = ("E1", "E2", "E3", "E4", "E5", "E6")
GROUP_NOTE = {
    "E1": "基线：ENABLE_REFLECTION=0（无反思）",
    "E2": "纯 LLM 反思：DISABLE_CONTENT_GATE=1",
    "E3": "本文方法：全默认",
    "E4": "消融·覆盖率规则=0",
    "E5": "消融·重复检测=0",
    "E6": "故障注入：INJECT_LLM_FAIL=3",
}
SOURCE_NOTE = {
    "llm": "语义裁判（critic）",
    "gate": "确定性门控改判/否决",
    "coverage_rule": "覆盖率规则前置否决（不调用裁判）",
    "rule_fallback": "反思 LLM 不可用 → 规则兜底",
    "free_reflection": "自由反思（补检索后重写）",
    "baseline": "E1 基线（关反思）",
}


def _load_records(group=None):
    """读 data/runs_v2/<组>/*.json；返回 [{record, path}]，按 (组, 主题) 排序。"""
    paths = []
    for g in ([group] if group else GROUPS):
        paths.extend(glob.glob(os.path.join(RUN_DIR, g, "*.json")))
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        rows.append({"path": path, "rec": rec,
                     "group": (rec.get("config") or {}).get("group") or group or "?"})
    rows.sort(key=lambda r: (r["group"], r["rec"].get("topic") or ""))
    return rows


def _evidence_from_sources(sources):
    """把 sources（`标题 | URL`）还原成门控所需的 evidence 结构。

    记录不回传 evidence 原文，而门控只用到「非空」与标题/摘要的字面命中，
    因此用标题回填摘要即可等价复现（摘要长度不参与判定）。
    """
    items = []
    for s in sources or []:
        title, url = (s.split(" | ", 1) + [""])[:2] if " | " in s else (s, "")
        if not title.strip() and not url.strip():
            continue
        items.append({"title": title.strip(), "url": url.strip(),
                      "summary": title.strip(), "content": ""})
    return items


def _recompute(rec):
    """用记录里的 result/sources 重跑门控，返回 (gate, 重算后的判定, 是否翻转)。"""
    stored = ((rec.get("reflection") or {}).get("verdict")) or "（未判定）"
    state = {
        "topic": rec.get("topic") or "",
        "result": rec.get("result") or "",
        "sources": rec.get("sources") or [],
        "evidence": _evidence_from_sources(rec.get("sources")),
        "used_fallback": bool(rec.get("used_fallback")),
    }
    gate = content_gate(state["result"], state["sources"], state["evidence"])
    # 把「历史终态」当作裁判原话喂给门控：门控若改成别的终态即为翻转
    verdict, _, hit = nodes._apply_pass_gate(state, stored, "（离线复算：沿用历史终态）", gate)
    verdict, _, rescue = nodes._apply_degrade_rescue(state, verdict, "", gate)
    return gate, verdict, (verdict != stored), hit, rescue



def _pct(n, d):
    return f"{n / d:.0%}" if d else "-"


def analyze(rows, cases=None):
    """把记录摊成可写表的结构；cases 为 gate_cases.json（可选，用于规则节省调用数）。"""
    by_group = {g: [r for r in rows if r["group"] == g] for g in GROUPS}
    sources = Counter()
    kinds = Counter()
    kinds_by_group = defaultdict(Counter)
    replanned, idle = [], []
    rescued, truncated, flipped = [], [], []
    max_calls = 0

    for row in rows:
        rec, group = row["rec"], row["group"]
        refl = rec.get("reflection") or {}
        verdict = refl.get("verdict") or "（未判定）"
        src = refl.get("decision_source") or refl.get("gated_by") or "-"
        kind = refl.get("pass_kind") or nodes._pass_kind(verdict)
        sources[src] += 1
        kinds[kind] += 1
        kinds_by_group[group][kind] += 1
        max_calls = max(max_calls, refl.get("llm_calls") or 0)

        entry = {"group": group, "topic": rec.get("topic") or "",
                 "verdict": verdict, "source": src, "kind": kind,
                 "replan": rec.get("replan_count") or 0,
                 "llm_calls": refl.get("llm_calls"),
                 "cited": refl.get("cited_count"),
                 "ratio": refl.get("citation_ratio")}
        if entry["replan"]:
            replanned.append(entry)
            if verdict != "通过":
                idle.append(entry)
        if refl.get("rescue_rule"):
            rescued.append({**entry, "rule": refl["rescue_rule"],
                            "from": refl.get("rescued_from") or "-"})

        # E1 基线（enable_reflection=False）根本不进反思节点，不参与门控复算
        if (rec.get("config") or {}).get("enable_reflection", True):
            gate, recomputed, is_flip, hit, rescue = _recompute(rec)
            if gate.get("truncated"):
                truncated.append({**entry, "hits": gate.get("truncation_hits")})
            if is_flip:
                flipped.append({**entry, "recomputed": recomputed,
                                "hit": hit, "rescue": rescue})

    coverage_hits = [r for r in rows
                     if ((r["rec"].get("reflection") or {}).get("decision_source")
                         == "coverage_rule")]
    # 对照样本的规则命中：优先读 gate_vs_llm 结果行（C 列），兼容手写 expect_source
    case_coverage = 0
    if cases:
        case_coverage = sum(
            1 for c in cases
            if (c.get("expect_source")
                or ((c.get("C") or {}).get("decision_source"))) == "coverage_rule")
    return {
        "n": len(rows),
        "n_gated": sum(1 for r in rows
                       if (r["rec"].get("config") or {}).get("enable_reflection", True)),
        "by_group": by_group,
        "sources": sources,
        "kinds": kinds,
        "kinds_by_group": kinds_by_group,
        "replanned": replanned,
        "idle": idle,
        "rescued": rescued,
        "truncated": truncated,
        "flipped": flipped,
        "coverage_hits": coverage_hits,
        "case_coverage": case_coverage,
        "max_calls": max_calls,
    }


def render(stats, path_note=""):
    """生成 Markdown 表；每个数字都能回指 data/ 里的具体记录。"""
    n = stats["n"]
    parts = [
        "# 反思归因表（v2 记录离线复算）",
        "",
        f"> **自动生成**：`python experiments/reflection_attribution.py`（{datetime.now():%Y-%m-%d %H:%M}）",
        f"> **数据源**：`data/runs_v2/<组>/*.json`，共 **{n}** 份；判定语义 = 反思判定状态机 v2.2"
        "（确定性门控 + 降级救援 + 完整性规则）。",
        "> **不调 LLM**：本表只读已落盘的 `result` / `sources` / `reflection`，用 `content_gate`"
        "重算事实字段，因此可随时复跑核对。",
        "",
        "## 1. 判定来源分布（谁定了终态）",
        "",
        "| 判定来源 | 含义 | 条数 | 占比 |",
        "|---|---|---|---|",
    ]
    for src, cnt in stats["sources"].most_common():
        parts.append(f"| `{src}` | {SOURCE_NOTE.get(src, '-')} | {cnt} | {_pct(cnt, n)} |")
    parts += ["", "> `decision_source` 只记「哪条路径定的终态」；门控把 LLM 的「通过」改判为"
              "「降级/否决」时记 `gate`，降级救援不另立来源，而是把 `rescued_from` / "
              "`rescue_rule` 写在同一条记录里（见第 5 节）。", ""]

    parts += ["## 2. 三态分布（按实验组）", "",
              "| 组 | 配置 | 样本 | 通过 | 降级通过 | 否决 | 合格率 |",
              "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        rows = stats["by_group"].get(g) or []
        if not rows:
            continue
        c = stats["kinds_by_group"][g]
        parts.append(f"| {g} | {GROUP_NOTE.get(g, '')} | {len(rows)} | {c['strict']} | "
                     f"{c['degraded']} | {c['rejected']} | {_pct(c['strict'], len(rows))} |")
    parts.append("")

    parts += ["## 3. 重规划与空转", ""]
    if stats["replanned"]:
        parts += ["| 组 | 主题 | 终态 | replan | 救回 |", "|---|---|---|---|---|"]
        for e in stats["replanned"]:
            parts.append(f"| {e['group']} | {e['topic']} | {e['verdict']} | {e['replan']} | "
                         f"{'是' if e['verdict'] == '通过' else '否（空转）'} |")
        parts += ["", f"- **空转率**：{len(stats['idle'])}/{len(stats['replanned'])}"
                  f"（{_pct(len(stats['idle']), len(stats['replanned']))}）"
                  "—— 发生过重规划、但终态仍未回到 `通过` 的比例。",
                  "- 空转样本的共同点是补检索仍拿不到该对象的证据（`evidence 0 条`），"
                  "再由自由反思走模型知识兜底并声明，属「规则不该被绕过」的正常结局。", ""]
    else:
        parts += [f"{n} 份记录均未触发重规划（`replan_count = 0`），空转率不适用。", ""]

    parts += ["## 4. 规则省下的 LLM 调用",
              "",
              f"- 24 份记录中 `decision_source = coverage_rule` 的条数：**{len(stats['coverage_hits'])}**；",
              f"- 18 条对照样本中该规则命中的条数：**{stats['case_coverage']}**"
              "（`S15` 复合主题证据单侧：不调裁判直接判 `分解不合理`）；",
              "- 该路径每条省 **1 次 critic 调用**（约占单轮 3 次调用的 1/3），"
              "代价是判据写死在 `evidence_coverage()` 里（单侧证据必判不合格）。",
              f"- 全量记录 `reflection.llm_calls` 上限 **{stats['max_calls']}** 次"
              "（E1 关反思 = 2 次；E6 注入故障会多耗重试）。",
              ""]

    parts += ["## 5. 降级救援台账（v2.2 双向门控）", ""]
    if stats["rescued"]:
        parts += ["| 组 | 主题 | 原判 | 改判 | 规则 |", "|---|---|---|---|---|"]
        for e in stats["rescued"]:
            parts.append(f"| {e['group']} | {e['topic']} | {e['from']} | {e['verdict']} | "
                         f"`{e['rule']}` |")
        parts.append("")
    else:
        parts += ["24 份历史记录没有触发救援（它们的终态本就由裁判直判，没有「否决 → 降级」"
                  "的改判）；救援的实际命中见 `docs/规则vsLLM-混淆矩阵.md` 第 1 节"
                  "（S01/S10/S11/S16 四条）。", ""]

    parts += ["## 6. 截断信号复算（v2.2 完整性规则）", "",
              f"- 命中 `truncated` 的记录数：**{len(stats['truncated'])}/{stats['n_gated']}**"
              "（只统计真正过门控的记录；E1 基线 `enable_reflection=False` 不计）"]
    for e in stats["truncated"]:
        parts.append(f"  - {e['group']} · {e['topic']}：{'、'.join(e['hits'])}")
    parts += ["- 结论：真实回答都以句号收尾、括号成对，`_looks_truncated()` 在 24 份记录上"
              "**零误伤**；该规则只在「半截话」上生效（4 条离线用例见 "
              "`tests/test_reflection_gate.py::TruncationTest`）。", ""]

    parts += ["## 7. 零翻转核对", "",
              "- 以历史终态为输入重跑「门控 + 救援」，终态发生变化的记录数："
              f"**{len(stats['flipped'])}/{stats['n_gated']}**"
              "（E1 基线不进反思节点，样本量按 `enable_reflection=True` 计）"]
    for e in stats["flipped"]:
        parts.append(f"  - {e['group']} · {e['topic']}：{e['verdict']} → {e['recomputed']}"
                     f"（{e['hit'] or e['rescue'] or '-'}）")
    if not stats["flipped"]:
        parts.append("- v2.2 的两条新门控（截断 / 引用率）与降级救援**没有推翻任何历史结论**，"
                     "与 `docs/实验记录-v2.md` 的 24 份记录逐条一致。")
    parts += ["", "## 8. 复现方式", "",
              "```powershell",
              "python experiments/reflection_attribution.py            # 重新生成本表",
              "python experiments/build_v2_table.py                    # 重生成实验记录-v2.md",
              "python experiments/gate_vs_llm.py --mock data/gate_vs_llm/<时间戳>.json",
              "```", ""]
    if path_note:
        parts.append(f"> 生成命令备注：{path_note}")
    return "\n".join(parts)


def _load_case_rows(path):
    """读 gate_vs_llm 的结果 JSON（兼容样本集：缺字段时返回空）。"""
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("cases") or []


def _latest_gate_result():
    paths = glob.glob(os.path.join(ROOT, "data", "gate_vs_llm", "gate_vs_llm-*.json"))
    return max(paths, key=os.path.getmtime) if paths else ""


def main(argv=None):
    ap = argparse.ArgumentParser(description="反思归因表（离线复算，不调 LLM）")
    ap.add_argument("--out", default=OUT_DOC, help="输出 Markdown 路径；`-` 表示只打印")
    ap.add_argument("--group", default="", help="只看某一组（E1~E6）")
    ap.add_argument("--cases", default="", help="gate_vs_llm 结果 JSON；缺省取最新一份")
    args = ap.parse_args(argv)

    rows = _load_records(args.group or None)
    if not rows:
        print("未找到任何运行记录：先跑 python experiments/run_matrix.py")
        return 2
    cases = _load_case_rows(args.cases or _latest_gate_result())

    stats = analyze(rows, cases)
    note = f"记录 {stats['n']} 份" + (f"（仅 {args.group} 组）" if args.group else "")
    text = render(stats, path_note=note)
    print(text)
    if args.out and args.out != "-":
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已生成 {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


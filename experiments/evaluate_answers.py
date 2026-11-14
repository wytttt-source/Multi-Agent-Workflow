# -*- coding: utf-8 -*-
"""v2.3 修复路径四臂汇总：把 data/runs_v3/P-*/ 的 JSON 变成 docs/修复路径对照.md。

用法：
    python experiments/evaluate_answers.py                      # 读 data/runs_v3，写 docs/修复路径对照.md
    python experiments/evaluate_answers.py --stdout              # 只打印不落盘
    python experiments/evaluate_answers.py --dir data/runs_v2 --stdout
                                                                 # 用旧记录验证判卷口径（应零翻转）

判卷与系统判定**相互独立**：判卷只用「任务给定的必答对象 + 可核验引用 + 字数/截断」，
不看反思节点的 verdict。两者不一致即记一次误判，这是"判得对不对"的唯一共同标准。

四张表：
    表 1  每臂汇总（修复率 / 无效重试率 / 成本 / 三态 / 误判）
    表 2  错误类型 × 臂（核心：哪种错该走哪条路）
    表 3  三态细化（降级拆分 + 兜底 / 救援命中）
    表 4  逐题配对（同一道题在四臂下的实际结果）
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import content_gate, answer_completeness   # noqa: E402

RUN_DIR = os.path.join(ROOT, "data", "runs_v3")
TASKS_PATH = os.path.join(ROOT, "experiments", "tasks_repair.json")
OUT_PATH = os.path.join(ROOT, "docs", "修复路径对照.md")

ARMS = ["B0", "F", "R", "H"]
ARM_LABEL = {"B0": "B0 无反思", "F": "F 固定类型", "R": "R 自由反思", "H": "H 混合（本文）"}
KIND_CN = {"pass": "通过", "degraded": "降级", "rejected": "否决"}
# 系统三态与判卷三态用同一套枚举，才能逐条比"判得对不对"
SYSTEM_KIND = {"通过": "pass", "降级通过": "degraded"}


def load_tasks(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]
    return {t["topic"]: t for t in tasks}


def load_records(run_dir: str) -> list:
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"[跳过] 解析失败：{os.path.relpath(path, ROOT)}")
            continue
        group = (rec.get("group") or "").strip()
        arm = group[2:] if group.startswith("P-") else group
        rows.append({"path": path, "arm": arm, "rec": rec})
    return rows


def system_kind(rec: dict) -> str:
    """系统终态：优先读 repair.final_verdict（v2.3 轨迹），退回 reflection.verdict（v2 记录）。"""
    repair = rec.get("repair") or {}
    verdict = repair.get("final_verdict") or (rec.get("reflection") or {}).get("verdict") or ""
    return SYSTEM_KIND.get(verdict, "rejected")


def judge(rec: dict, task: dict) -> dict:
    """独立判卷（不看系统 verdict），返回 rubric 三态 + 依据。"""
    result = rec.get("result") or ""
    gate = content_gate(result, rec.get("sources") or [], rec.get("evidence") or [])
    comp = answer_completeness(result, (task or {}).get("required_objects") or [])
    requires_citation = (task or {}).get("requires_citation", True)

    hard = []
    if not gate["len_ok"]:
        hard.append("正文过短")
    if gate["truncated"]:
        hard.append("截断")
    if not gate["no_fabricated_url"]:
        hard.append("不可核验URL")
    if comp["missing"]:
        hard.append("缺必答对象:" + "/".join(comp["missing"]))
    if hard:
        return {"rubric": "rejected", "why": hard, "gate": gate, "comp": comp}
    if requires_citation and gate["cited_count"] == 0:
        return {"rubric": "degraded", "why": ["可读但无引用"], "gate": gate, "comp": comp}
    return {"rubric": "pass", "why": [], "gate": gate, "comp": comp}


def mismatch_of(sys_kind: str, rubric: str) -> str:
    """误判分类：空串表示一致。误放行 / 误否决最严重，粒度错位只影响三态统计。"""
    if sys_kind == rubric:
        return ""
    if sys_kind == "pass":
        return "误放行"
    if sys_kind == "rejected" and rubric == "pass":
        return "误否决"
    return "粒度错位"


def build_rows(run_dir: str, tasks: dict) -> list:
    rows = []
    for item in load_records(run_dir):
        rec = item["rec"]
        task = tasks.get(rec.get("topic") or "", {})
        verdict = judge(rec, task)
        repair = rec.get("repair") or {}
        refl = rec.get("reflection") or {}
        s_kind = system_kind(rec)
        rows.append({
            "arm": item["arm"],
            "task_id": task.get("id") or "-",
            "group": task.get("group") or "-",
            "topic": rec.get("topic") or "",
            "inject": (task.get("inject") or (rec.get("config") or {}).get("inject_fault") or "-"),
            "outcome": repair.get("outcome") or "unknown",
            "rounds": len(repair.get("rounds") or []),
            "paths": [r.get("path") or "-" for r in (repair.get("rounds") or [])],
            "replan": rec.get("replan_count", 0),
            "calls": refl.get("llm_calls") or (repair.get("cost") or {}).get("llm_calls") or 0,
            "elapsed": rec.get("elapsed_seconds") or 0,
            "system": s_kind,
            "rubric": verdict["rubric"],
            "why": verdict["why"],
            "cited": verdict["gate"]["cited_count"],
            "fallback": bool(rec.get("used_fallback")),
            "rescued": bool(refl.get("rescued_from")),
            "mismatch": mismatch_of(s_kind, verdict["rubric"]),
            "injected": rec.get("injected") or [],
        })
    return rows


def _pct(num: int, den: int) -> str:
    return f"{num / den:.0%}" if den else "-"


def _avg(values: list) -> str:
    vals = [v for v in values if isinstance(v, (int, float))]
    return f"{sum(vals) / len(vals):.1f}" if vals else "-"


def _arm_order(rows: list) -> list:
    seen = []
    for r in rows:
        if r["arm"] and r["arm"] not in seen:
            seen.append(r["arm"])
    return sorted(seen, key=lambda a: (ARMS.index(a) if a in ARMS else 99, a))


def table1(rows: list) -> list:
    lines = [
        "| 臂 | n | 首轮即过 | 修好 | 部分修好 | 未变 | 失败(含撞上限) | 修复率 | 无效重试率 "
        "| 平均调用 | 平均轮次 | 平均耗时(s) | 判卷通过 | 判卷降级 | 判卷否决 | 误判率 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for arm in _arm_order(rows):
        group = [r for r in rows if r["arm"] == arm]
        n = len(group)
        first_try = sum(1 for r in group if r["outcome"] == "pass_first_try")
        fixed = sum(1 for r in group if r["outcome"] == "fixed")
        partial = sum(1 for r in group if r["outcome"] == "partial")
        unchanged = sum(1 for r in group if r["outcome"] == "degraded_unchanged")
        failed = sum(1 for r in group if r["outcome"] == "failed")
        need_repair = n - first_try
        wasted = sum(1 for r in group if r["outcome"] == "failed" and r["replan"] > 0)
        badge = lambda k: sum(1 for r in group if r["rubric"] == k)   # noqa: E731
        mismatch = sum(1 for r in group if r["mismatch"])
        lines.append(
            f"| {ARM_LABEL.get(arm, arm)} | {n} | {first_try} | {fixed} | {partial} | {unchanged} "
            f"| {failed} | {_pct(fixed, need_repair)} | {_pct(wasted, n)} "
            f"| {_avg([r['calls'] for r in group])} | {_avg([r['replan'] for r in group])} "
            f"| {_avg([r['elapsed'] for r in group])} "
            f"| {badge('pass')} | {badge('degraded')} | {badge('rejected')} | {_pct(mismatch, n)} |")
    return lines


def table2(rows: list) -> list:
    lines = [
        "| 错误类型 | 臂 | n | 判卷通过 | 通过率 | 平均调用 | 平均轮次 | 主要路径 |",
        "|---|--:|--:|--:|--:|--:|--:|---|",
    ]
    kinds = sorted({(r["group"], r["inject"]) for r in rows}, key=lambda x: (x[0], x[1]))
    for group, inject in kinds:
        for arm in _arm_order(rows):
            sub = [r for r in rows if r["group"] == group and r["inject"] == inject
                   and r["arm"] == arm]
            if not sub:
                continue
            passed = sum(1 for r in sub if r["rubric"] == "pass")
            paths = [p for r in sub for p in r["paths"] if p and p != "-"]
            top = max(set(paths), key=paths.count) if paths else "-"
            lines.append(f"| {group}（{inject}） | {arm} | {len(sub)} | {passed} "
                         f"| {_pct(passed, len(sub))} | {_avg([r['calls'] for r in sub])} "
                         f"| {_avg([r['replan'] for r in sub])} | {top} |")
    return lines


def table3(rows: list) -> list:
    lines = [
        "| 臂 | 系统·通过 | 系统·降级 | 系统·否决 | 判卷·通过 | 判卷·降级 | 判卷·否决 "
        "| 无证据兜底 | 降级救援命中 | 误放行 | 误否决 | 粒度错位 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for arm in _arm_order(rows):
        group = [r for r in rows if r["arm"] == arm]
        cnt = lambda key, val: sum(1 for r in group if r[key] == val)   # noqa: E731
        lines.append(
            f"| {ARM_LABEL.get(arm, arm)} | {cnt('system', 'pass')} "
            f"| {cnt('system', 'degraded')} | {cnt('system', 'rejected')} "
            f"| {cnt('rubric', 'pass')} | {cnt('rubric', 'degraded')} "
            f"| {cnt('rubric', 'rejected')} "
            f"| {sum(1 for r in group if r['fallback'])} "
            f"| {sum(1 for r in group if r['rescued'])} "
            f"| {sum(1 for r in group if r['mismatch'] == '误放行')} "
            f"| {sum(1 for r in group if r['mismatch'] == '误否决')} "
            f"| {sum(1 for r in group if r['mismatch'] == '粒度错位')} |")
    return lines


def table4(rows: list) -> list:
    arms = _arm_order(rows)
    header = "| 任务 | 注入 | " + " | ".join(arms) + " | 判卷 |"
    lines = [header, "|---|--:|" + "|".join(["--:"] * len(arms)) + "|--|"]
    task_ids = []
    for r in rows:
        if r["task_id"] not in task_ids:
            task_ids.append(r["task_id"])
    for tid in task_ids:
        sub = [r for r in rows if r["task_id"] == tid]
        cells = []
        for arm in arms:
            cell = next((r for r in sub if r["arm"] == arm), None)
            cells.append(f"{cell['outcome']}/{cell['calls']}次" if cell else "—")
        rubric = {r["rubric"] for r in sub}
        lines.append(f"| {tid} | {sub[0]['inject']} | " + " | ".join(cells)
                     + f" | {'/'.join(sorted(rubric))} |")
    return lines


def unique_pass(rows: list) -> dict:
    """每臂"独占合格"的题数：该题只有这一臂判卷通过，其余臂都没过。

    这是"每条路径不可替代"的最直接证据：如果某臂恒为 0，说明它在任何题上都不是最优解。
    """
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], {}).setdefault(r["arm"], []).append(r["rubric"])
    counts = {a: 0 for a in ARMS}
    for _, per_arm in by_task.items():
        winners = [arm for arm, rubrics in per_arm.items() if "pass" in rubrics]
        if len(winners) == 1 and per_arm.get(winners[0]):
            counts[winners[0]] = counts.get(winners[0], 0) + 1
    return counts


def _detail_table(rows: list) -> list:
    lines = ["| 任务 | 臂 | 注入 | 路径序列 | 系统终态 | 判卷 | 引用 | 误判 | 判卷缺项 |",
             "|---|---|---|---|---|---|--:|---|---|"]
    for r in rows:
        seq = " → ".join(r["paths"]) if r["paths"] else "-"
        lines.append(f"| {r['task_id']} | {r['arm']} | {r['inject']} | {seq} "
                     f"| {KIND_CN.get(r['system'], r['system'])} "
                     f"| {KIND_CN.get(r['rubric'], r['rubric'])} | {r['cited']} "
                     f"| {r['mismatch'] or '一致'} "
                     f"| {'；'.join(r['why']) or '-'} |")
    return lines


def render(rows: list, run_dir: str) -> str:
    arms = _arm_order(rows)
    unmatched = sum(1 for r in rows if r["task_id"] == "-")
    lines = [
        "# 修复路径对照（v2.3 自动生成）",
        "",
        f"> **生成时间**：{datetime.now():%Y-%m-%d %H:%M}　"
        f"**数据源**：`{os.path.relpath(run_dir, ROOT)}`，共 **{len(rows)}** 份记录。",
        f"> **臂**：{'、'.join(ARM_LABEL.get(a, a) for a in arms)}；"
        f"任务集 `experiments/tasks_repair.json`（{len({r['task_id'] for r in rows})} 道题）。",
        "> **判卷口径**（与系统判定独立）：必答对象全覆盖 + 正文 ≥120 字 + 无截断 + 无编造 URL "
        "→ 合格；以上都满足但没写可核验引用（该题要求引用时）→ 降级；否则否决。",
        "> **误判**：系统终态与判卷不一致。`误放行` = 系统判通过但判卷不合格；"
        "`误否决` = 系统判否决但判卷合格；`粒度错位` = 通过/降级/否决之间错位。",
        "",
        "## 表 1　每臂汇总",
        "",
        "> 修复率 = 修好 / 首轮失败；无效重试率 = 失败且发生过重试 / 该臂总数（\"白跑\"的比例）。",
        "",
        *table1(rows),
        "",
        "## 表 2　错误类型 × 臂（核心）",
        "",
        "> 分组来自任务集：A 检索缺口 / B 结构缺陷（3 对象）/ C 内容缺陷。"
        "同一行内比较四臂，才能看出\"这类错该走哪条路\"。",
        "",
        *table2(rows),
        "",
        "## 表 3　三态细化（降级拆分 + 兜底 / 救援）",
        "",
        "> `无证据兜底` = 全程没拿到检索证据；`降级救援命中` = 否决被改判降级（v2.2 反向闸门）。",
        "",
        *table3(rows),
        "",
        "## 表 4　逐题配对（同一道题在四臂下的实际结果）",
        "",
        "> 单元格为 `修复结果/LLM调用次数`；修复结果取 `pass_first_try|fixed|partial|"
        "degraded_unchanged|failed`。",
        "",
        *table4(rows),
        "",
        "## 自动观察（脚本算的，不含人工结论）",
        "",
    ]

    stats = []
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        n = len(group)
        passed = sum(1 for r in group if r["rubric"] == "pass")
        mismatch = sum(1 for r in group if r["mismatch"])
        stats.append((arm, n, passed, mismatch))
    if stats:
        best_pass = max(stats, key=lambda x: (x[2] / x[1] if x[1] else 0))
        least_bad = min(stats, key=lambda x: (x[3] / x[1] if x[1] else 1))
        lines.append(f"- 判卷合格率最高：**{ARM_LABEL.get(best_pass[0], best_pass[0])}**"
                     f"（{best_pass[2]}/{best_pass[1]}）")
        lines.append(f"- 误判率最低：**{ARM_LABEL.get(least_bad[0], least_bad[0])}**"
                     f"（{least_bad[3]}/{least_bad[1]}）")
    uniq = unique_pass(rows)
    lines.append("- 独占合格题数（只有该臂判卷通过的题）："
                 + "，".join(f"{ARM_LABEL.get(a, a)}={uniq.get(a, 0)}" for a in arms))
    if unmatched:
        lines.append(f"- 注意：有 **{unmatched}** 份记录没匹配到任务（`--dir` 指向旧记录或题目改名），"
                     "这些记录的必答对象为空，判卷只看了字数/引用/截断。")
    lines += [
        "",
        "## 明细（逐份记录，写案例集时按此取数）",
        "",
        *_detail_table(rows),
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v2.3 修复路径四臂汇总 → docs/修复路径对照.md")
    parser.add_argument("--dir", default=RUN_DIR, help="记录目录（默认 data/runs_v3）")
    parser.add_argument("--tasks", default=TASKS_PATH, help="任务集 JSON")
    parser.add_argument("--out", default=OUT_PATH, help="输出 Markdown")
    parser.add_argument("--stdout", action="store_true", help="只打印，不写文件")
    args = parser.parse_args(argv)

    rows = build_rows(args.dir, load_tasks(args.tasks))
    if not rows:
        print(f"[空] {args.dir} 下没有记录，先跑 experiments/repair_matrix.py")
        return 1
    text = render(rows, args.dir)
    if args.stdout:
        print(text)
        return 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[写出] {os.path.relpath(args.out, ROOT)}（{len(rows)} 份记录）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

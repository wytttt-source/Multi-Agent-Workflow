# -*- coding: utf-8 -*-
"""v2.5 修复路径四臂汇总：把 data/runs_v4/P-*/ 的 JSON 变成 docs/修复路径对照_v2.5.md。

用法：
    python experiments/evaluate_answers.py                      # 读 data/runs_v4，写 docs/修复路径对照_v2.5.md
    python experiments/evaluate_answers.py --stdout              # 只打印不落盘
    python experiments/evaluate_answers.py --dir data/runs_v3 --out docs/修复路径对照.md
                                                                 # 复算 v2.4 的 48 份表（口径相同，可直接对照）
    python experiments/evaluate_answers.py --dir data/runs_v2 --stdout
                                                                 # 用旧记录验证判卷口径（应零翻转）
    python experiments/evaluate_answers.py --dir data/runs_v3_pre_v24 --out tmp.md
                                                                 # 与更早批次对照（口径不同，只看趋势）

判卷与系统判定**相互独立**：判卷只用「任务给定的必答对象 + 可核验引用 + 字数/截断」，
不看反思节点的 verdict。两者不一致即记一次误判，这是"判得对不对"的唯一共同标准。

四张表：
    表 1  每臂汇总（修复率 / 无效重试率 / 成本 / 三态 / 误判）
    表 2  错误类型 × 臂（核心：哪种错该走哪条路）
    表 3  三态细化（降级拆分 + 兜底 / 救援命中）
    表 4  逐题配对（同一道题在四臂下的实际结果，判卷按臂分列）

v2.4 修正（都只改"怎么读"，不改判卷标准）：
    - 表 1 补「首轮失败」列；"平均轮次"改为"平均修复轮次"（轮次-1），不再把末轮算进去；
    - 表 2/4 按 (组, 任务编号) 排序，单样本组标 `（n=1）`，避免把 1/1 读成 100% 的能力；
    - 表 4 的判卷按臂分列（原来把四臂合成一个集合，读不出"谁合格"）；
    - 明细表区分「注入（任务声明）」与「实际注入」（运行时标记），并标注首轮是否真的过了门控；
    - 明细表的路径序列末位写终态与修复轮数，不再出现 "local → （空）"。
    - 同一 (臂, 主题) 多份记录只取最新一份（重跑不会把分母撑大），被覆盖的旧文件会打印出来。
    - 判卷的「必答对象」改用**实质覆盖**（`answer_completeness(..., substantive=True)`）：
      先剔掉「资料未提供 X / 仅覆盖 X」这类可得性声明句再数字面覆盖。只改判卷、不改系统，
      否则"答案自己声明没查到 Kafka"也会算讲清了 Kafka（假 `误否决`）。

v2.5 变更（只改默认值，不改判卷标准）：
    - 默认记录目录 `data/runs_v4`（17 题 × 4 臂 = 68 份），默认输出 `docs/修复路径对照_v2.5.md`；
    - 修复路径多一条 `rewrite`（复用已有证据、不再检索），表 2/表 4 的「主要路径」会直接显示它；
      判卷不看路径，仍按「必答对象 / 字数 / 截断 / 可核验引用」四项。
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

from utils import content_gate, answer_completeness, _body_chars   # noqa: E402

DOC_VERSION = "v2.5"   # 只决定报告标题与默认文件名，判卷口径不随版本变
RUN_DIR = os.path.join(ROOT, "data", "runs_v4")
TASKS_PATH = os.path.join(ROOT, "experiments", "tasks_repair.json")
OUT_PATH = os.path.join(ROOT, "docs", "修复路径对照_{0}.md".format(DOC_VERSION))

ARMS = ["B0", "F", "R", "H"]
ARM_LABEL = {"B0": "B0 无反思", "F": "F 固定类型", "R": "R 自由反思", "H": "H 混合（本文）"}
KIND_CN = {"pass": "通过", "degraded": "降级", "rejected": "否决"}
# 终轮判定是谁下的（repair.rounds[-1].decision_source）。"规则兜底"= 反思节点 LLM
# 超时/不可用时按确定性规则给的结论，不是语义裁判判的 —— 这类记录必须先标出来，
# 否则会把"没判成"当成"判准了"（2026-11-14 实跑里 C1 的 F/H 两臂都撞上过 GLM 超时）。
DECIDED_CN = {
    "llm": "语义裁判",
    "coverage_rule": "覆盖率规则",
    "gate": "确定性门控",
    "free_reflection": "自由反思(重写即过)",
    "rule_fallback": "规则兜底(LLM超时)",
    "baseline": "基线(不判)",
    "": "—",
}
# 系统三态与判卷三态用同一套枚举，才能逐条比"判得对不对"
SYSTEM_KIND = {"通过": "pass", "降级通过": "degraded"}


def load_tasks(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]
    return {t["topic"]: t for t in tasks}


def load_records(run_dir: str) -> list:
    """加载 run_dir 下各臂记录；同一 (臂, 主题) 只保留**最新**一份。

    为什么要去重：调参/重跑同一道题会留下多份记录，全量统计会把一次任务算成两票，
    修复率、误判率都会被稀释。这里按记录里的 ts 取最新，并提示被覆盖的旧文件，
    这样"数据没清干净"是可见的，而不是悄悄改变分母。
    """
    picked = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"[跳过] 解析失败：{os.path.relpath(path, ROOT)}")
            continue
        group = (rec.get("group") or "").strip()
        arm = group[2:] if group.startswith("P-") else group
        key = (arm or os.path.basename(os.path.dirname(path)),
               rec.get("topic") or os.path.basename(path))
        prev = picked.get(key)
        if prev and (prev["rec"].get("ts") or "") >= (rec.get("ts") or ""):
            print(f"[跳过] 旧记录（同题已更新）：{os.path.relpath(path, ROOT)}")
            continue
        if prev:
            print(f"[覆盖] 同题旧记录：{os.path.relpath(prev['path'], ROOT)}")
        picked[key] = {"path": path, "arm": arm, "rec": rec}
    return list(picked.values())


def system_kind(rec: dict) -> str:
    """系统终态：优先读 repair.final_verdict（v2.3 轨迹），退回 reflection.verdict（v2 记录）。"""
    repair = rec.get("repair") or {}
    verdict = repair.get("final_verdict") or (rec.get("reflection") or {}).get("verdict") or ""
    return SYSTEM_KIND.get(verdict, "rejected")


def judge(rec: dict, task: dict) -> dict:
    """独立判卷（不看系统 verdict），返回 rubric 三态 + 依据。

    注意：运行记录里**没有** `rec["evidence"]`（证据正文不落盘，只落条数），
    所以这里显式传空列表。判卷用的四项（字数 / 截断 / URL 可核验 / 引用条数）
    都不需要证据正文，只有 `sources` 参与 URL 核验。

    必答对象覆盖用 `substantive=True`（比系统侧更严）：先剔掉"资料未提供 X"这类
    可得性声明句再数字面覆盖。否则答案只要写一句"资料未提供 Kafka 的数据"，
    就会被判成"讲清了 Kafka"，把系统的正确否决记成误否决。
    """
    result = rec.get("result") or ""
    gate = content_gate(result, rec.get("sources") or [], [])
    comp = answer_completeness(result, (task or {}).get("required_objects") or [],
                               substantive=True)
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


def _total_searches(rec: dict, rounds: list) -> int:
    """一次运行实际发出的检索 query 数（首轮 + 各轮补检索）。

    不能直接读 `repair.cost.searches`：`search.per_query` 只累计"首轮 + 局部修复的补检索"，
    自由反思的补检索写在 `search.retry_query` / `search_retry` 里、**不并入** per_query，
    所以老记录里 free 臂全是 1（H 臂局部修复反而是 2），两臂根本不可比。
    口径统一成：per_query 条数 + 每轮 free 修复补 1 次（`_build_retry_query` 每轮只发一条 query）。
    """
    searches = len((rec.get("search") or {}).get("per_query") or [])
    searches += sum(1 for r in rounds
                    if r.get("path") == "free" and (r.get("round") or 0) > 0)
    return searches


def build_rows(run_dir: str, tasks: dict) -> list:
    rows = []
    for item in load_records(run_dir):
        rec = item["rec"]
        task = tasks.get(rec.get("topic") or "", {})
        verdict = judge(rec, task)
        repair = rec.get("repair") or {}
        refl = rec.get("reflection") or {}
        s_kind = system_kind(rec)
        rounds = repair.get("rounds") or []
        # 注入列取"实际执行时记录的标记"（run_stats.injected），任务定义只作兜底：
        # P1 那次事故就是任务定义写着 truncate、实际注入被截在【来源】里没生效，
        # 只看任务定义会把"没注入"当成"注入了但没修好"。
        injected = ((rec.get("run_stats") or {}).get("injected")
                    or rec.get("injected") or [])
        rows.append({
            "arm": item["arm"],
            "task_id": task.get("id") or "-",
            "group": task.get("group") or "-",
            "topic": rec.get("topic") or "",
            # inject = 任务集声明；injected = 运行时实际命中
            "inject": (task.get("inject") or (rec.get("config") or {}).get("inject_fault") or "-"),
            "injected": [m for m in injected if m],
            "outcome": repair.get("outcome") or "unknown",
            "rounds": len(rounds),
            "rounds_detail": rounds,
            # 首轮是否真的跑过内容门控：False/None 表示这轮的判定没测字数/引用
            "gate_measured": (rounds[0].get("gate_measured") if rounds else None),
            "paths": [r.get("path") or "-" for r in rounds],
            "replan": rec.get("replan_count", 0),
            "calls": refl.get("llm_calls") or (repair.get("cost") or {}).get("llm_calls") or 0,
            # 补检索次数（repair.cost.searches）：C 组看 H（局部重写）与 R（自由反思）的差别就在这一列。
            # v2.4：老记录里 free 臂的补检索没并入 search.per_query（成本恒为 1），
            # 这里统一按"每轮 free 修复 = 1 次补检索"重算，读数才可比。
            "searches": _total_searches(rec, rounds),
            # 终稿正文字数：判卷只卡 ≥120 字下限，不区分"刚够线"和"讲透了"，
            # 所以单列一栏，避免把 127 字的定向补写和 657 字的整篇重写看成同一档
            "chars": _body_chars(rec.get("result") or ""),
            "elapsed": rec.get("elapsed_seconds") or 0,
            "system": s_kind,
            "rubric": verdict["rubric"],
            "why": verdict["why"],
            "cited": verdict["gate"]["cited_count"],
            "fallback": bool(rec.get("used_fallback")),
            # 终轮判定是谁下的：语义裁判/覆盖率规则/确定性门控/规则兜底(LLM 超时)
            "decided_by": (rounds[-1].get("decision_source") or "") if rounds else "",
            "rescued": bool(refl.get("rescued_from")),
            "mismatch": mismatch_of(s_kind, verdict["rubric"]),
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
        "| 臂 | n | 首轮即过 | 首轮失败 | 修好 | 部分修好 | 未变 | 失败(含撞上限) | 修复率 "
        "| 无效重试率 | 平均调用 | 平均检索 | 平均修复轮次 | 平均耗时(s) | 平均正文(字) "
        "| 判卷通过 | 判卷降级 | 判卷否决 | 误判率 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
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
            f"| {ARM_LABEL.get(arm, arm)} | {n} | {first_try} | {need_repair} "
            f"| {fixed} | {partial} | {unchanged} "
            f"| {failed} | {_pct(fixed, need_repair)} | {_pct(wasted, n)} "
            f"| {_avg([r['calls'] for r in group])} "
            f"| {_avg([r['searches'] for r in group])} "
            f"| {_avg([r['rounds'] for r in group])} "
            f"| {_avg([r['elapsed'] for r in group])} "
            f"| {_avg([r['chars'] for r in group])} "
            f"| {badge('pass')} | {badge('degraded')} | {badge('rejected')} | {_pct(mismatch, n)} |")
    return lines


def table2(rows: list) -> list:
    lines = [
        "| 错误类型 | 臂 | n | 判卷通过 | 通过率 | 平均调用 | 平均修复轮次 | 主要路径 |",
        "|---|--:|--:|--:|--:|--:|--:|---|",
    ]
    # 按"组 + 组内最小任务编号"排序，避免 inject 名字的字典序把 A/B/C 打散
    kinds = {}
    for r in rows:
        kinds.setdefault((r["group"], r["inject"]), []).append(r["task_id"])
    for group, inject in sorted(kinds, key=lambda k: (k[0], min(kinds[k]))):
        for arm in _arm_order(rows):
            sub = [r for r in rows if r["group"] == group and r["inject"] == inject
                   and r["arm"] == arm]
            if not sub:
                continue
            passed = sum(1 for r in sub if r["rubric"] == "pass")
            paths = [p for r in sub for p in r["paths"] if p and p != "-"]
            top = max(set(paths), key=paths.count) if paths else "-"
            flag = "（n=1）" if len(sub) == 1 else ""
            lines.append(f"| {group}（{inject}）{flag} | {arm} | {len(sub)} | {passed} "
                         f"| {_pct(passed, len(sub))} | {_avg([r['calls'] for r in sub])} "
                         f"| {_avg([r['rounds'] for r in sub])} | {top} |")
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
    header = ("| 任务 | 注入 | " + " | ".join(arms) + " | "
              + " | ".join(f"{a}·判卷" for a in arms) + " |")
    lines = [header,
             "|---|--:|" + "|".join(["--:"] * len(arms))
             + "|" + "|".join(["--"] * len(arms)) + "|"]
    # 按 (组, 任务编号) 排序：原先按记录出现顺序，A3/C3/C2/A2 混排，读不出组内趋势
    order = sorted({(r["group"], r["task_id"]) for r in rows}, key=lambda x: (x[0], x[1]))
    for group, tid in order:
        sub = [r for r in rows if r["task_id"] == tid]
        cells, rubrics = [], []
        for arm in arms:
            cell = next((r for r in sub if r["arm"] == arm), None)
            cells.append(f"{cell['outcome']}/{cell['calls']}次" if cell else "—")
            rubrics.append(KIND_CN.get(cell["rubric"], cell["rubric"]) if cell else "—")
        flag = "（n=1）" if len(sub) == 1 else ""
        lines.append(f"| {tid}{flag} | {sub[0]['inject']} | " + " | ".join(cells)
                     + " | " + " | ".join(rubrics) + " |")
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


def _path_seq(row: dict) -> str:
    """路径序列：只画"走过哪条路"，末位标终态。

    末轮不再当成一条路径（它的 path 恒为空，直接画会出现 "local → "）；
    修复轮数 = 总轮次 - 1，写进末位，一眼能看出这条臂花了几个来回。
    """
    rounds = row.get("rounds_detail") or []
    if not rounds:
        return "-"
    seq = [r.get("path") for r in rounds if r.get("path")]
    final = KIND_CN.get(row["system"], row["system"])
    if len(rounds) > 1:
        seq.append(f"{final}（修复 {len(rounds) - 1} 轮）")
    else:
        seq.append(final)
    return " → ".join(seq)


def _detail_table(rows: list) -> list:
    lines = ["| 任务 | 臂 | 注入（任务） | 实际注入 | 首轮门控 | 路径序列 "
             "| 系统终态 | 终轮判定 | 判卷 | 引用 | 误判 | 判卷缺项 |",
             "|---|---|---|---|---|---|---|---|---|--:|---|---|"]
    ordered = sorted(rows, key=lambda r: (r["group"], r["task_id"],
                                          ARMS.index(r["arm"]) if r["arm"] in ARMS else 99))
    for r in ordered:
        gate = "—" if r["gate_measured"] in (None, False) else "已测"
        injected = "、".join(r["injected"]) or "—"
        lines.append(f"| {r['task_id']} | {r['arm']} | {r['inject']} | {injected} | {gate} "
                     f"| {_path_seq(r)} | {KIND_CN.get(r['system'], r['system'])} "
                     f"| {DECIDED_CN.get(r['decided_by'], r['decided_by'] or '—')} "
                     f"| {KIND_CN.get(r['rubric'], r['rubric'])} | {r['cited']} "
                     f"| {r['mismatch'] or '一致'} | {'；'.join(r['why']) or '-'} |")
    return lines


def render(rows: list, run_dir: str) -> str:
    arms = _arm_order(rows)
    unmatched = sum(1 for r in rows if r["task_id"] == "-")
    lines = [
        "# 修复路径对照（{0} 自动生成）".format(DOC_VERSION),
        "",
        f"> **生成时间**：{datetime.now():%Y-%m-%d %H:%M}　"
        f"**数据源**：`{os.path.relpath(run_dir, ROOT)}`，共 **{len(rows)}** 份记录。",
        f"> **臂**：{'、'.join(ARM_LABEL.get(a, a) for a in arms)}；"
        f"任务集 `experiments/tasks_repair.json`（{len({r['task_id'] for r in rows})} 道题）。",
        "> **判卷口径**（与系统判定独立）：必答对象全覆盖（**实质覆盖**：剔除「资料未提供 X」"
        "这类可得性声明句后再数字面覆盖）+ 正文 ≥120 字 + 无截断 + 无编造 URL "
        "→ 合格；以上都满足但没写可核验引用（该题要求引用时）→ 降级；否则否决。",
        "> **误判**：系统终态与判卷不一致。`误放行` = 系统判通过但判卷不合格；"
        "`误否决` = 系统判否决但判卷合格；`粒度错位` = 通过/降级/否决之间错位。",
        "> **读表注意**：`n=1` 的小组只作方向参考；`首轮门控=—` 表示该轮没落门控事实"
        "（早期记录或未进入门控的分支），不是「正文 0 字」。",
        "",
        "## 表 1　每臂汇总",
        "",
        "> 修复率 = 修好 / 首轮失败；无效重试率 = 失败且发生过重试 / 该臂总数（\"白跑\"的比例）；"
        "平均检索 = 一次运行实际发出的检索 query 数（首轮 + 各轮补检索）。"
        "自由反思的补检索不进 `search.per_query`，由判卷脚本按轮次补计，"
        "所以该列与记录里的 `repair.cost.searches` 可能差 1~2（老记录未计入 free 补检索）。"
        "平均正文(字) = 终稿正文（剔除【来源】后）字数均值：判卷只卡 ≥120 字下限，"
        "\"刚够线的定向补写\"与\"整篇重写\"的差别只在这一列看得出来。",
        "",
        *table1(rows),
        "",
        "## 表 2　错误类型 × 臂（核心）",
        "",
        "> 分组来自任务集：A 检索缺口 / B 结构缺陷（3 对象）/ C 内容缺陷。"
        "同一行内比较四臂，才能看出\"这类错该走哪条路\"。标 `（n=1）` 的小组只有一个样本，"
        "百分比只作方向参考。",
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
        "degraded_unchanged|failed`。右侧四列是**各臂各自的判卷结果**（不合并成一个集合），"
        "`—` 表示该臂没有这份记录。",
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
    active = [s for s in stats if s[1]]
    if len(active) >= 2:
        # 只说"最高/最低"是不够的：四臂都为 0 时"最高"等于没结论，要明确说不下结论，
        # 并允许并列（同一分数可能多臂共享，只报一个会误导成"唯一最优"）。
        best_rate = max(s[2] / s[1] for s in active)
        if best_rate == 0:
            lines.append("- 四臂判卷通过数都是 0：这批缺陷没有一条被真正修好（或判卷口径过严），"
                         "这里先不下\"谁更好\"的结论，去看明细表里被判卷否决的缺项。")
        else:
            top = [s for s in active if s[2] / s[1] == best_rate]
            lines.append("- 判卷合格率最高：" + "，".join(
                f"**{ARM_LABEL.get(s[0], s[0])}**（{s[2]}/{s[1]}，{best_rate:.0%}）"
                for s in top))
        worst_rate = min(s[3] / s[1] for s in active)
        bottom = [s for s in active if s[3] / s[1] == worst_rate]
        lines.append("- 误判率最低：" + "，".join(
            f"**{ARM_LABEL.get(s[0], s[0])}**（{s[3]}/{s[1]}，{worst_rate:.0%}）"
            for s in bottom))
    elif active:
        s = active[0]
        lines.append(f"- 目前只有 **{ARM_LABEL.get(s[0], s[0])}** 一个臂有记录（n={s[1]}），"
                     "不做跨臂比较。")
    if len(active) >= 2:
        # 成本也要自己说话：修复率打平时，靠"多查一次"还是"只重写一遍"是两条不同的路
        per_arm_search = {}
        for arm_, n_, _, _ in active:
            grp = [r for r in rows if r["arm"] == arm_]
            per_arm_search[arm_] = sum(r["searches"] for r in grp) / len(grp)
        lines.append("- 平均检索次数（首轮+补检索）：" + "，".join(
            f"{ARM_LABEL.get(a, a)}={v:.1f}" for a, v in per_arm_search.items())
            + "（修复率相同时，这一列越低 = 越少重查、越靠重写）")
        # 判卷只卡字数下限，"过门"与"讲透"的差别只能靠正文长度这一列看出来
        per_arm_chars = {}
        for arm_, n_, _, _ in active:
            grp = [r for r in rows if r["arm"] == arm_]
            per_arm_chars[arm_] = sum(r["chars"] for r in grp) / len(grp)
        lines.append("- 平均正文字数（终稿、不含来源清单）：" + "，".join(
            f"{ARM_LABEL.get(a, a)}={v:.0f}" for a, v in per_arm_chars.items())
            + "（判卷合格只看是否过 120 字门槛；这一列只作\"讲得够不够透\"的参考）")
    uniq = unique_pass(rows)
    lines.append("- 独占合格题数（只有该臂判卷通过的题）："
                 + "，".join(f"{ARM_LABEL.get(a, a)}={uniq.get(a, 0)}" for a in arms))
    # 终轮判定来源：规则兜底 = 反思节点 LLM 超时，按确定性规则给的结论。
    # 这类记录的"系统终态"不是判出来的，读表时必须单独标出（否则会把超时当判准）。
    fb = [r for r in rows if r["decided_by"] == "rule_fallback"]
    fb_tags = "、".join("{0}·{1}".format(r["task_id"], r["arm"]) for r in fb)
    lines.append("- 终轮判定为**规则兜底**（反思 LLM 超时/不可用，按确定性规则给结论）："
                 f"**{len(fb)}**/{len(rows)} 份"
                 + (f"（{fb_tags}）" if fb else "")
                 + "。这类记录的终态不能当\"判准了\"用，只能说明\"那一轮没判成\"。")
    lines.append("- 读法提醒：判卷合格 ≠ 修好了 —— 修到轮次上限（`outcome=failed`）时，"
                 "最后一轮答案也可能已经合格（那是**误否决**，白跑的成本仍要记在它头上）；"
                 "两列必须一起看。")
    if unmatched:
        lines.append(f"- 注意：有 **{unmatched}** 份记录没匹配到任务（`--dir` 指向旧记录或题目改名），"
                     "这些记录的必答对象为空，判卷只看了字数/引用/截断。")
    lines += [
        "",
        "## 明细（逐份记录，写案例集时按此取数）",
        "",
        "> `终轮判定` = 最后一轮的结论是谁下的：语义裁判 / 覆盖率规则 / 确定性门控 / "
        "自由反思（补检索重写后直接过）/ 规则兜底（反思 LLM 超时，按确定性规则给结论）。",
        "",
        *_detail_table(rows),
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v2.5 修复路径四臂汇总 → docs/修复路径对照_v2.5.md")
    parser.add_argument("--dir", default=RUN_DIR, help="记录目录（默认 data/runs_v4；v2.4 表用 --dir data/runs_v3）")
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
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[写出] {os.path.relpath(args.out, ROOT)}（{len(rows)} 份记录）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

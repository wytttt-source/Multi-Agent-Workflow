# -*- coding: utf-8 -*-
"""第三步预演：**终检收严 + 终态救援**，只读回放 runs_v4 / runs_v3，列出翻转清单。

用法：
    python experiments/replay_final_check.py                # 回放 v2.5(68) + v2.4(48)
    python experiments/replay_final_check.py --dir data/runs_v4
    python experiments/replay_final_check.py --dir data/runs_v3 --quiet

这个脚本**不跑流水线、不改数据、不改主代码**：它只把已落盘的 `result` / `sources`
拿来按新口径重算一遍终态，回答一个问题 —— "如果终检按新口径收严、并对撞上限的
终稿做救援，有多少条记录的结论会变"。

新口径两条（都只改"怎么判终稿"，不改检索/重规划任何一行代码）：

  1) 收严 `substantive=True`：把判卷侧早就用的「实质覆盖」推到**系统侧终检**。
     `answer_completeness(..., substantive=True)` 会先剔掉
     "资料未提供 X / 仅涉及 X / 建议补充…检索式"这类**可得性声明句**，再数字面覆盖。
     现状（v2.5）= 判卷用 True、系统侧仍用默认 False，于是答案里只要出现
     "PostgreSQL 与 SQLite 的资料未提供"，系统就认为对象已覆盖（B2·F 就是这么过的）。

  2) 终态救援：记录终态是 `否决`（含三轮撞 `R4_repeat` 的 `failed`）时，
     若**终稿本身**在新口径下达标（对象齐、正文够长、不截断、URL 可核验、有引用），
     则该题记"救援合格"，不再算失败。理由：判卷本来就只看终稿，系统终态是过程指标。

翻转按**成因**分四类，统计时分开列（不能混成一个数）：

    收严↓      旧终态 通过 → 新终态 否决/降级，且"不剔声明句"的对照尺仍放行
               → 这一格的三态只可能被 `substantive=True` 改掉，是收严的净效应
    救援↑      旧终态 否决 → 新终态 通过（含三轮撞上限的 `failed`）
    硬项不符   两把尺都否决，但记录终态是 通过 → 与收严无关：记录终态与终稿
               硬项（截断/字数/引用/必答对象）本来就不一致。B0 全部在此类
               （基线反思关闭、没有终检），所以**不计入**新口径翻转
    粒度       其余（降级 ↔ 通过）

⚠️ 口径警告（必须和数字一起引用）：新口径**就是判卷口径**，所以回放里
"系统终态 vs 判卷"的不一致必然被抹平 —— 这是同一把尺子的必然结果，**不能当能力提升**。
它只说明"过程指标（反思路径的判定）与结果指标（终稿质量）分离后，账怎么记"；
真要证明系统判得准，仍然只能靠实跑后比对，回放只能做上界估计。
"""
import argparse
import contextlib
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "experiments")):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils import content_gate, answer_completeness          # noqa: E402
import evaluate_answers as ev                                # noqa: E402

FLIP_GROUPS = ("收严↓", "救援↑", "硬项不符", "粒度")


def literal_vs_substantive(result: str, required: list) -> tuple:
    """字面覆盖 → 实质覆盖，用来看"收严打掉的到底是什么"。"""
    literal = answer_completeness(result, required, substantive=False)
    strict = answer_completeness(result, required, substantive=True)
    return literal, strict


def final_check(rec: dict, task: dict, substantive: bool = True) -> dict:
    """新终检：与 `evaluate_answers.judge` 同一把尺子（实质覆盖 + 字数/截断/URL/引用）。

    只读终稿（`result` / `sources`），不读记录里的任何 verdict。
    `substantive=False` 是回放用的**对照尺**（= v2.5 系统侧口径：不剔可得性声明句），
    用它可以判断某次翻转到底是不是"收严"带来的：只有 `lit` 放行、`strict` 否决的格子
    才算收严的净效应；两把尺都否决的格子是"硬项本来就不符"，与收严无关。
    """
    result = rec.get("result") or ""
    sources = rec.get("sources") or []
    required = (task or {}).get("required_objects") or []
    requires_citation = (task or {}).get("requires_citation", True)
    gate = content_gate(result, sources, [])
    literal, strict = literal_vs_substantive(result, required)
    comp = strict if substantive else literal

    hard = []
    if not gate["len_ok"]:
        hard.append("正文过短")
    if gate["truncated"]:
        hard.append("截断")
    if not gate["no_fabricated_url"]:
        hard.append("不可核验URL")
    if comp["missing"]:
        hard.append("缺必答对象:" + "/".join(comp["missing"]))
    rubric = "rejected" if hard else (
        "degraded" if requires_citation and gate["cited_count"] == 0 else "pass")
    return {"rubric": rubric,
            "why": hard or (["可读但无引用"] if rubric == "degraded" else []),
            "gate": gate, "literal": literal, "strict": strict,
            "substantive": bool(substantive)}


def load(run_dir: str) -> list:
    """复用 `evaluate_answers` 的加载与去重（同题只留最新），把它的提示音吞掉。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rows = ev.load_records(run_dir)
    notes = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    return rows, notes


def first_try_failed(rec: dict) -> bool:
    repair = rec.get("repair") or {}
    rounds = repair.get("rounds") or []
    if rounds:
        return (rounds[0].get("pass_kind") or "") == "rejected"
    return rec.get("injected") is not None


def arm_label(arm: str) -> str:
    return ev.ARM_LABEL.get(arm, arm)


def last_decision_source(rec: dict) -> str:
    """终轮判定是谁下的：规则兜底（LLM 超时）的格子不能当"判准了"用，要点名。"""
    rounds = (rec.get("repair") or {}).get("rounds") or []
    if rounds:
        return rounds[-1].get("decision_source") or ""
    return (rec.get("reflection") or {}).get("decision_source") or ""


def replay_dir(run_dir: str, tasks: dict) -> dict:
    """回放一个目录，返回 {dir, cells, notes}；每格一条记录，`flip` 为空表示零翻转。"""
    rows, notes = load(run_dir)
    cells = []
    for item in rows:
        rec = item["rec"]
        task = tasks.get(rec.get("topic") or "")
        old = ev.system_kind(rec)
        new = final_check(rec, task, substantive=True)
        lit = final_check(rec, task, substantive=False)
        cell = {"dir": run_dir, "topic": rec.get("topic") or "",
                "task_id": (task or {}).get("id") or "",
                "arm": item["arm"], "old": old, "new": new["rubric"],
                "lit": lit["rubric"],
                "why": new["why"], "lit_why": lit["why"], "gate": new["gate"],
                "literal_hit": len(new["literal"]["hit"]),
                "strict_hit": len(new["strict"]["hit"]),
                "terms": len(new["strict"]["terms"]),
                "strict_missing": new["strict"]["missing"],
                "first_failed": first_try_failed(rec),
                "decided": last_decision_source(rec),
                "outcome": (rec.get("repair") or {}).get("outcome") or ""}
        # 翻转分类按"成因"分，不能把 B0 的既有错位算成收严的功劳：
        #   收严↓    = 对照尺放行、实质尺否决 → 只有 substantive 才能解释这次翻转
        #   救援↑    = 记录否决、实质尺通过   → 终稿其实合格
        #   硬项不符 = 两把尺都否决           → 记录终态与终稿硬项（截断/字数/引用/对象）
        #                                        本来就不一致，与收严无关（B0 全在此类）
        #   粒度      = 其余（降级 ↔ 通过）
        if cell["old"] == cell["new"]:
            cell["flip"] = ""
        elif cell["old"] == "rejected" and cell["new"] == "pass":
            cell["flip"] = "救援↑"
        elif cell["old"] == "pass" and cell["lit"] == "pass":
            cell["flip"] = "收严↓"
        elif cell["lit"] != cell["old"]:
            cell["flip"] = "硬项不符"
        else:
            cell["flip"] = "粒度"
        cells.append(cell)
    return {"dir": run_dir, "cells": cells, "notes": notes}


def summarize(replay: dict) -> list:
    """按臂汇总：份数 / 旧通过 / 新通过 / 四类翻转 / 首轮失败 / 修复率 新旧。

    `首轮失败` 是修复率的分母（首轮被判否决的份数）；旧修复率数记录终态，
    新修复率数新终检（含救援），两者相减就是这一步改口径带来的净值。
    """
    out = []
    arms = [a for a in ev.ARMS] + sorted({c["arm"] for c in replay["cells"]} - set(ev.ARMS))
    for arm in arms:
        sub = [c for c in replay["cells"] if c["arm"] == arm]
        if not sub:
            continue
        denom = sum(1 for c in sub if c["first_failed"])
        old_fix = sum(1 for c in sub if c["old"] == "pass" and c["first_failed"])
        new_fix = sum(1 for c in sub if c["new"] == "pass" and c["first_failed"])
        out.append({"arm": arm, "n": len(sub), "denom": denom,
                    "old_pass": sum(1 for c in sub if c["old"] == "pass"),
                    "new_pass": sum(1 for c in sub if c["new"] == "pass"),
                    "收严↓": sum(1 for c in sub if c["flip"] == "收严↓"),
                    "救援↑": sum(1 for c in sub if c["flip"] == "救援↑"),
                    "硬项不符": sum(1 for c in sub if c["flip"] == "硬项不符"),
                    "粒度": sum(1 for c in sub if c["flip"] == "粒度"),
                    "old_rate": (old_fix / denom) if denom else None,
                    "new_rate": (new_fix / denom) if denom else None})
    return out


def fmt_rate(rate) -> str:
    return "  n/a" if rate is None else f"{rate * 100:5.1f}%"


def print_summary(replay: dict) -> dict:
    name = os.path.relpath(replay["dir"], ROOT)
    rows = summarize(replay)
    total = sum(r["n"] for r in rows)
    flips = [c for c in replay["cells"] if c["flip"]]
    real = [c for c in flips if c["flip"] in ("收严↓", "救援↑", "粒度")]
    print("")
    print(f"--- {name}：{total} 份；新口径真翻转 {len(real)} 份 / "
          f"{len({c['topic'] for c in real})} 题；另有 {len(flips) - len(real)} 份"
          f"「硬项不符」（收严前就对不上，见下）---")
    print(f"  {'臂':<14}{'份':>4}{'旧通过':>7}{'新通过':>7}{'收严↓':>7}{'救援↑':>7}"
          f"{'粒度':>6}{'硬项不符':>9}{'首轮失败':>9}{'修复率 旧 → 新':>18}")
    for r in rows:
        print(f"  {arm_label(r['arm']):<12}{r['n']:>5}{r['old_pass']:>8}{r['new_pass']:>8}"
              f"{r['收严↓']:>7}{r['救援↑']:>7}{r['粒度']:>7}{r['硬项不符']:>9}"
              f"{r['denom']:>10}{fmt_rate(r['old_rate']):>9} →{fmt_rate(r['new_rate']):>7}")
    b0 = [r for r in rows if r["arm"] == "B0"]
    if b0 and b0[0]["硬项不符"]:
        print(f"  注：B0 为基线（反思关闭、无终检），其 {b0[0]['硬项不符']} 格属于"
              "「记录通过 vs 终稿硬项不合格」，是本来就存在的错位，不计入新口径翻转，"
              "也不进误判率（基线不判）。")
    for note in replay["notes"]:
        print(f"  [去重] {note}")
    fallback = [c for c in replay["cells"] if c["decided"] == "rule_fallback"]
    if fallback:
        print("  注：终轮为**规则兜底**（反思 LLM 超时，非语义裁判）的格子："
              + "、".join(f"{c['task_id']}·{c['arm']}" for c in
                          sorted(fallback, key=lambda c: (c["task_id"], c["arm"])))
              + "\n      其终态只说明「那一轮没判成」，不能当「判准了」引用。")
    return {"rows": rows, "flips": flips, "real": real, "total": total}


def print_flips(replay: dict) -> None:
    flips = [c for c in replay["cells"] if c["flip"]]
    if not flips:
        print("  （零翻转）")
        return
    order = {g: i for i, g in enumerate(FLIP_GROUPS)}
    flips.sort(key=lambda c: (order.get(c["flip"], 9), c["task_id"] or c["topic"], c["arm"]))
    for c in flips:
        gate = c["gate"]
        head = (f"  [{c['flip']}] {c['task_id'] or '?':<3}{c['arm']:<3}"
                f"{ev.KIND_CN.get(c['old'], c['old'])} → {ev.KIND_CN.get(c['new'], c['new'])}")
        if c["flip"] == "收严↓":
            detail = (f"字面覆盖 {c['literal_hit']}/{c['terms']} → "
                      f"实质覆盖 {c['strict_hit']}/{c['terms']}")
            if c["strict_missing"]:
                detail += "（缺 " + "、".join(c["strict_missing"]) + "）"
        elif c["flip"] == "硬项不符":
            detail = "收严前就不符：" + ("；".join(c["lit_why"]) or "—")
        else:
            detail = "；".join(c["why"]) or "—"
        tail = (f" 正文 {gate['body_chars']} 字 / 引用 {gate['cited_count']} / "
                f"截断 {'是' if gate['truncated'] else '否'} / 记录 outcome {c['outcome'] or '—'}")
        print(head + "  " + detail + tail)


def main() -> int:
    ap = argparse.ArgumentParser(description="终检收严 + 终态救援 的只读回放")
    ap.add_argument("--dir", action="append", default=None,
                    help="要回放的记录目录，可重复；默认 data/runs_v4 与 data/runs_v3")
    ap.add_argument("--tasks", default=ev.TASKS_PATH,
                    help="任务声明 JSON（默认 experiments/tasks_repair.json）")
    ap.add_argument("--quiet", action="store_true", help="只打印汇总，不打印逐条翻转清单")
    args = ap.parse_args()

    dirs = args.dir or [os.path.join(ROOT, "data", "runs_v4"),
                        os.path.join(ROOT, "data", "runs_v3")]
    tasks = ev.load_tasks(args.tasks)

    print("=== 第三步预演：终检收严（substantive=True）+ 终态救援（只读，不跑流水线）===")
    print("  收严↓ = 旧 通过 → 新 否决/降级（只有剔除声明句才能解释）；"
          "救援↑ = 旧 否决 → 新 通过；")
    print("  粒度 = 其余错位；硬项不符 = 记录通过但终稿硬项本就不合格"
          "（B0 全属此类，不计入翻转）")

    grand = {"total": 0, "flips": [], "real": []}
    for run_dir in dirs:
        if not os.path.isdir(run_dir):
            print(f"[跳过] 目录不存在：{os.path.relpath(run_dir, ROOT)}")
            continue
        replay = replay_dir(run_dir, tasks)
        info = print_summary(replay)
        if not args.quiet:
            print_flips(replay)
        grand["total"] += info["total"]
        grand["flips"] += [dict(c, dir=run_dir) for c in info["flips"]]
        grand["real"] += [dict(c, dir=run_dir) for c in info["real"]]

    if len(dirs) > 1:
        by_group = {g: sum(1 for c in grand["real"] if c["flip"] == g) for g in FLIP_GROUPS}
        print("")
        print(f"=== 合计：{grand['total']} 份；新口径真翻转 {len(grand['real'])} 份 / "
              f"{len({c['topic'] for c in grand['real']})} 题 "
              f"（收严↓ {by_group['收严↓']}，救援↑ {by_group['救援↑']}，"
              f"粒度 {by_group['粒度']}）；另有硬项不符 "
              f"{len(grand['flips']) - len(grand['real'])} 份 ===")
        rescue = sorted((c for c in grand["real"] if c["flip"] == "救援↑"),
                        key=lambda c: (os.path.basename(c["dir"]), c["task_id"], c["arm"]))
        if rescue:
            print("  仅靠救援就会改结论的格子："
                  + "、".join(f"{os.path.basename(c['dir'])}/{c['task_id']}·{c['arm']}"
                              for c in rescue))
        if not by_group["收严↓"]:
            print("  substantive=True 单独翻转 0 格：三臂里唯一实质缺对象的 B2·F 本来就被判否决，" +
                  "\n  收严不改变任何一格三态 —— 要证明它有用，得靠 D/S 组那些"
                  "\"答案自称没查到\"的样本，不在本批记录里。")
    print("")
    print("  口径提醒：新口径即判卷口径，回放里'系统 vs 判卷'不一致必然为 0，")
    print("            这是同一把尺子的结果，不能当能力提升；实跑仍需重跑 68 份验证。")
    return 0


if __name__ == "__main__":
    sys.exit(main())


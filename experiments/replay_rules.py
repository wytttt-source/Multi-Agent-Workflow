# -*- coding: utf-8 -*-
"""v2.5 第一步验收：规则命名不改判定 —— 用旧记录离线复算，要求「路径零翻转」。

背景
----
v2.2~v2.4 的选路依据散落在 `agents/nodes.py::_decide_next` 的 if/else 里，
论文无法回答"这条记录凭什么走这条路"。v2.5 把这些依据定名成 R1~R5
（见 `agents/nodes.py::REPAIR_RULES`）。**第一步只改名、不改判定**，
因此必须能用历史记录复算出同一批路径；但凡有一轮翻转，就说明"改名"顺手改了判定。

用法
----
    python experiments/replay_rules.py                     # 默认 data/runs_v3
    python experiments/replay_rules.py --dir data/runs_v4
    python experiments/replay_rules.py --verbose           # 逐轮打印
    python experiments/replay_rules.py --expect-flips 0    # 翻转数不符时退出码 1

复算口径（只读：不联网、不调 LLM、不写任何文件）
------------------------------------------------
1. 复算范围 = `repair.rounds[]` 中 `pass_kind == "rejected"` 的**决策轮**。
   判定通过（strict / degraded）的轮次是执行环节的收尾，其 `path` 只是沿用上一轮
   的选路，并不是 `_decide_next` 的输出，**不参与**复算（计入 terminal_rounds）。
2. 基准 = 该轮落盘的 `path`（当时实际走的路）。
3. 复算 = `_rule_and_desired()` → `_apply_repair_policy()`；
   策略取自记录里的 `config.repair_policy`（用 mock 临时覆盖模块常量，结束即还原），
   这样同一份脚本能同时校验 F / R / H 三臂。
4. 证据条数：首轮取 `search.per_query[0].gated`，末轮取 `reflection.evidence_count`；
   中间轮记录里没有证据条数 → 按"有证据"处理并计入 unknown_evidence，
   **不把"不可复算"说成"复算一致"**。
   注意不能用 `search.gated` 当证据数：v2.4 有 2 份记录该字段为 0、实际有 6 条证据，
   用它会把 2 个本来一致的轮次误报成翻转。
5. 期望：flips == 0（v2.5 第一步）。

第二步之后的预期（v2.5 第二步：R2 由 local 改成 rewrite）
--------------------------------------------------
`_rule_and_desired` 的 R2 分支改成 rewrite 之后，再用 v2.4 记录复算，翻转数**应当**
恰为 R2 的决策轮数：F/H 两臂各 5 轮 `local → rewrite`，合计 10；P-R 臂记录里本来就是
free（它一律补检索），故不翻转。此时改用 `--expect-flips 10` 跑，退出码 0 即说明
第二步只动了 R2 这一条规则、其余 36 轮一字未变。
"""
import argparse
import collections
import glob
import json
import os
import sys
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agents.nodes as nodes          # noqa: E402

RULE_ORDER = ("R1_evidence_gap", "R2_answer_gap", "R3_structure",
              "R4_repeat", "R5_infra")


def load_records(run_dir):
    """读入 run_dir/<臂>/*.json；坏文件跳过并提示，避免整批复算被一个文件打断。"""
    records = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[跳过] {path}: {e}")
            continue
        rec["_path"] = path
        records.append(rec)
    return records


def evidence_count(rec, idx, last):
    """返回 (证据条数, 是否可复算)。只用记录里真实存在的字段，不做无依据假设。"""
    if idx == 0:
        per_query = ((rec.get("search") or {}).get("per_query") or [])
        if per_query and per_query[0].get("gated") is not None:
            return int(per_query[0]["gated"]), True
    if idx == last:
        ev = (rec.get("reflection") or {}).get("evidence_count")
        if ev is not None:
            return int(ev), True
    return 1, False          # 中间轮：按"有证据"处理，单独计数


def replay_record(rec):
    """复算一份记录的所有决策轮。

    返回 [{"round", "verdict", "recorded", "got", "rule", "known", "source",
           "repeat_hit", "evidence"}]；终态轮不计入（另由调用方统计）。
    """
    policy = ((rec.get("config") or {}).get("repair_policy") or "hybrid")
    rounds = (rec.get("repair") or {}).get("rounds") or []
    last = len(rounds) - 1
    out = []
    for idx, rnd in enumerate(rounds):
        if rnd.get("pass_kind") != "rejected":
            continue                     # 判定通过的收尾轮：path 不是决策输出
        ev_n, known = evidence_count(rec, idx, last)
        etype = nodes._norm_type((rnd.get("verdict") or "").strip())
        # repeat_hit 落盘即"重复规则真的被触发"，与 production 的 repeat_fired 同义
        repeat_fired = bool(rnd.get("repeat_hit"))
        with mock.patch.object(nodes, "REPAIR_POLICY", policy):
            desired, rule, _ = nodes._rule_and_desired(
                None, etype, rnd.get("reason") or "", repeat_fired,
                facts={"evidence_count": ev_n})
            got = nodes._apply_repair_policy(desired)
        out.append({"round": idx, "verdict": rnd.get("verdict"),
                    "recorded": rnd.get("path") or "", "got": got,
                    "rule": rule, "known": known,
                    "source": rnd.get("decision_source") or "",
                    "repeat_hit": repeat_fired, "evidence": ev_n})
    return out


def main():
    ap = argparse.ArgumentParser(
        description="v2.5 规则表离线回放（路径零翻转验收）")
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "runs_v3"),
                    help="记录目录，默认 data/runs_v3")
    ap.add_argument("--verbose", action="store_true", help="逐轮打印复算结果")
    ap.add_argument("--expect-flips", type=int, default=0,
                    help="期望的翻转轮数，默认 0；不符时退出码 1")
    args = ap.parse_args()

    # 第二步（规则表落地）之前跑这份脚本没有意义：明确提示而不是抛 AttributeError
    if not hasattr(nodes, "_rule_and_desired") or not hasattr(nodes, "REPAIR_RULES"):
        print("[未就绪] agents/nodes.py 里还没有 REPAIR_RULES / _rule_and_desired ——"
              "请先落地 v2.5 第一步代码。")
        return 3

    records = load_records(args.dir)
    if not records:
        print(f"[空] {args.dir} 下没有 */ *.json 记录")
        return 2

    stats = collections.Counter()
    by_arm = collections.Counter()
    flips = []
    for rec in records:
        arm = rec.get("group") or os.path.basename(
            os.path.dirname(rec.get("_path") or ""))
        rounds = (rec.get("repair") or {}).get("rounds") or []
        if not rounds:
            stats["records_without_rounds"] += 1
            continue
        stats["records"] += 1
        last = len(rounds) - 1
        stats["terminal_rounds"] += sum(
            1 for r in rounds if r.get("pass_kind") != "rejected")
        if rounds[last].get("pass_kind") == "rejected":
            stats["maxed_rounds"] += 1     # 达 repair_max_rounds，决策写出但未执行
        for item in replay_record(rec):
            stats["decision_rounds"] += 1
            stats["rule_" + item["rule"]] += 1
            by_arm[(arm, item["rule"])] += 1
            if not item["known"]:
                stats["unknown_evidence"] += 1
            ok = item["got"] == item["recorded"]
            stats["match" if ok else "flips"] += 1
            if not ok:
                flips.append((arm, (rec.get("topic") or "")[:20],
                              item["round"], item["verdict"],
                              item["source"], item["recorded"],
                              item["got"], item["rule"]))
            if args.verbose:
                print(f"  [{'OK' if ok else 'FLIP'}] {arm} "
                      f"{(rec.get('topic') or '')[:16]} 轮{item['round']} "
                      f"{item['verdict']} 记录={item['recorded'] or '-'} "
                      f"复算={item['got']} 规则={item['rule']} "
                      f"证据={item['evidence']}{'' if item['known'] else '(不可复算)'}")

    arms = sorted({k[0] for k in by_arm} | {
        (r.get("group") or "") for r in records if r.get("group")})
    print(f"\n数据源            {args.dir}")
    print(f"记录数            {len(records)}（参与复算 {stats['records']}）")
    print(f"决策轮            {stats['decision_rounds']}"
          f"（复算一致 {stats['match']} / 翻转 {stats['flips']}）")
    print(f"终态轮            {stats['terminal_rounds']}"
          f"（通过/降级收尾，path 非决策输出，不参与复算）")
    print(f"达上限未执行      {stats['maxed_rounds']}"
          f"（末轮 rejected：决策已写出、无后续执行，仍属 _decide_next 输出）")
    print("规则分布          " + "  ".join(
        f"{k}={stats.get('rule_' + k, 0)}" for k in RULE_ORDER))
    print(f"证据数不可复算    {stats['unknown_evidence']}（按\"有证据\"处理）")

    if by_arm:
        w = 6
        print("\n规则 × 臂分布：")
        print("  " + "臂".ljust(8) + "".join(k[:2].rjust(w + 2) for k in RULE_ORDER))
        for arm in arms:
            if not arm:
                continue
            print("  " + str(arm).ljust(8) + "".join(
                str(by_arm[(arm, k)]).rjust(w + 2) for k in RULE_ORDER))

    if flips:
        print("\n翻转明细（臂, 题, 轮, 判定, 判据来源, 记录path, 复算path, 规则）：")
        for x in flips:
            print("  ", x)
        if stats["flips"] == args.expect_flips:
            print(f"\n结论：翻转 {stats['flips']} 处，与 --expect-flips 的预期一致"
                  "（v2.5 第二步：R2 由 local 改成 rewrite，只有这一条规则变了）。")
        else:
            print("\n结论：发现翻转 —— 规则命名改变了历史路径，需回退后再复算。")
    else:
        print("\n结论：零翻转 —— 规则命名未改变任何历史路径，第一步可安全合入。")

    return 1 if stats["flips"] != args.expect_flips else 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""v2.4 修复路径四臂对照执行器（17 任务 × 4 臂 = 68 次运行）。

用法：
    python experiments/repair_matrix.py --dry-run                # 只打印将执行的 68 条命令
    python experiments/repair_matrix.py --arms H                 # 只跑 hybrid 臂（预跑筛题）
    python experiments/repair_matrix.py --tasks D1,D2,D3         # 只跑指定任务（v2.5 的 D 组专测 rewrite）
    python experiments/repair_matrix.py --arms F,R --skip-existing
    python experiments/repair_matrix.py --timeout 1200

四臂只靠环境变量区分（代码零改动），因此"同一份代码跑出四臂"可复现：
    B0  ENABLE_REFLECTION=0        无反思基线（判据直接判通过，缺陷不会被修）
    F   REPAIR_POLICY=fixed        固定类型映射：只按错误类型修，永不换检索式重写
    R   REPAIR_POLICY=free         自由反思：一律补检索+重写，不做定向修复
    H   REPAIR_POLICY=hybrid       本文方法：有确定性依据走定向，判断不清走自由反思

除臂开关外，每个任务还会注入三项**任务级**变量（三臂完全一致，比较才公平）：
    INJECT_FAULT     首轮受控缺陷（none/short/no_cite/truncate/one_side），只作用于第 0 轮
    REQUIRED_OBJECTS tasks_repair.json 的 required_objects —— 判定与判卷共用的必答对象，
                     覆盖率终判因此看"答案有没有讲到"，而不是"子任务串能不能在证据里逐字命中"

结果写入 data/runs_v4/P-<臂>/（v2.5 起的新口径目录；**不写 v2 的 data/runs_v2/**，
v2.4 及更早的 48 份记录已归档到 data/runs_v3_pre_v25/，判定口径变了不可混表），
汇总表用 `python experiments/evaluate_answers.py --dir data/runs_v4 --out docs/修复路径对照_v2.5.md` 生成。
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "data", "runs_v4")
TASKS_PATH = os.path.join(ROOT, "experiments", "tasks_repair.json")

# 基线环境：v2.5 的新记录一律进 data/runs_v4；温度归零，避免抖动影响"路径选择"这一被解释变量
BASE_ENV = {
    "RUN_RECORD_DIR": os.path.join("data", "runs_v4"),
    "LLM_TEMPERATURE": "0",
    "REPAIR_MAX_ROUNDS": "3",
}

ARMS = {
    "B0": {"ENABLE_REFLECTION": "0", "REPAIR_POLICY": "hybrid"},
    "F": {"REPAIR_POLICY": "fixed"},
    "R": {"REPAIR_POLICY": "free"},
    "H": {"REPAIR_POLICY": "hybrid"},
}
ARM_DESC = {
    "B0": "无反思基线",
    "F": "固定类型修复",
    "R": "自由反思",
    "H": "混合（本文）",
}


def load_tasks() -> list:
    with open(TASKS_PATH, encoding="utf-8") as f:
        return json.load(f)["tasks"]


def _record_for(group: str, topic: str) -> dict:
    """按主题精确匹配取该组最新一份记录（文件名排序不可靠：中文主题与运行顺序无关）。"""
    files = glob.glob(os.path.join(RUN_DIR, group, "*.json"))
    best, best_ts = {}, ""
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("topic") != topic:
            continue
        if (rec.get("ts") or "") >= best_ts:
            best, best_ts = rec, rec.get("ts") or ""
    return best


def _plan(arms: list, tasks: list) -> list:
    return [(arm, task) for task in tasks for arm in arms]


def run_one(arm: str, task: dict, timeout: int = 1200, dry_run: bool = False) -> bool:
    group = f"P-{arm}"
    env_over = {**ARMS[arm], "INJECT_FAULT": task["inject"],
                # v2.4：判定用与判卷同源的必答对象，覆盖率终判才不会被"子任务串匹配"污染
                "REQUIRED_OBJECTS": ",".join(task.get("required_objects") or [])}
    cmd = [sys.executable, "main.py", task["topic"]]
    print(f"\n[RUN] {group} | {task['id']} | {task['topic']}")
    print("      env: " + " ".join(f"{k}={v}" for k, v in env_over.items()))
    if dry_run:
        print(f"      cmd: {' '.join(cmd)}")
        return True

    env = {**os.environ, **BASE_ENV, **env_over, "EXP_GROUP": group}
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"      [失败] returncode={proc.returncode}，stderr 末尾：")
        print("      " + (proc.stderr or "")[-800:].replace("\n", "\n      "))
        return False

    rec = _record_for(group, task["topic"])
    refl = rec.get("reflection") or {}
    repair = rec.get("repair") or {}
    # 打印「判定/路径」链路与运行时注入标记：
    # 注入失败（任务写着 truncate 但没真注入）在跑的时候就该看见，而不是事后对表才发现
    rounds = repair.get("rounds") or []
    chain = " → ".join(
        f"{x.get('verdict')}{('/' + x['path']) if x.get('path') else ''}" for x in rounds) or "-"
    print(f"      [完成] {elapsed:.1f}s | outcome={repair.get('outcome')} | {chain} "
          f"| 注入={rec.get('injected')} | replan={rec.get('replan_count')} "
          f"| llm_calls={refl.get('llm_calls')}")
    return True


def _select(ids: str, tasks: list) -> list:
    wanted = [x.strip().upper() for x in (ids or "").split(",") if x.strip()]
    if not wanted:
        return tasks
    picked = [t for t in tasks if t["id"].upper() in wanted]
    missing = sorted(set(wanted) - {t["id"].upper() for t in picked})
    if missing:
        print(f"[警告] 任务集里没有这些编号：{', '.join(missing)}")
    return picked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v2.4 修复路径四臂对照（17 任务 × 4 臂）")
    parser.add_argument("--arms", default=",".join(ARMS),
                        help="逗号分隔，默认 B0,F,R,H")
    parser.add_argument("--tasks", default="", help="逗号分隔任务编号，如 A1,B2,D3；留空=全部 17 条")
    parser.add_argument("--timeout", type=int, default=1200, help="单次运行超时（秒）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    parser.add_argument("--skip-existing", action="store_true",
                        help="该臂该题已有记录则跳过（重跑前先删旧记录，避免混入两次结果）")
    args = parser.parse_args(argv)

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"未知臂：{', '.join(unknown)}（可选 {','.join(ARMS)}）")
    tasks = _select(args.tasks, load_tasks())
    plan = _plan(arms, tasks)

    print(f"[计划] {len(tasks)} 任务 × {len(arms)} 臂 = {len(plan)} 次运行")
    print(f"[记录] {os.path.relpath(RUN_DIR, ROOT)}/P-<臂>/   "
          f"温度={BASE_ENV['LLM_TEMPERATURE']} 轮次上限={BASE_ENV['REPAIR_MAX_ROUNDS']}")
    for arm in arms:
        print(f"        {arm:<3} {ARM_DESC[arm]:<10} {ARMS[arm]}")

    ok = skipped = 0
    for arm, task in plan:
        if args.skip_existing and not args.dry_run and \
                _record_for(f"P-{arm}", task["topic"]):
            print(f"[SKIP] P-{arm} | {task['id']} 已有记录")
            skipped += 1
            continue
        try:
            ok += 1 if run_one(arm, task, timeout=args.timeout,
                               dry_run=args.dry_run) else 0
        except subprocess.TimeoutExpired:
            print(f"      [超时] 超过 {args.timeout}s，跳过")

    if args.dry_run:
        print(f"\n[干跑] 共 {len(plan)} 条命令，未执行。去掉 --dry-run 即开始跑。")
    else:
        print(f"\n[汇总] 成功 {ok} / 计划 {len(plan)}，跳过 {skipped}")
        print("       下一步：python experiments/evaluate_answers.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""E1~E6 批量实验执行器（v2 判定口径）。

用法：
    python experiments/run_matrix.py                       # 6 组 × 默认 4 主题
    python experiments/run_matrix.py --groups E1,E3        # 只跑指定组
    python experiments/run_matrix.py --groups E3 --topics "生物细胞的结构"
    python experiments/run_matrix.py --dry-run             # 只打印将执行的命令

每组只靠环境变量区分（代码零改动），因此"同一份代码跑出 6 组"可复现：
    E1  ENABLE_REFLECTION=0                       无反思基线
    E2  DISABLE_CONTENT_GATE=1 + 关覆盖率/重复规则  纯 LLM 反思
    E3  （默认）                                   本文方法
    E4  ENABLE_COVERAGE_RULE=0                    消融·覆盖率
    E5  ENABLE_REPEAT_RULE=0                      消融·重复检测
    E6  INJECT_LLM_FAIL=3                         故障注入

结果由 run_store 写入 data/runs_v2/<组>/（**不写 v1 的 data/runs/**），
汇总表格用 `python experiments/build_v2_table.py` 生成。
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "data", "runs_v2")

TOPICS = [
    "生物细胞的结构",                                   # 单主题：正常路径
    "比较Python和Java在并发编程与内存管理上的差异",       # 复合主题：覆盖率规则
    "量子计算和经典计算的区别",                          # 复合主题：对称句
    "比较Zig和Nim在内存安全与编译期检查上的差异",         # 易 0 证据：降级路径
]

GROUPS = {
    "E1": {"ENABLE_REFLECTION": "0"},
    "E2": {"DISABLE_CONTENT_GATE": "1",
           "ENABLE_COVERAGE_RULE": "0",
           "ENABLE_REPEAT_RULE": "0"},
    "E3": {},
    "E4": {"ENABLE_COVERAGE_RULE": "0"},
    "E5": {"ENABLE_REPEAT_RULE": "0"},
    "E6": {"INJECT_LLM_FAIL": "3"},
}


def _latest_record(group: str) -> dict:
    files = glob.glob(os.path.join(RUN_DIR, group, "*.json"))
    if not files:
        return {}
    # 按修改时间取最新，不能按文件名排序（中文主题的排序结果与运行顺序无关）
    with open(max(files, key=os.path.getmtime), encoding="utf-8") as f:
        return json.load(f)


def run_one(group: str, topic: str, timeout: int = 900) -> bool:
    env = {**os.environ, **GROUPS[group], "EXP_GROUP": group}
    cmd = [sys.executable, "main.py", topic]
    print(f"\n[RUN] {group} | {topic}")
    print(f"      env: {' '.join(f'{k}={v}' for k, v in GROUPS[group].items()) or '(默认)'}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"      [失败] returncode={proc.returncode}，stderr 末尾：")
        print("      " + (proc.stderr or "")[-800:].replace("\n", "\n      "))
        return False

    rec = _latest_record(group)
    refl = rec.get("reflection") or {}
    print(f"      [完成] {elapsed:.1f}s | verdict={refl.get('verdict')} "
          f"| decision_source={refl.get('decision_source')} "
          f"| replan={rec.get('replan_count')} | cited={refl.get('cited_count')} "
          f"| llm_calls={refl.get('llm_calls')}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="E1~E6 批量实验（v2 口径）")
    parser.add_argument("--groups", default=",".join(GROUPS),
                        help="逗号分隔，默认 E1,E2,E3,E4,E5,E6")
    parser.add_argument("--topics", default="",
                        help="用 | 分隔自定义主题；留空用默认 4 条")
    parser.add_argument("--timeout", type=int, default=900, help="单次运行超时（秒）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = parser.parse_args(argv)

    groups = [g.strip().upper() for g in args.groups.split(",") if g.strip()]
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        print(f"未知实验组：{unknown}（可选 {list(GROUPS)}）")
        return 2
    topics = ([t.strip() for t in args.topics.split("|") if t.strip()]
              if args.topics else TOPICS)

    print(f"实验组：{groups}")
    print(f"主题（{len(topics)}）：{topics}")
    print(f"记录目录：{os.path.relpath(RUN_DIR, ROOT)}/<组>/")
    if args.dry_run:
        for g in groups:
            for t in topics:
                env = " ".join(f"{k}={v}" for k, v in GROUPS[g].items())
                print(f"  [DRY] EXP_GROUP={g} {env} python main.py \"{t}\"")
        return 0

    ok, fail = 0, []
    t_start = time.perf_counter()
    for g in groups:
        for t in topics:
            if run_one(g, t, timeout=args.timeout):
                ok += 1
            else:
                fail.append((g, t))
    print(f"\n=== 批量结束：成功 {ok} / 失败 {len(fail)}，"
          f"总耗时 {(time.perf_counter() - t_start) / 60:.1f} 分钟 ===")
    for g, t in fail:
        print(f"  失败：{g} | {t}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())

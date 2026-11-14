# -*- coding: utf-8 -*-
"""把 data/runs_v3/ 收敛成「每个 (臂, 题) 只留最新一份」的正式批次。

用法：
    python experiments/prune_runs.py --dry-run            # 只列会被修剪的旧记录（不动文件）
    python experiments/prune_runs.py --apply              # 旧记录先移到备份目录，再离开正式目录
    python experiments/prune_runs.py --apply --no-backup  # 直接删（确认别处已有备份时）
    python experiments/prune_runs.py --dir data/runs_v3_v24a --dry-run

为什么需要：`repair_matrix.py` 重跑同一道题是**追加新文件**、不覆盖旧文件（文件名带时间戳），
所以跑过几轮后目录里会同时留着 v2.4a 与 v2.4 的记录（例：67 份 = 48 正式 + 19 旧）。
`evaluate_answers.py` 靠「同 (臂, 主题) 取最新」才凑出 48 份分母，表里的数字不受影响；
但目录会膨胀，且提交 GitHub 前应当只有正式批次。

本脚本的分组 key 与 `evaluate_answers.load_records()` 完全一致（group 去 P- 前缀 + topic，ts 取最大），
所以「删掉的就是判卷本来就会跳过的那几份」。剪枝前后各跑一次
`python experiments/evaluate_answers.py`，汇总表应逐字节相同（只有生成时间会变）。
"""
import argparse
import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, "data", "runs_v3")
DEFAULT_BACKUP = os.path.join(ROOT, "data", "runs_v3_superseded")


def key_of(path, rec):
    """与 evaluate_answers.load_records() 同口径：(臂, 主题)。"""
    group = (rec.get("group") or "").strip()
    arm = group[2:] if group.startswith("P-") else group
    return (arm or os.path.basename(os.path.dirname(path)),
            rec.get("topic") or os.path.basename(path))


def scan(run_dir):
    """返回 (保留清单, 待修剪清单)，保留清单为 {(臂, 题): (path, rec)}。"""
    picked, stale = {}, []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"[跳过] 解析失败：{os.path.relpath(path, ROOT)}")
            continue
        k = key_of(path, rec)
        prev = picked.get(k)
        if prev and (prev[1].get("ts") or "") >= (rec.get("ts") or ""):
            stale.append(path)
            continue
        if prev:
            stale.append(prev[0])
        picked[k] = (path, rec)
    return picked, stale


def main():
    ap = argparse.ArgumentParser(description="按 (臂, 题) 只保留最新一份运行记录")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="记录目录（默认 data/runs_v3）")
    ap.add_argument("--dry-run", action="store_true", help="只打印、不改动（默认行为，可显式写）")
    ap.add_argument("--apply", action="store_true", help="真删；不带则只打印（dry-run）")
    ap.add_argument("--backup", default=DEFAULT_BACKUP, help="旧记录移入的备份目录")
    ap.add_argument("--no-backup", action="store_true", help="直接删除，不备份")
    args = ap.parse_args()

    total = len(glob.glob(os.path.join(args.dir, "*", "*.json")))
    picked, stale = scan(args.dir)
    print(f"目录 {os.path.relpath(args.dir, ROOT)}：共 {total} 份，"
          f"{len(picked)} 个 (臂, 题) 组，保留 {len(picked)} 份，待修剪 {len(stale)} 份")
    for p in stale:
        print(f"  - {os.path.relpath(p, ROOT)}")
    if not stale:
        print("[完成] 已经是干净批次，无需修剪。")
        return 0
    if not args.apply:
        print("\n[dry-run] 未改动任何文件；加 --apply 才会落实。")
        return 0

    for p in stale:
        if args.no_backup:
            os.remove(p)
            continue
        dst = os.path.join(args.backup, os.path.relpath(p, args.dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(p, dst)
    tail = "已直接删除" if args.no_backup else f"已移到 {os.path.relpath(args.backup, ROOT)}/"
    print(f"\n[完成] {len(stale)} 份旧记录{tail}；目录现剩 {len(picked)} 份。")
    print("下一步：python experiments/evaluate_answers.py   # 复核汇总表数字未变")
    return 0


if __name__ == "__main__":
    sys.exit(main())
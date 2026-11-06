# -*- coding: utf-8 -*-
"""运行产物的文本归档（替代终端截图，保证实验可复现）

- logs/<主题>-<时间戳>.log       : 全链路控制台输出（Tee 双写，终端观感不变）
- data/runs/<主题>-<时间戳>.json : 结构化运行记录（论文表格里的数字唯一来源）

两者默认不入库（见 .gitignore）；需要作为附录的精选日志请拷到 experiments/ 下。
"""
import json
import os
import platform
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict

from config import (CRITIC_MODEL, DISABLE_CONTENT_GATE, ENABLE_COVERAGE_RULE,
                    ENABLE_DEGRADE_RESCUE, ENABLE_REFLECTION, ENABLE_REPEAT_RULE,
                    ENABLE_TRUNCATION_RULE, EVIDENCE_TOP_K,
                    EXP_GROUP, GENERATOR_MODEL, INJECT_LLM_FAIL, PASS_MIN_CHARS,
                    PASS_MIN_CITATION_RATIO, PASS_MIN_CITED, RETRY_QUERY_MODE,
                    RUN_LOG_DIR, RUN_RECORD_DIR,
                    SEARCH_COUNT, SEARCH_NEED_CONTENT, SEARCH_PROVIDER)


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_project_root(), path)


def _slug(text: str, limit: int = 24) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text).strip("_")
    return (text[:limit] or "topic")


class _Tee:
    """把写入同时送到终端与日志文件；单个流异常不影响另一个。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _header(topic: str) -> str:
    return "\n".join([
        "# 运行日志（自动生成，可直接作为论文附录引用）",
        f"- 主题：{topic}",
        f"- 实验组：{EXP_GROUP}",
        f"- 开始时间：{datetime.now(timezone.utc).isoformat()}",
        f"- 命令行：{' '.join(sys.argv)}",
        f"- 环境：Python {platform.python_version()} / {platform.system()}",
        f"- 检索：provider={SEARCH_PROVIDER} count={SEARCH_COUNT} "
        f"need_content={SEARCH_NEED_CONTENT}",
        f"- 模型：generator={GENERATOR_MODEL} critic={CRITIC_MODEL}",
        f"- 判定开关：reflection={ENABLE_REFLECTION} coverage_rule={ENABLE_COVERAGE_RULE} "
        f"repeat_rule={ENABLE_REPEAT_RULE} disable_gate={DISABLE_CONTENT_GATE} "
        f"inject_fail={INJECT_LLM_FAIL} truncation={ENABLE_TRUNCATION_RULE} "
        f"rescue={ENABLE_DEGRADE_RESCUE} min_citation_ratio={PASS_MIN_CITATION_RATIO}",
        "-" * 64,
        "",
    ])


@contextmanager
def run_log(topic: str):
    """用法：with run_log(topic) as log_path: ...；退出时自动恢复 stdout/stderr。"""
    base = _abs(RUN_LOG_DIR)
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"{_slug(topic)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    handle = open(path, "w", encoding="utf-8")
    handle.write(_header(topic))
    handle.flush()

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_stdout, handle)
    sys.stderr = _Tee(old_stderr, handle)
    try:
        yield path
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        try:
            handle.write("\n" + "-" * 64 +
                         f"\n# 结束时间：{datetime.now(timezone.utc).isoformat()}\n")
        finally:
            handle.close()


def save_run_record(topic: str, state: Dict[str, Any], log_path: str,
                    elapsed_s: float) -> str:
    """把一次运行的配置、检索/反思统计与最终结果写成 JSON，供论文与 README 引用。

    目录按实验组隔离：data/runs_v2/<E1..E6>/，避免不同判定口径的记录混在同一张表里。
    """
    base = os.path.join(_abs(RUN_RECORD_DIR), _slug(EXP_GROUP))
    os.makedirs(base, exist_ok=True)
    stats = state.get("run_stats") or {}
    record = {
        "topic": topic,
        "group": EXP_GROUP,
        "ts": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_s, 2),
        "config": {
            "group": EXP_GROUP,
            "search_provider": SEARCH_PROVIDER,
            "search_count": SEARCH_COUNT,
            "generator_model": GENERATOR_MODEL,
            "critic_model": CRITIC_MODEL,
            # 判定语义开关（E1~E6 的 A/B 变量）：读 JSON 即可复现这一组
            "enable_reflection": ENABLE_REFLECTION,
            "enable_coverage_rule": ENABLE_COVERAGE_RULE,
            "enable_repeat_rule": ENABLE_REPEAT_RULE,
            "disable_content_gate": DISABLE_CONTENT_GATE,
            "inject_llm_fail": INJECT_LLM_FAIL,
            "pass_min_cited": PASS_MIN_CITED,
            "pass_min_chars": PASS_MIN_CHARS,
            "evidence_top_k": EVIDENCE_TOP_K,
            "retry_query_mode": RETRY_QUERY_MODE,
            # v2.2 双向门控：这两项决定「否决能否被改判降级」，必须随记录落盘
            "enable_truncation_rule": ENABLE_TRUNCATION_RULE,
            "enable_degrade_rescue": ENABLE_DEGRADE_RESCUE,
            "pass_min_citation_ratio": PASS_MIN_CITATION_RATIO,
        },
        "search": stats.get("search", {}),
        "reflection": stats.get("reflection", {}),
        "replan_count": state.get("replan_count", 0),
        "subtasks": state.get("subtasks", []),
        "used_fallback": state.get("used_fallback", False),
        "sources": state.get("sources", []),
        "result": state.get("result", ""),
        "artifacts": {
            "log": os.path.relpath(log_path),
            "evidence": stats.get("search", {}).get("evidence_path", ""),
        },
    }
    path = os.path.join(base, f"{_slug(topic)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"  -> [记录] {os.path.relpath(path)}")
    return path

# -*- coding: utf-8 -*-
"""检索结果的磁盘缓存与证据归档

- data/search_cache/<sha1>.json : Search API 原始响应（可复现、可写论文附录）
- data/evidence/<slug>-<ts>.md  : 装配好的证据块（含来源链接）
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

from config import SEARCH_STORE_DIR, EVIDENCE_STORE_DIR


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_project_root(), path)


def store_key(query: str, provider: str, count: int, need_content: bool) -> str:
    """与 check_search.py 保持完全一致的 key 规则。"""
    raw = f"{provider}|{query}|{count}|{int(bool(need_content))}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _store_path(key: str) -> str:
    base = _abs(SEARCH_STORE_DIR)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{key}.json")


def save_raw(query: str, provider: str, results: List[dict],
             count: int, need_content: bool) -> str:
    path = _store_path(store_key(query, provider, count, need_content))
    payload = {
        "query": query,
        "provider": provider,
        "count": count,
        "need_content": bool(need_content),
        "ts": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_raw(query: str, provider: str, count: int,
             need_content: bool) -> Optional[List[dict]]:
    path = _store_path(store_key(query, provider, count, need_content))
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        results = payload.get("results")
        return results if isinstance(results, list) else None
    except Exception:
        return None


def _slug(text: str, limit: int = 24) -> str:
    text = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text).strip("_")
    return (text[:limit] or "topic")


def save_evidence(topic: str, context: str, sources: List[str]) -> str:
    base = _abs(EVIDENCE_STORE_DIR)
    os.makedirs(base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(base, f"{_slug(topic)}-{ts}.md")
    lines = [
        f"# 检索证据：{topic}",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).isoformat()}",
        f"- 来源条数：{len(sources)}",
        f"- 证据字数：{len(context)}",
        "",
        "## 来源清单",
        "",
    ]
    for s in sources:
        if " | " in s:
            title, url = s.split(" | ", 1)
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {s}")
    lines += ["", "## 证据全文", "", context, ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

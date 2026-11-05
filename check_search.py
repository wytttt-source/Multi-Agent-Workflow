# -*- coding: utf-8 -*-
"""DeepSeek Search / Fetch API 契约探针（自包含，不依赖项目内其它模块）

用途：在改动 tools.py 之前，先用真实 Key 验证
  1) 当前可用的模型名（防止 deepseek-chat 这类别名突然下线）
  2) 检索返回的字段是否齐全（summary / content 是否为空）
  3) rerank_score、authority 的实际分布，用于设定 EVIDENCE_MIN_RERANK

用法：
    python check_search.py --models
    python check_search.py "量子计算和经典计算的区别" --count 6 --save
    python check_search.py "https://example.com/article" --fetch
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.deepseek.com"
SEARCH_URL = f"{BASE_URL}/v1/tools/search"
FETCH_URL = f"{BASE_URL}/v1/tools/fetch"
MODELS_URL = f"{BASE_URL}/models"

# 与 search_store.py 保持完全一致的 key 规则（探针落盘的文件可被主程序缓存层直接命中）
STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "search_cache")


def _load_key() -> str:
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        print("[错误] 未找到 DEEPSEEK_API_KEY，请检查 .env")
        sys.exit(1)
    return key


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def store_key(query: str, provider: str, count: int, need_content: bool) -> str:
    raw = f"{provider}|{query}|{count}|{int(bool(need_content))}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def probe_models(key: str) -> None:
    print("=== GET /models ===")
    try:
        resp = requests.get(MODELS_URL, headers=_headers(key), timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[失败] {e}")
        return
    data = resp.json().get("data", [])
    print(f"可用模型（{len(data)} 个）：")
    for m in data:
        print(f"  - {m.get('id')}  (owned_by={m.get('owned_by')})")
    ids = {m.get("id") for m in data}
    for candidate in ("deepseek-flash", "deepseek-v4-pro"):
        print(f"  [{'OK' if candidate in ids else '缺失'}] {candidate}")


def probe_search(key: str, query: str, count: int, save: bool) -> None:
    payload = {
        "query": query,
        "count": count,
        "need_content": True,
        "ori_question": query,
    }
    print(f"=== POST /v1/tools/search ===\nquery={query!r}  count={count}  need_content=True")
    t0 = time.perf_counter()
    try:
        resp = requests.post(SEARCH_URL, headers=_headers(key), json=payload, timeout=40)
    except Exception as e:
        print(f"[失败] 网络异常: {e}")
        return
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        print(f"[失败] HTTP {resp.status_code}: {resp.text[:400]}")
        return

    results = resp.json().get("results", [])
    print(f"[成功] 耗时 {elapsed:.2f}s，返回 {len(results)} 条\n")

    header = f"{'#':<3}{'rerank':>7}{'auth':>6}{'summary':>9}{'content':>9}  title / url"
    print(header)
    print("-" * 92)
    for i, r in enumerate(results, 1):
        summary = r.get("summary") or ""
        content = r.get("content") or ""
        title = (r.get("title") or "")[:34]
        print(
            f"{i:<3}{r.get('rerank_score', 0):>7.2f}{r.get('authority', -1):>6}"
            f"{len(summary):>9}{len(content):>9}  {title}"
        )
        print(f"     {r.get('url', '')}")

    if results:
        s_lens = [len(r.get("summary") or "") for r in results]
        c_lens = [len(r.get("content") or "") for r in results]
        scores = [r.get("rerank_score") or 0 for r in results]
        print("\n=== 统计 ===")
        print(f"summary 平均长度: {sum(s_lens) / len(s_lens):.0f} 字（空值 {sum(1 for x in s_lens if x == 0)} 条）")
        print(f"content 平均长度: {sum(c_lens) / len(c_lens):.0f} 字（空值 {sum(1 for x in c_lens if x == 0)} 条）")
        print(f"rerank_score 区间: {min(scores):.2f} ~ {max(scores):.2f}，≥1.0 的有 {sum(1 for s in scores if s >= 1.0)} 条")
        authorities = [r.get("authority") for r in results if r.get("authority") is not None]
        if authorities:
            print(f"authority 区间: {min(authorities)} ~ {max(authorities)}")

    if save:
        os.makedirs(STORE_DIR, exist_ok=True)
        key_name = store_key(query, "deepseek", count, True)
        path = os.path.join(STORE_DIR, f"{key_name}.json")
        payload_out = {
            "query": query,
            "provider": "deepseek",
            "count": count,
            "need_content": True,
            "ts": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload_out, f, ensure_ascii=False, indent=2)
        print(f"\n[落盘] {path}")


def probe_fetch(key: str, url: str) -> None:
    print(f"=== POST /v1/tools/fetch ===\nurl={url}")
    t0 = time.perf_counter()
    try:
        resp = requests.post(FETCH_URL, headers=_headers(key), json={"url": url}, timeout=40)
    except Exception as e:
        print(f"[失败] 网络异常: {e}")
        return
    print(f"耗时 {time.perf_counter() - t0:.2f}s，HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"[失败] {resp.text[:400]}")
        return
    page = resp.json()
    print(f"title: {page.get('title')}")
    print(f"content 长度: {len(page.get('content') or '')}")
    print(f"content 前 300 字:\n{(page.get('content') or '')[:300]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek Search / Fetch API 契约探针")
    parser.add_argument("query", nargs="?", default="", help="检索词或待抓取的 URL")
    parser.add_argument("--models", action="store_true", help="列出当前可用模型")
    parser.add_argument("--count", type=int, default=6, help="检索条数（1~50，默认 6）")
    parser.add_argument("--save", action="store_true", help="把原始响应落盘到 data/search_cache/")
    parser.add_argument("--fetch", action="store_true", help="把 query 当作 URL 调用 Fetch 接口")
    args = parser.parse_args()

    key = _load_key()

    if args.models:
        probe_models(key)
        if not args.query:
            return

    if not args.query:
        parser.print_help()
        return

    if args.fetch:
        probe_fetch(key, args.query)
    else:
        probe_search(key, args.query, max(1, min(args.count, 50)), args.save)


if __name__ == "__main__":
    main()


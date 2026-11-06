#搜索与过滤工具
import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import List, Tuple
from urllib.parse import quote, urlparse

import requests

from config import (
    USE_LLM_RELEVANCE_FILTER,
    ENABLE_SEARCH_CACHE,
    SEARCH_PROVIDER,
    SEARCH_COUNT,
    SEARCH_NEED_CONTENT,
    SEARCH_TIMEOUT,
    USE_QUERY_EXPANSION,
    EVIDENCE_MIN_RERANK,
    EVIDENCE_SUMMARY_MAX,
    EVIDENCE_CONTENT_MAX,
    EVIDENCE_MAX_CHARS,
    ENABLE_EVIDENCE_ARCHIVE,
    api_key,
)
from utils import safe_llm_invoke, _SEARCH_CACHE
import prompts
import search_store

DEEPSEEK_SEARCH_URL = "https://api.deepseek.com/v1/tools/search"


class SearchAuthError(Exception):
    """API Key 缺失或无效（401）"""


class SearchQuotaError(Exception):
    """账号余额不足（402）"""


class SearchError(Exception):
    """其它检索错误（400/403/404/5xx/网络）"""


def _http_get(url: str, headers: dict, timeout: int = 10) -> requests.Response:
    return requests.get(url, headers=headers, timeout=timeout)


def _url_key(url: str) -> str:
    """用于去重的 URL 归一化：去掉协议、www、query、fragment、末尾斜杠。"""
    if not url:
        return ""
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = re.sub(r"/+$", "", p.path or "")
        return f"{host}{path}"
    except Exception:
        return url.strip()


def _norm_result(r: dict, provider: str) -> dict:
    """统一 DeepSeek / 必应的字段结构，下游无需关心来源。"""
    return {
        "title": (r.get("title") or "").strip(),
        "url": (r.get("url") or "").strip(),
        "snippet": (r.get("snippet") or "").strip(),
        "summary": (r.get("summary") or "").strip(),
        "content": (r.get("content") or "").strip(),
        "site_name": r.get("site_name"),
        "date": r.get("date"),
        "authority": r.get("authority"),
        "rerank_score": r.get("rerank_score"),
        "provider": provider,
    }


def search_deepseek(query: str, count: int = SEARCH_COUNT,
                    need_content: bool = SEARCH_NEED_CONTENT,
                    ori_question: str = None,
                    retries: int = 2, backoff: float = 2.0) -> List[dict]:
    """调用 DeepSeek Search API，返回结构化结果（含 summary / content / url）。"""
    if not api_key:
        raise SearchAuthError("未配置 DEEPSEEK_API_KEY")

    payload = {
        "query": query,
        "count": max(1, min(int(count), 50)),
        "need_content": bool(need_content),
    }
    if ori_question:
        payload["ori_question"] = ori_question[:128]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err = None

    for attempt in range(retries + 1):
        try:
            resp = requests.post(DEEPSEEK_SEARCH_URL, headers=headers, json=payload,
                                 timeout=SEARCH_TIMEOUT)
        except requests.RequestException as e:
            last_err = e
            if attempt < retries:
                wait = backoff * (2 ** attempt)
                print(f"  -> [DeepSeek检索] 网络异常，{wait:.1f}s 后重试：{str(e)[:60]}")
                time.sleep(wait)
                continue
            raise SearchError(f"网络异常: {e}")

        if resp.status_code == 200:
            data = resp.json()
            results = [_norm_result(r, "deepseek") for r in data.get("results", [])]
            print(f"  -> [DeepSeek检索] 命中 {len(results)} 条（查询词：{query}）")
            return results

        if resp.status_code == 401:
            raise SearchAuthError("API Key 无效（401）")
        if resp.status_code == 402:
            raise SearchQuotaError("账号余额不足（402）")

        if resp.status_code in (408, 429) or resp.status_code >= 500:
            last_err = f"HTTP {resp.status_code}"
            if attempt < retries:
                wait = backoff * (2 ** attempt)
                print(f"  -> [DeepSeek检索] {last_err}，{wait:.1f}s 后重试")
                time.sleep(wait)
                continue
            raise SearchError(f"服务暂不可用（{last_err}）")

        raise SearchError(f"HTTP {resp.status_code}: {resp.text[:120]}")

    raise SearchError(f"检索失败: {last_err}")


def build_query(topic: str) -> str:
    """中文主题追加 lang:zh；把语言限定交给 API，而不是手工堆关键词后缀。"""
    t = topic.strip()
    if re.search(r"[\u4e00-\u9fff]", t) and "lang:" not in t:
        return f"{t} lang:zh"
    return t


def expand_query(topic: str) -> str:
    t = topic.strip()
    extras = []
    if re.search(r"[\u4e00-\u9fff]", t):
        if any(k in t for k in ["结构", "组成", "构成"]):
            extras.append("组成 功能 特点")
        elif any(k in t for k in ["原理", "机制", "工作"]):
            extras.append("原理 机制 应用")
        elif any(k in t for k in ["区别", "对比", "差异"]):
            extras.append("区别 对比 异同")
        elif any(k in t for k in ["历史", "发展"]):
            extras.append("历史 发展 演变")
        elif any(k in t for k in ["应用", "用途"]):
            extras.append("应用 场景 案例")
        else:
            extras.append("定义 特点 应用")
    elif re.match(r"^[A-Za-z0-9\s\-]+$", t):
        if any(k in t.lower() for k in ["structure", "composition"]):
            extras.append("structure components function")
        elif any(k in t.lower() for k in ["principle", "mechanism"]):
            extras.append("principle mechanism application")
        else:
            extras.append("definition overview application")
    return f"{t} {' '.join(extras)}" if extras else t

def _parse_bing_results(html: str, top_n: int = 5) -> List[dict]:
    results = []
    blocks = re.findall(r'<li class="b_algo".*?</li>', html, re.S)
    for b in blocks[:top_n]:
        t = re.search(r'<h2.*?>(.*?)</h2>', b, re.S)
        p = re.search(r'<p.*?>(.*?)</p>', b, re.S)
        a = re.search(r'<h2.*?<a[^>]+href="([^"]+)"', b, re.S)
        title = re.sub(r'<.*?>', '', t.group(1)).strip() if t else ""
        snippet = re.sub(r'<.*?>', '', p.group(1)).strip() if p else ""
        url = a.group(1) if a else ""
        if title:
            results.append(_norm_result({"title": title, "snippet": snippet, "url": url}, "bing"))
    return results

def search_bing_cn(query: str) -> List[dict]:
    # 必应不认识 DeepSeek 的检索运算符，降级时先剔掉
    query = re.sub(r"\b(?:lang|site|after|before):[\w,\-]+", "", query).strip()
    url = f"https://cn.bing.com/search?q={quote(query)}&ensearch=0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    try:
        resp = _http_get(url, headers, timeout=10)
        resp.raise_for_status()
        results = _parse_bing_results(resp.text, top_n=5)
        if results:
            print(f"  -> [必应] 命中 {len(results)} 条（查询词：{query}）")
        else:
            print(f"  -> [必应] 未命中（查询词：{query}）")
        return results
    except Exception as e:
        print(f"  -> [必应] 失败（查询词：{query}）: {str(e)[:80]}")
        return []

def score_gate(results: List[dict], min_rerank: float = EVIDENCE_MIN_RERANK) -> List[dict]:
    """第①层过滤：结构化信号（排序分 / 同页去重 / 标题近似），不消耗 LLM 调用。"""
    if not results:
        return results
    kept, seen_url, seen_titles = [], set(), []
    for r in results:
        title = (r.get("title") or "").strip()
        score = r.get("rerank_score")
        if score is not None and score < min_rerank:
            print(f"  -> [过滤-分数] 剔除低排序分({score:.2f})：{title[:30]}")
            continue
        url_key = _url_key(r.get("url", ""))
        if url_key and url_key in seen_url:
            print(f"  -> [过滤-去重] 同站同页：{title[:30]}")
            continue
        if title and any(SequenceMatcher(None, title, t).ratio() >= 0.85 for t in seen_titles):
            print(f"  -> [过滤-去重] 标题近似：{title[:30]}")
            continue
        if url_key:
            seen_url.add(url_key)
        if title:
            seen_titles.append(title)
        kept.append(r)
    return kept

def rule_filter(topic: str, results: List[dict]) -> List[dict]:
    """【旧链路保留】改造前的正则规则过滤，当前主流程已由 score_gate() 取代。

    仅用于论文中"旧过滤策略"的 A/B 对照，不要在 multi_source_search 里重新调用。
    """
    if not results:
        return results
    topic_core = re.sub(
        r"(的)?(结构|组成|构成|原理|机制|区别|对比|历史|发展|应用|用途)$", "", topic
    ).strip()
    topic_chars = set(topic_core) - set("的了吗呢是在和与及或")
    kept = []
    for r in results:
        title = r.get("title", "")
        if len(title) <= 3 and not (set(title) & topic_chars):
            print(f"  -> [过滤-规则] 剔除泛词条：{title}")
            continue
        kept.append(r)
    return kept

def llm_relevance_filter(topic: str, results: List[dict],
                         field: str = "summary",
                         max_len: int = EVIDENCE_SUMMARY_MAX) -> List[dict]:
    """第②层过滤：GLM 异构裁判，基于长摘要判断是否切题（上位概念应判不相关）。"""
    if not results:
        return results
    items = "\n".join(
        f"{i+1}. 标题：{r.get('title', '')}\n"
        f"   长摘要：{(r.get(field) or r.get('snippet', ''))[:max_len]}"
        for i, r in enumerate(results)
    )
    prompt = prompts.FILTER_PROMPT.format(topic=topic, items=items)
    
    try:
        raw = safe_llm_invoke(prompt, use_critic=True).strip()
        if raw.startswith("```json"):
            raw = raw[7:-3].strip()
        elif raw.startswith("```"):
            raw = raw[3:-3].strip()
        arr = json.loads(raw)
        drop_idx = {a["idx"] - 1 for a in arr if a.get("judge") == "不相关"}
        kept = []
        for i, r in enumerate(results):
            if i in drop_idx:
                print(f"  -> [过滤-LLM] 剔除低相关：{r.get('title', '')[:40]}")
            else:
                kept.append(r)
        print(f"  -> [过滤-LLM] 保留 {len(kept)}/{len(results)} 条")
        return kept
    except Exception as e:
        print(f"  -> [过滤-LLM] 判断失败({str(e)[:60]})，保留全部结果")
        return results

def evidence_coverage(topic: str, results: List[dict]) -> dict:
    """确定性信号：复合主题里的比较对象是否都出现在证据中（喂给反思节点做量化判断）。

    返回字段：
    - terms / hit / missing / ratio：归一化字面覆盖（term 是否在全文任何位置出现）
    - per_term_counts：每个 term 在"标题 + 摘要前 300 字"里命中的证据条数
    - balance：最弱一侧命中条数 / 总条数，用于识别"偏侧"（字面覆盖 100% 但分布失衡）
    """
    core = _clean_for_coverage(topic)
    parts = [p.strip() for p in re.split(r"\s*(?:和|与|、|以及|对比|比较|vs|VS)\s*", core) if p.strip()]
    terms: List[str] = []
    for p in parts:
        t = _clean_term(p)
        if t and t not in terms:
            terms.append(t)
    norm = {t: _norm_text(t) for t in terms}

    # 字面覆盖：term 在全文任何位置出现即算命中（大小写/空白不敏感）
    blob = _norm_text(" ".join(
        " ".join([r.get("title") or "", r.get("summary") or "",
                  r.get("snippet") or "", (r.get("content") or "")[:200]])
        for r in results
    ))
    hit = [t for t in terms if norm[t] and norm[t] in blob]
    missing = [t for t in terms if t not in hit]
    ratio = len(hit) / len(terms) if terms else 1.0

    # 偏侧检测（P1-3）：标题 + 摘要前 300 字联合统计。
    # 只看标题会把"标题里没写对象名"的真实证据误判成偏侧。
    per_term_counts: dict = {}
    for t in terms:
        per_term_counts[t] = sum(
            1 for r in results
            if norm[t] and norm[t] in _norm_text(
                (r.get("title") or "") + " " + (r.get("summary") or "")[:300]))
    total = len(results) or 1
    weakest = min(per_term_counts.values()) if per_term_counts else total
    balance = round(weakest / total, 2)

    return {
        "terms": terms, "hit": hit, "missing": missing,
        "ratio": ratio, "count": len(results),
        "per_term_counts": per_term_counts,
        "balance": balance,
    }


def build_evidence(results: List[dict], max_chars: int = EVIDENCE_MAX_CHARS) -> Tuple[str, List[str]]:
    """第③层：装配证据块（长摘要 + 正文节选 + 可核验 URL），并返回来源清单。"""
    contexts, sources, used = [], [], 0
    for r in results:
        title = (r.get("title") or "").strip() or "(无标题)"
        url = (r.get("url") or "").strip()
        digest = (r.get("summary") or r.get("snippet") or "").strip()
        body = (r.get("content") or "").strip()
        meta = "，".join(x for x in [(r.get("site_name") or "").strip(),
                                     (r.get("date") or "")[:10]] if x)
        lines = [f"【{title}】" + (f"（{meta}）" if meta else "")]
        if url:
            lines.append(f"URL: {url}")
        if digest:
            lines.append(f"摘要: {digest[:EVIDENCE_SUMMARY_MAX]}")
        if body:
            lines.append(f"正文节选: {body[:EVIDENCE_CONTENT_MAX]}")
        text = "\n".join(lines)
        if contexts and used + len(text) > max_chars:
            print(f"  -> [证据] 达到 {max_chars} 字上限，截断剩余 {len(results) - len(contexts)} 条")
            break
        contexts.append(text)
        used += len(text)
        sources.append(f"{title} | {url}" if url else title)
    print(f"  -> [证据] 装配 {len(contexts)} 条，共 {used} 字")
    return "\n\n".join(contexts), sources


def search_with_evidence(topic: str, query: str = "") -> Tuple[str, List[str], List[dict], dict]:
    """检索编排：内存缓存 → 磁盘缓存 → 主检索（DeepSeek，失败降级必应）→ 三层过滤 → 证据装配。

    query 可显式指定检索式：补检索（自由反思）改写查询时必须传，
    否则会与原查询撞同一份缓存、等于没重试；留空则按 build_query(topic) 生成。

    返回 (证据文本, 来源清单, 结构化结果, 统计字典)：
    - 结构化结果落进 State["evidence"]，用于可复现记录与覆盖率计算；
    - 统计字典记录原始/过滤后条数、字数、缓存来源、耗时，用于落盘 data/runs/*.json。
    """
    t_start = time.perf_counter()
    query = (query or "").strip() or build_query(topic)
    cache_key = (topic, query, SEARCH_PROVIDER, SEARCH_COUNT, SEARCH_NEED_CONTENT,
                 USE_QUERY_EXPANSION, USE_LLM_RELEVANCE_FILTER)
    if ENABLE_SEARCH_CACHE and cache_key in _SEARCH_CACHE:
        context, sources, results, stats = _SEARCH_CACHE[cache_key]
        if not context and not results:
            # 空结果不视为有效缓存：删除并继续走真实检索，保证“补检索”能重试
            print(f"  -> [缓存] 命中空缓存，忽略并重新检索（{topic}）")
            del _SEARCH_CACHE[cache_key]
        else:
            print(f"  -> [缓存] 命中内存缓存（{topic}）")
            # cache_source 必须一起改写：executor 聚合 run_stats.search.cache_sources 时
            # 优先读 cache_source，只改 cache 会让"同进程命中"被记成 disk/network。
            return context, sources, results, {**stats, "cache": "memory",
                                               "cache_source": "memory"}
       

    print(f"  -> [检索] 主查询 '{query}'（provider={SEARCH_PROVIDER}）")

    results: List[dict] = []
    # P1-4：只有"真的没联网"或"真的联了网"才写对应值；空缓存不得冒充 disk
    cache_source = "network"
    if SEARCH_PROVIDER == "deepseek":
        if ENABLE_SEARCH_CACHE:
            cached = search_store.load_raw(query, "deepseek", SEARCH_COUNT, SEARCH_NEED_CONTENT)
            if cached:
                print(f"  -> [缓存] 命中磁盘缓存（{len(cached)} 条）")
                results = cached
                cache_source = "disk"
            elif cached is not None:
                print("  -> [缓存] 磁盘里只有空记录，忽略并重新联网检索")
        if not results:
            cache_source = "network"
            try:
                results = search_deepseek(query, ori_question=topic)
                if results and ENABLE_SEARCH_CACHE:
                    path = search_store.save_raw(query, "deepseek", results,
                                                 SEARCH_COUNT, SEARCH_NEED_CONTENT)
                    print(f"  -> [落盘] {os.path.relpath(path)}")
            except (SearchAuthError, SearchQuotaError, SearchError) as e:
                print(f"  -> [检索] DeepSeek 不可用（{str(e)[:60]}），降级到必应抓取")

    if not results:
        results = search_bing_cn(query)
        cache_source = "bing"

    if USE_QUERY_EXPANSION:
        expanded = expand_query(topic)
        print(f"  -> [检索] 二段查询（对照实验开关）：'{expanded}'")
        extra: List[dict] = []
        if SEARCH_PROVIDER == "deepseek":
            try:
                extra = search_deepseek(expanded, ori_question=topic)
            except (SearchAuthError, SearchQuotaError, SearchError) as e:
                print(f"  -> [检索] 二段查询失败：{str(e)[:60]}")
        if not extra:
            extra = search_bing_cn(expanded)
        results = results + extra

    raw_count = len(results)
    print(f"  -> [检索] 原始结果：{raw_count} 条")
    results = score_gate(results)
    gated_count = len(results)
    print(f"  -> [过滤] 结构化信号过滤后：{gated_count} 条")

    if results and USE_LLM_RELEVANCE_FILTER:
        results = llm_relevance_filter(topic, results)

    context, sources = build_evidence(results)
    archive_path = ""
    if context and ENABLE_EVIDENCE_ARCHIVE:
        path = search_store.save_evidence(topic, context, sources)
        archive_path = os.path.relpath(path)
        print(f"  -> [归档] {archive_path}")

    elapsed = time.perf_counter() - t_start
    print(f"  -> [检索] 完成，耗时 {elapsed:.2f}s")
    providers = sorted({r.get("provider", "?") for r in results})
    stats = {
        "topic": topic,
        "query": query,
        "raw": raw_count,
        "gated": gated_count,
        "filtered": len(results),
        "evidence_chars": len(context),
        "sources": len(sources),
        "avg_summary": round(sum(len(r.get("summary") or "") for r in results) / len(results), 1) if results else 0.0,
        "avg_content": round(sum(len(r.get("content") or "") for r in results) / len(results), 1) if results else 0.0,
        "providers": providers,
        "cache": cache_source,          # memory / disk / network / bing
        "cache_source": cache_source,   # P1-4：显式口径，论文统计只看这个字段
        "degraded": bool(SEARCH_PROVIDER == "deepseek" and "bing" in providers),
        "elapsed_s": round(elapsed, 2),
        "evidence_path": archive_path,
    }
    result = (context, sources, results, stats)
    if ENABLE_SEARCH_CACHE:
        if context or results:
            _SEARCH_CACHE[cache_key] = result
        else:
            print("  -> [缓存] 空结果不缓存，下次补检索将重新请求")
    return result


def multi_source_search(topic: str) -> Tuple[str, List[str]]:
    """兼容旧调用点：只要证据文本与来源清单。"""
    context, sources, _, _ = search_with_evidence(topic)
    return context, sources

def _is_composite_topic(topic: str) -> bool:
    return bool(re.search(r"(和|与|对比|比较|vs|VS)", topic))

def _clean_composite_topic(topic: str) -> str:
    """供 executor_node 使用：只剥掉开头的'比较/对比'，保留完整主题做检索词。"""
    return re.sub(r"^(比较|对比|分析|简述)\s*", "", topic).strip()


# ================= 覆盖率 term 清洗（P1-2） =================
# 这些尾巴属于"论述框架"，不是"比较对象"，必须剥掉，否则 term 永远不可能字面命中。
_TERM_TAIL_RE = re.compile(
    r"(?:的)?(?:并发)?(?:性能|表现|优劣势|优缺点|区别|差异|对比|特点|特征|"
    r"结构|组成|构成|原理|机制|历史|发展|应用|用途)$"
)
# "在…上的表现/性能/差异" 这类状语段（可能夹在 term 内部）
_TERM_ADVERBIAL_RE = re.compile(
    r"(?:在|于).+?(?:上|中|里)?的?(?:表现|性能|差异|区别|对比|比较|异同|优劣势|优缺点)$"
)
# 疑问尾巴："Go哪个更适合写后端" → "Go"
_TERM_QUESTION_RE = re.compile(r"(?:哪个|哪些|如何|怎么|怎样|为什么|是什么|有哪些|谁|多少).*$")


def _clean_term(part: str) -> str:
    """把切出来的一侧清洗成"对象名"：去疑问尾巴 → 去状语段 → 去框架后缀。"""
    t = _TERM_QUESTION_RE.sub("", part or "")
    t = _TERM_ADVERBIAL_RE.sub("", t)
    t = _TERM_TAIL_RE.sub("", t)
    return t.strip().strip("的").strip()


def _norm_text(text: str) -> str:
    """命中判定用的归一化：去空白 + 小写（Python / python / Py thon 视为同一 term）。"""
    return re.sub(r"\s+", "", text or "").lower()


def _clean_for_coverage(topic: str) -> str:
    """仅供 evidence_coverage 使用：剥掉'在...上的差异'结构与疑问尾巴，只留比较对象。
    不要用于 executor_node 的检索词，否则会让 query 变短、召回下降。"""
    t = re.sub(r"^(比较|对比|分析|简述|说说|讲讲)\s*", "", (topic or "").strip())
    t = re.sub(r"(?:在|于).+?(?:上|中|里)?的?(?:差异|区别|对比|比较|异同|表现|性能|优劣势|优缺点)$",
               "", t).strip()
    t = _TERM_QUESTION_RE.sub("", t).strip()
    return t
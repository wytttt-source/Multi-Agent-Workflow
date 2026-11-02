#搜索与过滤工具
import json
import re
import requests
from urllib.parse import quote
from typing import List, Tuple

from config import USE_LLM_RELEVANCE_FILTER, ENABLE_SEARCH_CACHE
from utils import safe_llm_invoke, _SEARCH_CACHE
import prompts

def _http_get(url: str, headers: dict, timeout: int = 10) -> requests.Response:
    return requests.get(url, headers=headers, timeout=timeout)

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

def _parse_bing_results(html: str, top_n: int = 3) -> List[dict]:
    results = []
    blocks = re.findall(r'<li class="b_algo".*?</li>', html, re.S)
    for b in blocks[:top_n]:
        t = re.search(r'<h2.*?>(.*?)</h2>', b, re.S)
        p = re.search(r'<p.*?>(.*?)</p>', b, re.S)
        title = re.sub(r'<.*?>', '', t.group(1)).strip() if t else ""
        snippet = re.sub(r'<.*?>', '', p.group(1)).strip() if p else ""
        if title:
            results.append({"title": title, "snippet": snippet})
    return results

def search_bing_cn(query: str) -> List[dict]:
    url = f"https://cn.bing.com/search?q={quote(query)}&ensearch=0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    try:
        resp = _http_get(url, headers, timeout=10)
        resp.raise_for_status()
        results = _parse_bing_results(resp.text, top_n=3)
        if results:
            print(f"  -> [必应] 命中 {len(results)} 条（查询词：{query}）")
        else:
            print(f"  -> [必应] 未命中（查询词：{query}）")
        return results
    except Exception as e:
        print(f"  -> [必应] 失败（查询词：{query}）: {str(e)[:80]}")
        return []

def rule_filter(topic: str, results: List[dict]) -> List[dict]:
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

def llm_relevance_filter(topic: str, results: List[dict]) -> List[dict]:
    if not results:
        return results
    items = "\n".join(
        f"{i+1}. 标题：{r['title']}\n   摘要：{r.get('snippet','')[:120]}"
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
                print(f"  -> [过滤-LLM] 剔除低相关：{r['title']}")
            else:
                kept.append(r)
        return kept
    except Exception as e:
        print(f"  -> [过滤-LLM] 判断失败({str(e)[:60]})，保留全部结果")
        return results

def multi_source_search(topic: str) -> Tuple[str, List[str]]:
    expanded = expand_query(topic)
    cache_key = (topic, expanded, USE_LLM_RELEVANCE_FILTER)
    if ENABLE_SEARCH_CACHE and cache_key in _SEARCH_CACHE:
        print(f"  -> [缓存] 命中，复用上次检索结果（{topic}）")
        return _SEARCH_CACHE[cache_key]

    raw_results = []
    seen_titles = set()

    print(f"  -> [检索] 第一段：原主题 '{topic}'")
    for r in search_bing_cn(topic):
        title = r.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            raw_results.append(r)

    print(f"  -> [检索] 第二段：扩展词 '{expanded}'")
    for r in search_bing_cn(expanded):
        title = r.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            raw_results.append(r)

    print(f"  -> [检索] 两段合并去重后：{len(raw_results)} 条")

    after_rule = rule_filter(topic, raw_results)
    print(f"  -> [过滤] 规则过滤后：{len(after_rule)} 条")

    if after_rule and USE_LLM_RELEVANCE_FILTER:
        final_results = llm_relevance_filter(topic, after_rule)
    else:
        final_results = after_rule

    print(f"  -> [过滤] 最终保留：{len(final_results)} 条")

    contexts, sources = [], []
    for r in final_results:
        title = r.get("title", "").strip()
        snippet = r.get("snippet", "").strip()
        if title:
            contexts.append(f"【{title}】: {snippet}")
            sources.append(f"必应 - {title}")

    result = ("\n".join(contexts), sources)
    if ENABLE_SEARCH_CACHE:
        _SEARCH_CACHE[cache_key] = result
    return result

def _is_composite_topic(topic: str) -> bool:
    return bool(re.search(r"(和|与|对比|比较|vs|VS)", topic))

def _clean_composite_topic(topic: str) -> str:
    return re.sub(r"^(比较|对比|分析|简述)\s*", "", topic).strip()
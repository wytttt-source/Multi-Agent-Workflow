import os
import json
import time
import re
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Tuple
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import requests
from urllib.parse import quote

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
glm_api_key = os.getenv("GLM_API_KEY")

# ================= 0. 全局配置 =================
USE_LLM_RELEVANCE_FILTER = True   # 关闭则只走规则过滤，用于对比实验
ENABLE_SEARCH_CACHE = True        # 检索结果内存缓存

# ================= 1. 定义状态 =================
class State(TypedDict):
    topic: str
    subtasks: List[str]
    result: str
    error_type: str
    replan_scope: str
    replan_count: int
    reflection_history: List[str]
    sources: List[str]
    used_fallback: bool
    llm_fail_count: int

# ================= 2. 初始化双模型架构 =================
# 2.1 执行/生成模型（Generator）：使用 DeepSeek 生成摘要
llm_generator = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com",
    temperature=0.1,
    timeout=30,
    max_retries=0,
)

# 2.2 反思/审查模型（Critic）：使用智谱 GLM-4.7-Flash（免费）
llm_critic = ChatOpenAI(
    model="glm-5.3-flash",
    openai_api_key=glm_api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
    timeout=30,
    max_retries=0,
)

# ================= 3. LLM 调用容错 =================
class LLMInfraError(Exception):
    pass

class LLMParseError(Exception):
    pass

def safe_llm_invoke(prompt: str, use_critic: bool = False, retries: int = 2, backoff: float = 3.0) -> str:
    """
    统一调用入口。
    use_critic=False -> 使用 DeepSeek 生成
    use_critic=True  -> 使用 GLM-4.7-Flash 审查
    """
    target_llm = llm_critic if use_critic else llm_generator
    last_err = None
    
    for attempt in range(retries + 1):
        try:
            response = target_llm.invoke(prompt)
            return response.content
        except Exception as e:
            msg = str(e)
            is_infra = any(k in msg for k in [
                "503", "429", "Service is too busy", "service_unavailable",
                "timeout", "Timeout", "Connection", "rate limit", "RateLimit"
            ])
            if is_infra:
                last_err = e
                if attempt < retries:
                    wait = backoff * (2 ** attempt)
                    print(f"  -> [LLM基础设施故障] {msg[:80]}... {wait:.1f}s 后重试")
                    time.sleep(wait)
                    continue
                raise LLMInfraError(f"LLM 服务持续不可用: {msg[:120]}")
            else:
                raise
    raise LLMInfraError(f"LLM 调用失败: {last_err}")

# ================= 4. 检索层 =================
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
    prompt = f"""判断以下检索结果与主题 '{topic}' 的相关性。
只返回一个 JSON 数组，包含每条的序号和判定（"相关"或"不相关"），不要其他文字。
格式：[{{"idx":1,"judge":"相关"}},{{"idx":2,"judge":"不相关"}}]

检索结果：
{items}
"""
    try:
        # 审查过滤使用 Critic 模型（GLM）
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

_SEARCH_CACHE: dict = {}

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

# ================= 5. 节点函数 =================

def planner_node(state: State):
    print("\n【规划节点】拆解任务...")
    topic = state["topic"]

    if state.get("replan_scope") == "全局":
        print(f"  -> [规划] 检测到全局重规划，尝试拆解主题：{topic}")
        clean_topic = re.sub(r"^(比较|对比|分析|简述)\s*", "", topic).strip()
        parts = re.split(r"\s*(?:和|与|对比|vs|VS)\s*", clean_topic, maxsplit=1)
        if len(parts) == 2 and all(p.strip() for p in parts):
            subtasks = [p.strip() for p in parts]
            print(f"  -> [规划] 拆解为 {len(subtasks)} 个子任务：")
            for i, st in enumerate(subtasks, 1):
                print(f"       {i}. {st}")
            return {
                "result": "",
                "subtasks": subtasks,
                "replan_count": state.get("replan_count", 0),
                "sources": state.get("sources", []),
                "used_fallback": False,
                "llm_fail_count": state.get("llm_fail_count", 0),
            }
        else:
            print("  -> [规划] 未识别出可拆解的复合结构，回退为单任务处理")

    return {
        "result": "",
        "subtasks": state.get("subtasks", []),
        "replan_count": state.get("replan_count", 0),
        "sources": state.get("sources", []),
        "used_fallback": False,
        "llm_fail_count": state.get("llm_fail_count", 0),
    }

def executor_node(state: State):
    print("【执行节点】开始执行任务...")
    topic = state["topic"]
    subtasks = state.get("subtasks", [])
    prev_sources = state.get("sources", [])

    if subtasks:
        search_topics = subtasks
        print(f"  -> [执行] 按 {len(subtasks)} 个子任务分别检索")
    else:
        query_topic = _clean_composite_topic(topic) if _is_composite_topic(topic) else topic
        search_topics = [query_topic]
        if _is_composite_topic(topic):
            print(f"  -> [执行] 检测到复合主题，第一轮整体检索（清洗后：'{query_topic}'），不自动拆解")

    all_contexts = []
    all_sources = []

    for st in search_topics:
        print(f"  -> [执行] 检索：{st}")
        context, sources = multi_source_search(st)
        if context:
            all_contexts.append(f"=== 检索：{st} ===\n{context}")
            all_sources.extend(sources)

    context = "\n\n".join(all_contexts)
    sources = list(dict.fromkeys(all_sources))

    if not context and prev_sources:
        print("  -> [检索] 本轮失败，复用历史来源")
        context = "\n".join(prev_sources)
        sources = prev_sources

    previous_errors = "\n".join(state.get("reflection_history", []))

    if context:
        if subtasks:
            prompt = f"""
你是一个严谨的学术助手。请根据以下针对多个子任务的参考资料，围绕原主题 '{topic}' 写一段200字左右的科普摘要。

【参考资料】：
{context}

要求：
1. 内容必须基于参考资料，不要编造事实。
2. 需要综合覆盖所有子任务的核心内容。
3. 末尾用一行标注"【来源】"，列出实际使用的资料标题。
"""
        else:
            prompt = f"""
你是一个严谨的学术助手。请根据以下提供的参考资料，针对主题 '{topic}' 写一段150字左右的科普摘要。

【参考资料】：
{context}

要求：
1. 内容必须基于参考资料，不要编造事实。
2. 必须紧扣主题 '{topic}'，不要偏离到上位概念。
3. 末尾用一行标注"【来源】"，列出实际使用的资料标题。
"""
    else:
        print("  -> [生成] 无外部检索证据，走模型知识兜底并声明")
        prompt = f"""
你是一个严谨的学术助手。请针对主题 '{topic}' 写一段150字左右的科普摘要。

注意：本次未能获取外部参考资料，请基于你自身的知识作答，并在末尾注明：
"【声明】本文未获取到外部检索证据，内容基于模型内部知识，仅供参考。"
"""

    if previous_errors:
        prompt += f"\n注意：上一次的尝试因为以下原因被指出了问题：\n{previous_errors}\n请务必避免重蹈覆辙。"

    try:
        # 摘要生成使用 Generator 模型（DeepSeek）
        content = safe_llm_invoke(prompt, use_critic=False)
        return {
            "result": content,
            "sources": sources,
            "used_fallback": (context == ""),
        }
    except LLMInfraError as e:
        print(f"  -> [执行] LLM 基础设施故障: {e}")
        return {
            "result": f"执行暂停：LLM 服务不可用 ({e})",
            "error_type": "系统繁忙",
            "llm_fail_count": state.get("llm_fail_count", 0) + 1,
        }
    except Exception as e:
        return {"result": f"执行崩溃：{e}"}

def reflection_node(state: State):
    print("【反思节点】正在真实检查结果质量...")

    if state.get("error_type") == "系统繁忙":
        print("  -> 执行阶段 LLM 故障，跳过内容质量判定，直接触发重试")
        return {"error_type": "系统繁忙"}

    topic = state["topic"]
    is_composite = _is_composite_topic(topic)
    has_subtasks = bool(state.get("subtasks"))

    composite_check_hint = ""
    if is_composite and not has_subtasks:
        composite_check_hint = f"""
【复合主题证据完整性检查（重要）】
当前主题 '{topic}' 是复合比较结构（含和/与/对比/比较）。
本轮是第一轮整体检索，尚未拆解为子任务。
请检查参考资料/来源列表：是否同时覆盖了比较的两个对象？
- 如果参考资料只来自其中一个对象，或资料为空，
  即使回答写得完整，也应判 "分解不合理"，理由是"证据只覆盖单一对象或为空，需拆解后分别检索"。
- 如果参考资料同时覆盖了两个对象，才允许判"通过"。
"""
    else:
        composite_check_hint = """
【单主题或已拆解后的检查】
正常检查主题贴合度、内容完整性。如果回答主要讲上位概念，判"偏题"。
"""

    prompt = f"""
你是严格的系统检查员。请检查以下关于 '{topic}' 的回答是否合格：
回答内容：{state['result']}
实际检索来源：{state.get('sources', [])}

请以严格的 JSON 格式输出（不要包含 Markdown 标记），包含两个字段：
1. "error_type": 只能是以下四种之一：
   - "通过"：回答合格
   - "内容不合格"：偏题、字数太少、未提供细节等
   - "检索失败"：检索无结果
   - "分解不合理"：主题本身包含多个需要独立回答的子问题，
     单段摘要无法完整覆盖，或证据只覆盖部分对象，需要重新拆解任务
2. "reason": 简短的理由
{composite_check_hint}
额外检查：
- 回答是否真正围绕主题 '{topic}' 展开？
- 如果回答主要讲的是主题的上位概念，应判"偏题"。
"""

    try:
        # 反思质检使用 Critic 模型（GLM）
        raw_content = safe_llm_invoke(prompt, use_critic=True).strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:-3].strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:-3].strip()
        res_dict = json.loads(raw_content)
        error_type = res_dict.get("error_type", "通过")
        reason = res_dict.get("reason", "")
    except LLMInfraError as e:
        print(f"  -> [反思] LLM 基础设施故障: {e}，走规则兜底")
        return rule_based_reflection(state, reason=f"反思节点LLM不可用: {e}")
    except (json.JSONDecodeError, LLMParseError) as e:
        print(f"  -> [反思] LLM 返回解析失败: {e}，走规则兜底")
        return rule_based_reflection(state, reason="格式解析错误")
    except Exception as e:
        print(f"  -> [反思] 未知异常: {e}，走规则兜底")
        return rule_based_reflection(state, reason=f"未知异常: {e}")

    print(f"  -> LLM 判定结果: {error_type}, 理由: {reason}")
    return _decide_next(state, error_type, reason)

def rule_based_reflection(state: State, reason: str):
    result = state.get("result", "")
    length = len(result.strip())

    if length < 20:
        error_type = "内容不合格"
        reason = f"字数过少({length}字)"
        print(f"  -> [规则兜底] 判定: {error_type}, 理由: {reason}")
        return _decide_next(state, error_type, reason)

    if state.get("used_fallback"):
        print("  -> [规则兜底] 无外部证据兜底生成，放行但标注")
    return {"error_type": "通过"}

def _decide_next(state: State, error_type: str, reason: str):
    if error_type == "通过":
        return {"error_type": "通过"}

    current_count = state.get("replan_count", 0)
    history = state.get("reflection_history", [])

    keywords = ["偏题", "字数太少", "不完整", "缺少", "无实质", "未提供", "超时", "无结果",
                "子问题", "未覆盖", "拆解", "单一对象", "证据只覆盖"]
    keyword_hit = any(kw in reason for kw in keywords)
    repeat_hit = history.count(error_type) >= 1

    if error_type == "分解不合理":
        print("  -> 判定为分解不合理，触发【全局】重规划")
        return {
            "error_type": error_type,
            "replan_scope": "全局",
            "replan_count": current_count + 1,
            "reflection_history": history + [f"[{error_type}] {reason}"],
        }

    evidence_status = "充分" if (keyword_hit or repeat_hit) else "不足"
    print(f"  -> 规则信号交叉验证：关键词命中={keyword_hit}, 历史重复={repeat_hit} => 证据{evidence_status}")

    if evidence_status == "充分":
        print("  -> 证据充分，触发【局部】重规划")
        return {
            "error_type": error_type,
            "replan_scope": "局部",
            "replan_count": current_count + 1,
            "reflection_history": history + [f"[{error_type}] {reason}"],
        }
    else:
        print("  -> 证据不足，回退到【自由反思】")
        return {
            "error_type": "证据不足",
            "replan_count": current_count + 1,
            "reflection_history": history + [f"[证据不足] {reason}"],
        }

def free_reflection_node(state: State):
    print("【自由反思节点】证据不足，先补检索再决定策略...")

    topic = state["topic"]
    context, sources = multi_source_search(topic)

    if context:
        print("  -> [补检索] 成功，基于新证据重写")
        prompt = f"""
你之前关于 '{topic}' 的回答被退回。现在补充了新的参考资料：

【参考资料】：
{context}

请基于以上资料重新写一段100字左右的科普介绍，必须紧扣主题 '{topic}'，
末尾标注"【来源】"列出使用的标题。
"""
        used_fallback = False
    else:
        print("  -> [补检索] 仍失败，走模型知识兜底并声明")
        prompt = f"""
你之前关于 '{topic}' 的回答被退回，且无法获取外部资料。
请基于自身知识写一段100字左右的科普介绍，并在末尾注明：
"【声明】未获取到外部检索证据，内容基于模型内部知识。"
"""
        used_fallback = True

    try:
        # 自由反思重写依然用 Generator（DeepSeek）
        content = safe_llm_invoke(prompt, use_critic=False)
        return {
            "result": content,
            "error_type": "通过",
            "sources": sources if context else state.get("sources", []),
            "used_fallback": used_fallback,
        }
    except LLMInfraError as e:
        print(f"  -> [自由反思] LLM 仍不可用: {e}")
        return {
            "result": state.get("result", "") + "\n\n【声明】反思阶段 LLM 不可用，保留上一轮结果。",
            "error_type": "通过",
        }
    except Exception:
        return {"result": "自由反思失败，系统终止。", "error_type": "通过"}

# ================= 6. 路由函数 =================

def route_after_reflection(state: State):
    if state["error_type"] == "通过":
        return "end"

    if state["error_type"] == "系统繁忙":
        if state.get("llm_fail_count", 0) >= 3:
            print("  -> LLM 连续故障达 3 次，强制结束")
            return "end"
        print("  -> LLM 系统繁忙，触发【局部】重试")
        return "replan_local"

    if state["replan_count"] >= 3:
        print("  -> 达到最大重试次数，强制结束。")
        return "end"

    if state["error_type"] == "证据不足":
        return "free_reflection"

    if state.get("replan_scope") == "全局":
        return "replan_global"
    else:
        return "replan_local"

# ================= 7. 组装状态图 =================

workflow = StateGraph(State)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("free_reflection", free_reflection_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "reflection")

workflow.add_conditional_edges(
    "reflection",
    route_after_reflection,
    {
        "end": END,
        "free_reflection": "free_reflection",
        "replan_global": "planner",
        "replan_local": "executor",
    },
)

workflow.add_edge("free_reflection", END)

app = workflow.compile()

if __name__ == "__main__":
    print("=== 欢迎使用多智能体工作流系统 ===")
    user_topic = input("请输入你想检索并生成摘要的主题（例如：量子计算、大语言模型）：\n> ")
    if not user_topic.strip():
        user_topic = "LangGraph多智能体"
        print(f"检测到空输入，默认使用主题：{user_topic}")

    print(f"\n=== 开始运行工作流，当前主题：{user_topic} ===")

    initial_state = {
        "topic": user_topic.strip(),
        "subtasks": [],
        "replan_count": 0,
        "reflection_history": [],
        "sources": [],
        "used_fallback": False,
        "llm_fail_count": 0,
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 40)
    print("=== 最终执行结果 ===")
    print(result["result"])

    if result.get("sources"):
        print("\n=== 引用来源 ===")
        for s in result["sources"]:
            print(f"- {s}")

    if result.get("subtasks"):
        print(f"\n=== 子任务拆解 ===\n{result['subtasks']}")

    if result.get("used_fallback"):
        print("\n[注意] 本次结果未获取到外部检索证据，基于模型内部知识生成。")

    print(f"\n总共进行了 {result['replan_count']} 次重规划")
#核心的 4 个节点
import re
import json

from config import State
from utils import safe_llm_invoke, LLMInfraError, LLMParseError
from tools import multi_source_search, _is_composite_topic, _clean_composite_topic
import prompts

def planner_node(state: State):
    print("\n【规划节点】拆解任务...")
    topic = state["topic"]

    if state.get("replan_scope") == "全局":
        print(f"  -> [规划] 检测到全局重规划，尝试拆解主题：{topic}")
        try:
            # 用大模型进行语义级拆解
            prompt = prompts.TASK_DECOMPOSE_PROMPT.format(topic=topic)
            raw = safe_llm_invoke(prompt, use_critic=False).strip()
            
            # 清理可能的 Markdown
            if raw.startswith("```json"):
                raw = raw[7:-3].strip()
            elif raw.startswith("```"):
                raw = raw[3:-3].strip()
                
            subtasks = json.loads(raw)
            
            if isinstance(subtasks, list) and len(subtasks) >= 2:
                print(f"  -> [规划] 大模型智能拆解为 {len(subtasks)} 个子任务：")
                for i, st in enumerate(subtasks, 1):
                    print(f"       {i}. {st}")
                return {
                    "result": "", "subtasks": subtasks,
                    "replan_count": state.get("replan_count", 0),
                    "sources": state.get("sources", []),
                    "used_fallback": False, "llm_fail_count": state.get("llm_fail_count", 0),
                }
        except Exception as e:
            print(f"  -> [规划] LLM 拆解失败 ({str(e)[:40]})，降级使用正则兜底")

        # 如果大模型拆解失败，兜底使用正则（以防万一）
        clean_topic = re.sub(r"^(比较|对比|分析|简述)\s*", "", topic).strip()
        parts = re.split(r"\s*(?:和|与|对比|vs|VS)\s*", clean_topic, maxsplit=1)
        if len(parts) == 2 and all(p.strip() for p in parts):
            subtasks = [p.strip() for p in parts]
            print(f"  -> [规划] 正则兜底拆解为 {len(subtasks)} 个子任务：")
            for i, st in enumerate(subtasks, 1):
                print(f"       {i}. {st}")
            return {
                "result": "", "subtasks": subtasks,
                "replan_count": state.get("replan_count", 0),
                "sources": state.get("sources", []),
                "used_fallback": False, "llm_fail_count": state.get("llm_fail_count", 0),
            }
        else:
            print("  -> [规划] 未识别出可拆解的复合结构，回退为单任务处理")

    return {
        "result": "", "subtasks": state.get("subtasks", []),
        "replan_count": state.get("replan_count", 0),
        "sources": state.get("sources", []),
        "used_fallback": False, "llm_fail_count": state.get("llm_fail_count", 0),
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

    all_contexts, all_sources = [], []
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
            prompt = prompts.GENERATE_COMPOSITE_PROMPT.format(topic=topic, context=context)
        else:
            prompt = prompts.GENERATE_SINGLE_PROMPT.format(topic=topic, context=context)
    else:
        print("  -> [生成] 无外部检索证据，走模型知识兜底并声明")
        prompt = prompts.GENERATE_FALLBACK_PROMPT.format(topic=topic)

    if previous_errors:
        prompt += f"\n注意：上一次的尝试因为以下原因被指出了问题：\n{previous_errors}\n请务必避免重蹈覆辙。"

    try:
        content = safe_llm_invoke(prompt, use_critic=False)
        return {"result": content, "sources": sources, "used_fallback": (context == "")}
    except LLMInfraError as e:
        print(f"  -> [执行] LLM 基础设施故障: {e}")
        return {"result": f"执行暂停：LLM 服务不可用 ({e})", "error_type": "系统繁忙", "llm_fail_count": state.get("llm_fail_count", 0) + 1}
    except Exception as e:
        return {"result": f"执行崩溃：{e}"}

def reflection_node(state: State):
    print("【反思节点】正在真实检查结果质量...")

    if state.get("error_type") == "系统繁忙":
        print("  -> 执行阶段 LLM 故障，跳过内容质量判定，直接触发重试")
        return {"error_type": "系统繁忙"}
    
    # 只有当字数极度异常（比如少于 20 字）才强制拦截，否则交给大模型去判
    if state.get("used_fallback") and len(state.get("result", "")) < 20:
        print("  -> [反思] 无资料兜底，且字数极少，判定不合格")
        return _decide_next(state, "内容不合格", "字数太少")

    topic = state["topic"]
    is_composite = _is_composite_topic(topic)
    has_subtasks = bool(state.get("subtasks"))

    if is_composite and not has_subtasks:
        composite_check_hint = prompts.COMPOSITE_CHECK_HINT_TRUE.format(topic=topic)
    else:
        composite_check_hint = prompts.COMPOSITE_CHECK_HINT_FALSE

    prompt = prompts.REFLECTION_PROMPT.format(
        topic=topic, result=state['result'], sources=state.get('sources', []), composite_check_hint=composite_check_hint
    )

    try:
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

    keywords = ["偏题", "字数太少", "不完整", "缺少", "无实质", "未提供", "超时", "无结果", "子问题", "未覆盖", "拆解", "单一对象", "证据只覆盖"]
    keyword_hit = any(kw in reason for kw in keywords)
    repeat_hit = history.count(error_type) >= 1

    if error_type == "分解不合理":
        print("  -> 判定为分解不合理，触发【全局】重规划")
        return {"error_type": error_type, "replan_scope": "全局", "replan_count": current_count + 1, "reflection_history": history + [f"[{error_type}] {reason}"]}

    evidence_status = "充分" if (keyword_hit or repeat_hit) else "不足"
    print(f"  -> 规则信号交叉验证：关键词命中={keyword_hit}, 历史重复={repeat_hit} => 证据{evidence_status}")

    if evidence_status == "充分":
        print("  -> 证据充分，触发【局部】重规划")
        return {"error_type": error_type, "replan_scope": "局部", "replan_count": current_count + 1, "reflection_history": history + [f"[{error_type}] {reason}"]}
    else:
        print("  -> 证据不足，回退到【自由反思】")
        return {"error_type": "证据不足", "replan_count": current_count + 1, "reflection_history": history + [f"[证据不足] {reason}"]}

def free_reflection_node(state: State):
    print("【自由反思节点】证据不足，先补检索再决定策略...")

    topic = state["topic"]
    context, sources = multi_source_search(topic)

    if context:
        print("  -> [补检索] 成功，基于新证据重写")
        prompt = prompts.FREE_REFLECT_SUCCESS_PROMPT.format(topic=topic, context=context)
        used_fallback = False
    else:
        print("  -> [补检索] 仍失败，走模型知识兜底并声明")
        prompt = prompts.FREE_REFLECT_FAIL_PROMPT.format(topic=topic)
        used_fallback = True

    try:
        content = safe_llm_invoke(prompt, use_critic=False)
        return {"result": content, "error_type": "通过", "sources": sources if context else state.get("sources", []), "used_fallback": used_fallback}
    except LLMInfraError as e:
        print(f"  -> [自由反思] LLM 仍不可用: {e}")
        return {"result": state.get("result", "") + "\n\n【声明】反思阶段 LLM 不可用，保留上一轮结果。", "error_type": "通过"}
    except Exception:
        return {"result": "自由反思失败，系统终止。", "error_type": "通过"}
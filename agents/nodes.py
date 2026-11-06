#核心的 4 个节点
import re
import json
from typing import List

from config import (State, PASS_MIN_CITED, PASS_MIN_CHARS, RETRY_QUERY_MODE,
                    DISABLE_CONTENT_GATE, ENABLE_COVERAGE_RULE,
                    ENABLE_DEGRADE_RESCUE, ENABLE_REFLECTION, ENABLE_REPEAT_RULE,
                    ENABLE_TRUNCATION_RULE)
from utils import (safe_llm_invoke, LLMInfraError, LLMParseError, content_gate,
                   _body_chars, _body_text, llm_call_count)
from tools import (search_with_evidence, evidence_coverage,
                   _is_composite_topic, _clean_composite_topic, build_query)
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
                    "replan_scope": "",   # P2-3：显式复位，避免全局标记粘到下一轮
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
                "replan_scope": "",
                "sources": state.get("sources", []),
                "used_fallback": False, "llm_fail_count": state.get("llm_fail_count", 0),
            }
        else:
            print("  -> [规划] 未识别出可拆解的复合结构，回退为单任务处理")

    return {
        "result": "", "subtasks": state.get("subtasks", []),
        "replan_count": state.get("replan_count", 0),
        "replan_scope": "",
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

    all_contexts, all_sources, all_results, per_query = [], [], [], []
    for st in search_topics:
        print(f"  -> [执行] 检索：{st}")
        context, sources, results, stats = search_with_evidence(st)
        per_query.append(stats)
        if context:
            all_contexts.append(f"=== 检索：{st} ===\n{context}")
            all_sources.extend(sources)
        all_results.extend(results)

    context = "\n\n".join(all_contexts)
    sources = list(dict.fromkeys(all_sources))

    # 量化统计：落盘到 data/runs/*.json，README 与论文表格的数字统一取自这里
    search_stats = {
        "per_query": per_query,
        "raw": sum(s.get("raw", 0) for s in per_query),
        "gated": sum(s.get("gated", 0) for s in per_query),
        "filtered": sum(s.get("filtered", 0) for s in per_query),
        "evidence_chars": len(context),
        "sources": len(sources),
        "providers": sorted({p for s in per_query for p in s.get("providers", [])}),
        "cache": [s.get("cache") for s in per_query],
        "cache_sources": [s.get("cache_source", s.get("cache")) for s in per_query],
        "degraded": any(s.get("degraded") for s in per_query),
        # v2.2：过滤是否真的跑了（applied=False 表示 GLM 故障或开关关闭）
        "llm_filter": [s.get("llm_filter", {}) for s in per_query],
        "elapsed_s": round(sum(s.get("elapsed_s", 0) for s in per_query), 2),
        "evidence_path": "; ".join(
            s.get("evidence_path", "") for s in per_query if s.get("evidence_path")),
    }
    run_stats = {"search": search_stats}

    if all_results:
        avg_summary = sum(len(r.get("summary") or "") for r in all_results) / len(all_results)
        avg_content = sum(len(r.get("content") or "") for r in all_results) / len(all_results)
        providers = sorted({r.get("provider", "?") for r in all_results})
        print(f"  -> [执行] 证据统计：检索 {len(all_results)} 条 / {len(sources)} 个链接 / "
              f"装配 {len(context)} 字（平均摘要 {avg_summary:.0f} 字，平均正文 {avg_content:.0f} 字，"
              f"provider={','.join(providers)}）")
    else:
        print("  -> [执行] 证据统计：0 条来源")

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
        # error_type 置空：清掉上一轮遗留的"系统繁忙"，否则反思节点会误判为执行侧仍故障
        return {"result": content, "sources": sources, "error_type": "",
                "replan_scope": "",   # P2-3：每轮执行后复位，路由不再读到上一轮的"全局"
                "evidence": all_results or state.get("evidence", []),
                "run_stats": {**run_stats, "generation": {
                    "prompt_chars": len(prompt), "result_chars": len(content)}},
                "used_fallback": (context == "")}
    except LLMInfraError as e:
        print(f"  -> [执行] LLM 基础设施故障: {e}")
        return {"result": f"执行暂停：LLM 服务不可用 ({e})", "error_type": "系统繁忙"}
    except Exception as e:
        print(f"  -> [执行] 生成失败: {str(e)[:120]}")
        # 检索证据本身有效，保留下来供反思节点做覆盖率判断
        return {"result": f"执行崩溃：{e}",
                "evidence": all_results or state.get("evidence", []),
                "run_stats": {**run_stats, "generation": {"error": str(e)[:200]}}}

def _pass_kind(verdict: str) -> str:
    """把终态压成论文可统计的三态：严格通过 / 降级通过 / 被否决。"""
    if verdict == "通过":
        return "strict"
    if verdict == "降级通过":
        return "degraded"
    return "rejected"


def _reflection_stats(state: State, coverage, verdict: str, reason: str,
                      gated_by: str, gate: dict = None, repeat_hit: bool = False,
                      decision_source: str = None, rescue: tuple = None) -> dict:
    """把反思节点的量化信号并入 run_stats（state 是覆盖语义，必须带上已有字段）。"""
    run_stats = dict(state.get("run_stats") or {})
    stats = {
        "coverage": coverage,
        "verdict": verdict,
        "reason": reason,
        "gated_by": gated_by,
        # P2-1：判定来源（llm / coverage_rule / gate / rule_fallback / free_reflection / baseline）；
        # gated_by 保留旧语义，两者默认一致，便于 v1 脚本继续读
        "decision_source": decision_source or gated_by,
        "repeat_hit": bool(repeat_hit),
        "llm_calls": llm_call_count(),
        "evidence_count": len(state.get("evidence", [])),
        "result_chars": len((state.get("result") or "").strip()),
        "pass_kind": _pass_kind(verdict),
    }
    if gate:
        # 门控事实一并落盘：事后可复查「当时到底缺哪一项」，不必重跑
        stats.update({
            "has_source_line": gate["has_source_line"],
            "cited_count": gate["cited_count"],
            "citation_ratio": gate["citation_ratio"],
            "citation_ratio_ok": gate.get("citation_ratio_ok", True),
            "len_ok": gate["len_ok"],
            "body_chars": gate.get("body_chars", 0),
            "len_full": gate.get("len_full", 0),
            "cited_loose": gate.get("cited_loose", []),
            # v2.2 完整性：截断命中项原样落盘，判「内容不合格」时能指出是哪一种
            "truncated": gate.get("truncated", False),
            "truncation_hits": gate.get("truncation_hits", []),
        })
    if rescue:
        # v2.2 降级救援：记录「从哪个否决类型救回来、凭什么救」，
        # 三态分布变化必须能逐条追溯，否则等于放宽了判定
        stats["rescued_from"], stats["rescue_rule"] = rescue
    run_stats["reflection"] = stats
    return {"run_stats": run_stats}


def _apply_pass_gate(state: State, error_type: str, reason: str, gate: dict):
    """对「通过」施加确定性否决（顺序即优先级），三条判定路径共用同一套规则。

    三条路径分别是：裁判 LLM 正常返回、裁判 LLM 故障走规则兜底、自由反思补检索成功。
    以前只有第一条过门控，另两条各自直接判「通过」，导致"无来源"也能进合格样本。

    返回 (error_type, reason, gate_hit)：gate_hit 为空表示未拦下，维持原判定。
    """
    if error_type != "通过":
        return error_type, reason, ""

    # E2 消融：关掉确定性否决链，完全以语义裁判为准
    if DISABLE_CONTENT_GATE:
        print("  -> [门控] DISABLE_CONTENT_GATE=True（E2 纯 LLM 反思），跳过确定性否决")
        return error_type, reason, ""

    result_len = len((state.get("result") or "").strip())
    if not gate["evidence_nonempty"]:
        return ("降级通过", "回答可读但无外部证据支撑，降级记录（不计入合格样本）",
                "证据为空")
    if state.get("used_fallback"):
        return ("降级通过", "无外部检索证据（模型知识兜底），降级记录（不计入合格样本）",
                "无资料兜底")
    if not gate["no_fabricated_url"]:
        return ("内容不合格", f"来源含不可核验 URL：{gate['fabricated_urls']}",
                "疑似编造 URL")
    if not gate["len_ok"]:
        return ("内容不合格",
                (f"回答正文过短（正文 {gate.get('body_chars', result_len)} 字 / "
                 f"全文 {result_len} 字），未达 {PASS_MIN_CHARS} 字下限"),
                "回答过短")
    if ENABLE_TRUNCATION_RULE and gate.get("truncated"):
        # v2.2：完整性是「通过」的必要条件——半截话即使字数够、引用可核验也不算合格
        return ("内容不合格",
                (f"回答疑似截断（{'、'.join(gate.get('truncation_hits') or ['未完成'])}），"
                 f"完整性不达标"),
                "回答截断")
    if not gate["has_source_line"] or gate["cited_count"] < PASS_MIN_CITED:
        return ("降级通过",
                (f"回答缺少可核验来源（【来源】={'有' if gate['has_source_line'] else '无'}，"
                 f"可核验引用 {gate['cited_count']} 条 < {PASS_MIN_CITED}），"
                 "降级记录（不计入合格样本）"),
                "无引用支撑")
    if not gate.get("citation_ratio_ok", True):
        # v2.2：引用率门槛默认 0（不参与判定），实验里设 PASS_MIN_CITATION_RATIO 才生效
        return ("降级通过",
                (f"引用率 {gate.get('citation_ratio')} 未达门槛 "
                 f"{gate.get('citation_ratio_limit')}，降级记录（不计入合格样本）"),
                "引用率不足")
    return error_type, reason, ""


# ================= 降级救援（v2.2） =================
# 为什么需要：门控原先只能否决（error_type != 通过 时直接放行），于是
# 「回答可读、只是没拿外部资料 / 已如实声明兜底」的样本被判成「否决」，
# 与论文口径（合格 / 降级 / 否决 三态）对不上。救援只做一件事：
# 把这类否决改判为「降级通过」，**永不**改判为「通过」，二分类指标不受影响。
_DECLARE_RE = re.compile(
    r"【声明】[^\n]{0,60}(未获取到|无法获取|未检索到|基于模型内部知识|模型知识兜底)")
# 主题词抽取：仅用于「这条回答是不是在讲这个题」的粗判，不参与覆盖率统计。
_TOPIC_SPLIT_RE = re.compile(r"[和与及、/]|比较|对比|分析")
_TOPIC_STOP_SUFFIX = ("的结构", "的特点", "的原理", "的方法", "的作用", "的定义",
                      "的区别", "是什么", "有哪些", "怎么样")


def _topic_terms(topic: str) -> List[str]:
    terms = []
    for part in _TOPIC_SPLIT_RE.split((topic or "").strip()):
        p = part.strip().strip("的 ")
        for suf in _TOPIC_STOP_SUFFIX:
            if p.endswith(suf):
                p = p[: -len(suf)].strip()
                break
        if len(p) >= 2:
            terms.append(p)
    return terms


def _answer_on_topic(state: State) -> bool:
    """粗判「回答是否在讲主题」：主题词至少命中一个。偏题回答靠这条挡住误救援。"""
    body = _body_text(state.get("result") or "")
    terms = _topic_terms(state.get("topic") or "")
    return bool(terms) and any(t in body for t in terms)


def _apply_degrade_rescue(state: State, error_type: str, reason: str, gate: dict = None):
    """把「可读但缺资料」的否决改判「降级通过」（顺序即优先级）。

    三条规则都只依赖可复现的事实，且都要求回答本身可读（len_ok、无编造 URL、未截断）：
    R1 声明式兜底：回答里【声明】了没有外部证据 —— 如实交代不该按「胡说」记否决；
    R2 无资料兜底：检索为空（证据 0 条）或执行侧已标记 used_fallback；
    R3 未引用来源：有证据、可读、但没写可核验引用，且确实在讲主题。

    返回 (error_type, reason, rescue_rule)；rescue_rule 为空表示未救援。
    """
    if error_type in ("通过", "降级通过"):
        return error_type, reason, ""
    if DISABLE_CONTENT_GATE or not ENABLE_DEGRADE_RESCUE:
        return error_type, reason, ""
    if gate:
        readable = (gate.get("len_ok") and gate.get("no_fabricated_url")
                    and not (ENABLE_TRUNCATION_RULE and gate.get("truncated")))
        if not readable:
            return error_type, reason, ""
    else:
        readable = _body_chars(state.get("result") or "") >= PASS_MIN_CHARS
        if not readable:
            return error_type, reason, ""

    def _degrade(rule: str, why: str):
        return ("降级通过", f"{reason}｜{why}，降级记录（不计入合格样本）", rule)

    if _DECLARE_RE.search(state.get("result") or ""):
        return _degrade("声明式兜底", "回答已声明无外部检索证据")
    if state.get("used_fallback"):
        return _degrade("无资料兜底", "检索为空，回答基于模型内部知识")
    if gate and not gate.get("evidence_nonempty"):
        # 证据为空但答案没声明：与门控对「通过」的处理保持同一口径（一律记降级）
        return _degrade("无资料兜底", "无外部检索证据，回答基于模型内部知识")
    if (gate and gate.get("evidence_nonempty") and gate.get("cited_count") == 0
            and _answer_on_topic(state)):
        return _degrade("未引用来源", "回答可读但未写可核验引用")
    return error_type, reason, ""


def reflection_node(state: State):
    print("【反思节点】正在真实检查结果质量...")

    # E1 基线：关闭反思 → 直接判「通过」并结束（不看内容、不重规划）
    if not ENABLE_REFLECTION:
        print("  -> [反思] ENABLE_REFLECTION=False（E1 基线），跳过质量判定，直接判「通过」")
        return {"error_type": "通过",
                **_reflection_stats(state, None, "通过", "基线：关闭反思（E1）",
                                    "baseline", decision_source="baseline")}

    if state.get("error_type") == "系统繁忙":
        print("  -> 执行阶段 LLM 故障，跳过内容质量判定，直接触发重试")
        # 计数只在反思侧累加，避免与执行侧重复计数导致提前熔断或计数不增而死循环
        return {"error_type": "系统繁忙",
                "llm_fail_count": state.get("llm_fail_count", 0) + 1}

    if (state.get("result") or "").startswith(("执行崩溃", "执行暂停")):
        print("  -> 执行阶段异常，跳过内容质量判定，直接触发重试")
        return {"error_type": "系统繁忙",
                "llm_fail_count": state.get("llm_fail_count", 0) + 1}

    # 只有当字数极度异常（比如少于 20 字）才强制拦截，否则交给大模型去判
    if state.get("used_fallback") and _body_chars(state.get("result", "")) < 20:
        print("  -> [反思] 无资料兜底，且字数极少，判定不合格")
        return _decide_next(state, "内容不合格", "字数太少")

    topic = state["topic"]
    is_composite = _is_composite_topic(topic)
    has_subtasks = bool(state.get("subtasks"))
    coverage_info = None

    if is_composite and not has_subtasks:
        composite_check_hint = prompts.COMPOSITE_CHECK_HINT_TRUE.format(topic=topic)
    else:
        composite_check_hint = prompts.COMPOSITE_CHECK_HINT_FALSE

    # 确定性证据覆盖率（不依赖 LLM，可复现）
    evidence_items = state.get("evidence", [])
    subtasks = state.get("subtasks", [])

    if ENABLE_COVERAGE_RULE and has_subtasks and evidence_items:
        # 已拆解：每个子任务都是"需覆盖对象"，用子任务字面量匹配证据 blob
        blob = " ".join(
            " ".join([r.get("title") or "", r.get("summary") or "",
                      r.get("snippet") or "", (r.get("content") or "")[:200]])
            for r in evidence_items
        )
        hit = [st for st in subtasks if st in blob]
        missing = [st for st in subtasks if st not in blob]
        ratio = len(hit) / len(subtasks) if subtasks else 1.0
        coverage = {"terms": subtasks, "hit": hit, "missing": missing,
                    "ratio": ratio, "count": len(evidence_items)}
        coverage_info = coverage
        evidence_stats = (
            f"证据 {coverage['count']} 条；子任务 {subtasks}；"
            f"已覆盖 {hit or ['无']}；未覆盖 {missing or ['无']}；"
            f"覆盖率 {ratio:.0%}"
        )
        print(f"  -> [反思] {evidence_stats}")

        if ratio < 1.0:
            reason = f"子任务覆盖率仅 {ratio:.0%}，未覆盖：{'、'.join(missing)}"
            print("  -> [反思] 已拆解但子任务覆盖率 < 100%，判分解不合理")
            decision = _decide_next(state, "分解不合理", reason)
            repeat_hit = decision.pop("_repeat_hit", False)
            return {**decision,
                    **_reflection_stats(state, coverage, "分解不合理", reason,
                                        "coverage_rule", repeat_hit=repeat_hit)}

    elif ENABLE_COVERAGE_RULE and is_composite and evidence_items:
        # 首轮整体检索：走原来的复合主题切分逻辑
        coverage = evidence_coverage(topic, evidence_items)
        coverage_info = coverage
        # 偏侧检测：把 per_term_counts 和 balance 一并喂给 GLM 裁判
        balance_str = (
            f"；偏侧检测：{coverage.get('per_term_counts', {})}，"
            f"balance={coverage.get('balance', 1.0):.2f}"
        )
        evidence_stats = (
            f"证据 {coverage['count']} 条；需覆盖对象 {coverage['terms'] or ['无']}；"
            f"已覆盖 {coverage['hit'] or ['无']}；未覆盖 {coverage['missing'] or ['无']}；"
            f"覆盖率 {coverage['ratio']:.0%}{balance_str}"
        )
        print(f"  -> [反思] {evidence_stats}")

        if coverage["terms"] and coverage["ratio"] < 1.0:
            print("  -> [反思] 复合主题证据覆盖率 < 100%，跳过 LLM 直接判分解不合理")
            reason = f"证据覆盖率仅 {coverage['ratio']:.0%}，未覆盖：{'、'.join(coverage['missing'])}"
            decision = _decide_next(state, "分解不合理", reason)
            repeat_hit = decision.pop("_repeat_hit", False)
            return {**decision,
                    **_reflection_stats(state, coverage, "分解不合理", reason,
                                        "coverage_rule", repeat_hit=repeat_hit)}
    else:
        if not ENABLE_COVERAGE_RULE:
            evidence_stats = (f"证据 {len(evidence_items)} 条"
                              f"（覆盖率规则已关闭：ENABLE_COVERAGE_RULE=False）")
        elif is_composite:
            evidence_stats = "证据 0 条（本轮未获得证据，覆盖率检查跳过）"
        else:
            evidence_stats = f"证据 {len(evidence_items)} 条（单主题，无需比较对象覆盖率检查）"
        print(f"  -> [反思] {evidence_stats}")

    prompt = prompts.REFLECTION_PROMPT.format(
        topic=topic, result=state['result'],
        sources="\n".join(state.get('sources', [])) or "（无）",
        evidence_stats=evidence_stats, composite_check_hint=composite_check_hint,
        min_chars=PASS_MIN_CHARS,     # 阈值单一来源：config.PASS_MIN_CHARS
    )

    try:
        raw_content = safe_llm_invoke(prompt, use_critic=True).strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:-3].strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:-3].strip()
        res_dict = json.loads(raw_content)
        llm_error_type = res_dict.get("error_type", "通过")
        reason = res_dict.get("reason", "")
    except LLMInfraError as e:
        print(f"  -> [反思] LLM 基础设施故障: {e}，走规则兜底")
        return rule_based_reflection(state, reason=f"反思节点LLM不可用: {e}", coverage=coverage_info)
    except (json.JSONDecodeError, LLMParseError) as e:
        print(f"  -> [反思] LLM 返回解析失败: {e}，走规则兜底")
        return rule_based_reflection(state, reason="格式解析错误", coverage=coverage_info)
    except Exception as e:
        print(f"  -> [反思] 未知异常: {e}，走规则兜底")
        return rule_based_reflection(state, reason=f"未知异常: {e}", coverage=coverage_info)

    print(f"  -> LLM 判定结果: {llm_error_type}, 理由: {reason}")

    # ===== 确定性门控（不依赖 LLM，对「通过」拥有最终否决权）=====
    gate = content_gate(state.get("result", ""), state.get("sources", []),
                        evidence_items)
    error_type, reason, gate_hit = _apply_pass_gate(state, llm_error_type, reason, gate)
    if gate_hit:
        print(f"  -> [门控] {gate_hit}，将「{llm_error_type}」改判为「{error_type}」")

    # v2.2 反向纠偏：可读但缺资料的否决 → 降级通过（不会变成通过）
    rescue = None
    rescued, reason, rescue_rule = _apply_degrade_rescue(state, error_type, reason, gate)
    if rescue_rule:
        print(f"  -> [降级救援] {rescue_rule}，将「{error_type}」改判为「{rescued}」")
        rescue = (error_type, rescue_rule)
        error_type = rescued

    gated_by = "llm" if error_type == llm_error_type else "gate"
    decision = _decide_next(state, error_type, reason)
    repeat_hit = decision.pop("_repeat_hit", False)
    return {**decision,
            **_reflection_stats(state, coverage_info, error_type, reason, gated_by,
                                gate, repeat_hit=repeat_hit, rescue=rescue)}

def rule_based_reflection(state: State, reason: str, coverage=None):
    result = state.get("result", "")
    # v2.2：与门控同口径，按正文计字数（旧写法用全文，【来源】行能凑数）
    length = _body_chars(result)
    reason_full = f"规则兜底：{reason}"

    if length < 20:
        error_type = "内容不合格"
        reason_full = f"字数过少({length}字)"
        print(f"  -> [规则兜底] 判定: {error_type}, 理由: {reason_full}")
        decision = _decide_next(state, error_type, reason_full)
        repeat_hit = decision.pop("_repeat_hit", False)
        return {**decision,
                **_reflection_stats(state, coverage, error_type, reason_full,
                                    "rule_fallback", repeat_hit=repeat_hit)}

    # 兜底路径复用同一套确定性门控：没有可核验引用，同样不能记「通过」
    gate = content_gate(result, state.get("sources", []), state.get("evidence", []))
    error_type, reason_full, gate_hit = _apply_pass_gate(state, "通过", reason_full, gate)
    if gate_hit:
        print(f"  -> [门控] 规则兜底后 {gate_hit}，判定「{error_type}」")

    gated_by = "gate" if gate_hit else "rule_fallback"
    decision = _decide_next(state, error_type, reason_full)
    repeat_hit = decision.pop("_repeat_hit", False)
    return {**decision,
            **_reflection_stats(state, coverage, error_type, reason_full, gated_by,
                                gate, repeat_hit=repeat_hit)}

# 错误类型 → 是否走定向（局部/全局）重规划。不在表里的（检索失败/系统繁忙/证据不足）一律走自由反思。
_REPLAN_TYPES = {
    "内容不合格": "局部",
    "分解不合理": "全局",
}
# 真正表示"检索层没拿到东西"的信号，用于把"检索失败"从定向重规划里摘出来
_NO_EVIDENCE_HINTS = ("证据 0 条", "检索失败", "未获取到", "无结果", "检索无结果")

# LLM 可自报、但落盘口径需要归一的标签。
# 背景：prompts.REFLECTION_PROMPT 允许裁判返回「检索失败」，而历史写入时用的是归一后的
# 「证据不足」；早先匹配却按 `[检索失败]` 找前缀，导致同类失败第 2 次出现时永远匹配不到
# （repeat_hit 恒为 False）。现在「存储」与「匹配」共用 _norm_type 的结果。
_TYPE_ALIASES = {"检索失败": "证据不足"}


def _norm_type(error_type: str) -> str:
    """把裁判自报的失败类型归一到落盘口径（只做标签映射，不改任何判定/路由）。"""
    et = (error_type or "").strip()
    return _TYPE_ALIASES.get(et, et)


def _decide_next(state: State, error_type: str, reason: str):
    """判定 → 重规划决策。返回里带一个临时键 `_repeat_hit`，由调用方 pop 后写进 run_stats。

    P1-1：重复检测从"只对分解不合理生效"扩到**任意类型**——同一类否决连发 2 次，
    说明重规划没带来新信息（同一条 query / 同一批证据），继续原地重试只会空转到上限，
    因此第 2 次起直接转「证据不足」走自由反思（换检索式）。
    """
    raw_type = (error_type or "").strip()
    error_type = _norm_type(raw_type)      # 存储 / 匹配同形，见 _TYPE_ALIASES
    if error_type in ("通过", "降级通过"):
        return {"error_type": error_type, "_repeat_hit": False}

    current_count = state.get("replan_count", 0)
    history = state.get("reflection_history", [])

    # 前缀匹配，与历史写入格式 f"[{error_type}] {reason}" 对齐
    repeat_times = sum(1 for h in history
                       if isinstance(h, str) and h.startswith(f"[{error_type}]"))
    repeat_hit = repeat_times > 0

    # 归一发生了（如 检索失败→证据不足）时保留原判，便于论文里追溯裁判措辞
    note = f"原判{raw_type}：" if raw_type != error_type else ""

    if repeat_hit and ENABLE_REPEAT_RULE:
        print(f"  -> [{error_type}] 已重复出现 {repeat_times} 次，停止同类重规划，"
              f"转【自由反思】换检索式")
        return {"error_type": "证据不足", "replan_count": current_count + 1,
                "reflection_history":
                    history + [f"[证据不足] {error_type}重复{repeat_times + 1}次"],
                "_repeat_hit": True}

    # 检索层无证据时硬否决定向重规划
    no_evidence = (len(state.get("evidence", [])) == 0) or any(
        hint in (reason or "") for hint in _NO_EVIDENCE_HINTS)

    if error_type == "分解不合理":
        print("  -> 判定为分解不合理，触发【全局】重规划")
        return {"error_type": error_type, "replan_scope": "全局",
                "replan_count": current_count + 1,
                "reflection_history": history + [f"[{error_type}] {note}{reason}"],
                "_repeat_hit": False}

    scope = _REPLAN_TYPES.get(error_type)

    if scope and not no_evidence:
        print(f"  -> 规则信号交叉验证：error_type={error_type}, 历史重复={repeat_hit} "
              f"=> 触发【{scope}】重规划")
        return {"error_type": error_type, "replan_scope": scope,
                "replan_count": current_count + 1,
                "reflection_history": history + [f"[{error_type}] {note}{reason}"],
                "_repeat_hit": False}

    print(f"  -> 无定向重规划依据（error_type={error_type}, no_evidence={no_evidence}），"
          f"回退到【自由反思】")
    return {"error_type": "证据不足", "replan_count": current_count + 1,
            "reflection_history": history + [f"[证据不足] {note}{reason}"],
            "_repeat_hit": False}

def _build_retry_query(state: State) -> tuple:
    """补检索的查询改写：默认把首轮未覆盖的对象追加进查询。

    同一条 query 会命中同一份磁盘缓存（缓存键含 query），等于没有重试 ——
    这是"补检索永远救不回来"的根因。默认改写，可用 RETRY_QUERY_MODE 切回对照。
    """
    topic = (state.get("topic") or "").strip()
    mode = (RETRY_QUERY_MODE or "missing_term").lower()

    if mode == "same":
        return build_query(topic), "same"
    if mode == "drop_lang":
        return topic, "drop_lang"

    coverage = ((state.get("run_stats") or {}).get("reflection") or {}).get("coverage") or {}
    missing = [m for m in (coverage.get("missing") or []) if m]
    if not missing:
        missing = [t for t in (state.get("subtasks") or []) if t]
    if missing:
        return build_query(f"{topic} {missing[0]}"), f"missing_term:{missing[0]}"
    return build_query(f"{topic} 补充资料 最新进展"), "missing_term:fallback"


def free_reflection_node(state: State):
    print("【自由反思节点】证据不足，先补检索再决定策略...")

    topic = state["topic"]
    retry_query, retry_mode = _build_retry_query(state)
    print(f"  -> [补检索] 改写查询：'{retry_query}'（策略 {retry_mode}）")
    context, sources, results, retry_stats = search_with_evidence(topic, query=retry_query)
    retry_stats = {**retry_stats, "retry_query": retry_query, "retry_mode": retry_mode}

    if context:
        print("  -> [补检索] 成功，基于新证据重写")
        prompt = prompts.FREE_REFLECT_SUCCESS_PROMPT.format(topic=topic, context=context)
        used_fallback = False
    else:
        print("  -> [补检索] 仍失败，走模型知识兜底并声明")
        prompt = prompts.FREE_REFLECT_FAIL_PROMPT.format(topic=topic)
        used_fallback = True

    # 把最终判定写回 run_stats.reflection，保证 JSON 与终端一致
    def _merge_reflection(verdict: str, gated_by: str, final_reason: str,
                          result_chars: int, evidence_count: int,
                          gate: dict = None) -> dict:
        run_stats = {**(state.get("run_stats") or {}), "search_retry": retry_stats}
        # 补检索换了 query，论文里要能看出"重试用的检索式"确实与首轮不同
        run_stats["search"] = {**(run_stats.get("search") or {}),
                               "retry_query": retry_query,
                               "retry_mode": retry_mode}
        prev_refl = dict(run_stats.get("reflection") or {})
        prev_refl.update({
            "verdict": verdict,
            "gated_by": gated_by,
            "decision_source": gated_by,
            "repeat_hit": bool(prev_refl.get("repeat_hit", False)),
            "llm_calls": llm_call_count(),
            "final_reason": final_reason,
            "evidence_count": evidence_count,
            "result_chars": result_chars,
            "pass_kind": _pass_kind(verdict),
        })
        if gate:
            prev_refl.update({
                "has_source_line": gate["has_source_line"],
                "cited_count": gate["cited_count"],
                "citation_ratio": gate["citation_ratio"],
                "len_ok": gate["len_ok"],
                "body_chars": gate.get("body_chars", 0),
                "len_full": gate.get("len_full", 0),
                "cited_loose": gate.get("cited_loose", []),
                # v2.2 完整性与引用率：补检索成功的回答同样要能逐条复查
                "citation_ratio_ok": gate.get("citation_ratio_ok", True),
                "truncated": gate.get("truncated", False),
                "truncation_hits": gate.get("truncation_hits", []),
            })
        run_stats["reflection"] = prev_refl
        return run_stats

    try:
        content = safe_llm_invoke(prompt, use_critic=False)
        if used_fallback:
            final_error_type = "降级通过"
            final_reason = "补检索仍失败，降级收尾"
            # 仍算一份 gate：声明式兜底也要留下正文长度/引用/截断事实
            gate = content_gate(content or "", sources or [], results or [])
        else:
            # 补检索成功也要过同一套门控：拿得到资料 ≠ 用了资料
            gate = content_gate(content, sources, results)
            gate_state = {**state, "result": content, "used_fallback": False}
            final_error_type, final_reason, gate_hit = _apply_pass_gate(
                gate_state, "通过", "补检索成功，基于新证据重写", gate)
            if gate_hit:
                print(f"  -> [门控] {gate_hit}，判定「{final_error_type}」")
            # v2.2：同样的反向纠偏，避免补检索成功的半截话被记成否决
            rescued, final_reason, rescue_rule = _apply_degrade_rescue(
                gate_state, final_error_type, final_reason, gate)
            if rescue_rule:
                print(f"  -> [降级救援] {rescue_rule}，判定「{rescued}」")
                final_error_type = rescued
        run_stats = _merge_reflection(final_error_type, "free_reflection",
                                      final_reason, _body_chars(content or ""),
                                      len(results), gate)
        return {"result": content, "error_type": final_error_type,
                "sources": sources if context else state.get("sources", []),
                "evidence": results, "used_fallback": used_fallback,
                "run_stats": run_stats}
    except LLMInfraError as e:
        print(f"  -> [自由反思] LLM 仍不可用: {e}")
        run_stats = _merge_reflection(
            "降级通过", "free_reflection",
            f"自由反思 LLM 不可用，保留上一轮结果：{str(e)[:80]}",
            _body_chars(state.get("result") or ""), len(results))
        return {"result": state.get("result", "") + "\n\n【声明】反思阶段 LLM 不可用，保留上一轮结果。",
                "error_type": "降级通过", "used_fallback": True,
                "run_stats": run_stats}
    except Exception as e:
        print(f"  -> [自由反思] 未知异常: {e}")
        run_stats = _merge_reflection(
            "降级通过", "free_reflection",
            f"自由反思未知异常，保留上一轮结果：{str(e)[:80]}",
            _body_chars(state.get("result") or ""), len(results))
        return {"result": state.get("result", "") or "自由反思失败，系统终止。",
                "error_type": "降级通过", "used_fallback": True,
                "run_stats": run_stats}
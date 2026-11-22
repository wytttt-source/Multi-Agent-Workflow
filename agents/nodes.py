#核心的 4 个节点
import re
import json
from typing import List

from config import (State, PASS_MIN_CITED, PASS_MIN_CHARS, RETRY_QUERY_MODE,
                    DISABLE_CONTENT_GATE, ENABLE_COVERAGE_RULE,
                    ENABLE_DEGRADE_RESCUE, ENABLE_REFLECTION, ENABLE_REPEAT_RULE,
                    ENABLE_REWRITE_PATH, ENABLE_TRUNCATION_RULE, REPAIR_POLICY,
                    INJECT_FAULT, REQUIRED_OBJECTS)
from utils import (safe_llm_invoke, LLMInfraError, LLMParseError, content_gate,
                   _body_chars, _body_text, llm_call_count, answer_completeness)
from tools import (search_with_evidence, evidence_coverage, build_evidence,
                   _is_composite_topic, _clean_composite_topic, build_query)
import prompts

# 修复轮的最小正文长度（v2.4）：与自由反思**同一口径**。
# 旧口径只给自由反思的改写提示加了字数要求，局部修复沿用"生成"的简述式回答，
# v2.4a 记录里 H 臂 C 组修复后正文只有 127~158 字（刚过 120 字门，判卷也只卡 120），
# 而 R 臂同一道题重写到 657 字 —— 同一份证据、同一道题，差别只来自提示措辞，
# 不是"定向修复不如整篇重写"。门限必须两边一致，否则 C 组的结论无法归因。
REPAIR_MIN_CHARS = max(PASS_MIN_CHARS * 2, 200)

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
                    "repair_path": "",    # v2.3：路径标记同样按轮复位
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
                "repair_path": "",
                "sources": state.get("sources", []),
                "used_fallback": False, "llm_fail_count": state.get("llm_fail_count", 0),
            }
        else:
            print("  -> [规划] 未识别出可拆解的复合结构，回退为单任务处理")

    return {
        "result": "", "subtasks": state.get("subtasks", []),
        "replan_count": state.get("replan_count", 0),
        "replan_scope": "",
        "repair_path": "",
        "sources": state.get("sources", []),
        "used_fallback": False, "llm_fail_count": state.get("llm_fail_count", 0),
    }

# ================= 修复路径实验（v2.3）：受控缺陷注入 + 局部修复换式 =================
# 这一段的代码只在实验里起作用，默认（INJECT_FAULT=none）行为与 v2.2 完全一致。
_FAULTS = ("short", "no_cite", "truncate", "one_side")


def _active_fault(state: State) -> str:
    """当前轮生效的注入类型；空字符串表示不注入。

    只作用于首轮（replan_count == 0）：修复轮次必须看到"修过之后"的真实结果，
    否则注入会把每一轮都打断，实验就退化成"永远修不好"。
    """
    fault = (INJECT_FAULT or "none").strip().lower()
    if fault not in _FAULTS:
        return ""
    if int(state.get("replan_count", 0) or 0) > 0:
        return ""
    return fault


def _object_terms(topic: str) -> List[str]:
    """必答对象清单：优先用实验任务集注入的 REQUIRED_OBJECTS，否则回落到主题切分。

    回落时要在"的"处截断：`evidence_coverage` 会保留"Java的并发性能"这类修饰语，
    而答案里只可能出现"Java"，拿长串去匹配等于永远匹配不上（P2 的同一个坑）。
    """
    if REQUIRED_OBJECTS:
        return [t for t in REQUIRED_OBJECTS if t]
    terms = []
    for t in (evidence_coverage(topic, []).get("terms") or []):
        short = (t or "").split("的", 1)[0].strip()
        if len(short) >= 2 and short not in terms:
            terms.append(short)
    return terms


def _split_body(content: str) -> tuple:
    """拆出正文与【来源】段。门控只认正文，注入因此也必须落在正文里（P1 的根因）。"""
    head, sep, tail = (content or "").partition("【来源】")
    return head.rstrip(), (sep + tail if sep else "")


_FAULT_SENT_SPLIT = re.compile(r"(?<=[。！？!?])|\n+")
_FAULT_CLAUSE_SPLIT = re.compile(r"(?<=[，；、,;])")


def _drop_object_sentences(body: str, terms: List[str],
                           keep_min: int = PASS_MIN_CHARS) -> str:
    """删掉只谈指定对象的句子，制造"答案只讲了一侧"的真实缺陷（P3）。

    两级策略：
    1. 整句删（默认）——比较类答案通常一句讲一个对象，删得干净；
    2. 退到分句删——长答案常把两个对象写进同一句（"资料里有 Python…；也没有 Java…"），
       整句删会剩不下内容，于是按逗号/分号逐分句剔。
    删完不足 `keep_min` 字就原样返回：注入要制造"缺一侧"，
    不能把答案删成"过短"，两种缺陷必须可区分（由调用方如实记"未生效"）。
    """
    picked = [t.lower() for t in terms if t]
    if not picked:
        return body
    sentences = [s for s in _FAULT_SENT_SPLIT.split(body) if s.strip()]
    hit = lambda s: any(t in s.lower() for t in picked)   # noqa: E731

    kept = [s for s in sentences if not hit(s)]
    if len(kept) >= 3 and len("".join(kept).strip()) >= keep_min:
        return "".join(kept).strip()

    rebuilt = []
    for sent in sentences:
        clauses = [c for c in _FAULT_CLAUSE_SPLIT.split(sent) if c.strip()]
        remain = "".join(c for c in clauses if not hit(c))
        if remain.strip():
            rebuilt.append(remain)
    text = "".join(rebuilt).strip()
    return text if len(text) >= keep_min else body


def _inject_fault_content(content: str, state: State) -> tuple:
    """首轮答案受控缺陷：short（过短）/ no_cite（无引用）/ truncate（截断）/ one_side。

    注入的是"真实世界里会出现的坏形态"，不是随机字符串：
    - short    只保留半段正文 → 触发字数门；
    - no_cite  砍掉【来源】段并明说没列来源 → 触发引用门（会被降级救援接住）；
    - truncate 停在连接词上 → 触发 v2.2 的确定性截断规则；
    - one_side 删掉"另一侧对象"的句子 → 触发答案侧对象覆盖缺口。

    v2.4 修正两处"注入实际没生效"的实现缺陷（详见 docs/反思判定状态机-改动清单.md 附录 D）：
    - D5 旧实现按**全文**长度算截断点，而内容门只认【来源】之前的正文；切点常常落在
      来源清单里，被 `_body_text()` 整段剔除，于是 C1/C4 两次注入从未真的截断过。
    - D6 旧 one_side 只改检索、不改答案，模型靠自身知识补齐另一侧，缺陷到不了答案层。
    """
    fault = _active_fault(state)
    if not fault:
        return content or "", []
    body, source_block = _split_body(content)
    if fault == "short":
        # 只截正文：按全文截同样会把来源段整个切掉，缺陷就变成"无引用"了
        return body[: max(PASS_MIN_CHARS // 2, 40)], ["short"]
    if fault == "no_cite":
        return body + "\n\n（本节未列出来源链接）", ["no_cite"]
    if fault == "truncate":
        cut = body[: max(PASS_MIN_CHARS + 20, int(len(body) * 0.6))]
        truncated = cut.rstrip("。！？!?\n，、；： ") + "，以及"
        # 保留来源段：缺陷只落在"内容不完整"上，不牵连引用，两个信号才可分离
        return truncated + (("\n\n" + source_block) if source_block else ""), ["truncate"]
    if fault == "one_side":
        drop = _object_terms(state.get("topic") or "")[1:]
        degraded = _drop_object_sentences(body, drop) if drop else body
        if degraded == body:
            return content or "", []
        return degraded + (("\n\n" + source_block) if source_block else ""), [
            "one_side:answer_drop:" + "/".join(drop)]
    return content or "", []


def _retry_query_for(search_topic: str, state: State) -> tuple:
    """局部修复的换式检索：保证重试 query 与首轮不同。

    为什么必须换：`search_with_evidence` 的缓存键含 query，subtasks 在局部重规划里不变，
    不换 query 就会命中首轮的同一份缓存 —— 这是"重试了但没带来任何新证据"的根因，
    也是 E4/E5 里 2/2 空转的机制来源（P0-5 当时只修了自由反思那一侧）。
    """
    mode = (RETRY_QUERY_MODE or "missing_term").lower()
    if mode == "same":
        return build_query(search_topic), "same"
    if mode == "drop_lang":
        return search_topic, "drop_lang"
    coverage = ((state.get("run_stats") or {}).get("reflection") or {}).get("coverage") or {}
    missing = [m for m in (coverage.get("missing") or []) if m]
    term = missing[0] if missing else ""
    if term and term not in search_topic:
        return build_query(f"{search_topic} {term}"), f"missing_term:{term}"
    return build_query(f"{search_topic} 最新进展 官方文档"), "extra_terms"


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
    retry_round = int(state.get("replan_count", 0) or 0)
    # 上一轮的 run_stats：v2.3 的 repair.rounds / 检索明细要跨轮累加，
    # 否则每轮回写 `{"search": ...}` 会把上一次的轨迹一起冲掉。
    prev_stats = dict(state.get("run_stats") or {})
    prev_injected = [m for m in (prev_stats.get("injected") or []) if m]
    # v2.5 第二步：rewrite = 用**已有证据**重写答案，本轮不发检索请求。
    # R1 与 R2 的分工就落在这里：R1 认为"缺的东西证据里也没有"（必须换检索式），
    # R2 认为"证据里有、只是没写"（再检索一次也只会拿回同一批资料）。
    # 若 rewrite 仍去检索，两条规则的行为就没差别，"改答案"与"换检索式"再也分不开。
    rewrite_only = (state.get("repair_path") or "") == "rewrite"
    # v2.3 受控缺陷：单侧证据（只作用于首轮）。做法是"只检索第一个对象"，
    # 而不是事后裁剪结果——实跑发现"结果只提到 Python"几乎不存在（对比类文章都会提到两者），
    # 事后裁剪会把注入变成空操作（P-B0 首轮直接判通过就是这种假阴性）。
    injected = []
    if not rewrite_only and _active_fault(state) == "one_side" and search_topics:
        # v2.4：对象清单与判卷同源（REQUIRED_OBJECTS），回落才用主题切分
        terms = _object_terms(topic)
        if len(search_topics) > 1:
            search_topics = search_topics[:1]
            injected.append("one_side:first_subtask")
        elif len(terms) >= 2:
            search_topics = [terms[0]]
            injected.append(f"one_side:first_object:{terms[0]}")
        if injected:
            print(f"  -> [注入] 单侧证据：首轮只检索 '{search_topics[0]}'")

    if rewrite_only:
        # 不换检索式、不新增资料：把上一轮留下的结构化证据重新装配成 context。
        # 用 build_evidence 而不是自己拼字符串，是为了让"来源清单"与首轮同源，
        # 引用可核验性（content_gate 的 URL 校验）才有意义。
        base_evidence = [r for r in (state.get("evidence") or []) if r]
        context, sources = build_evidence(base_evidence)
        per_query = []      # 本轮零检索：per_query 必须为空，成本表才不会重复计数
        print(f"  -> [执行] 重写路径：跳过检索，复用已有 {len(base_evidence)} 条证据"
              f"（装配 {len(context)} 字）")
    else:
        for st in search_topics:
            retry_query, retry_mode = "", ""
            if retry_round > 0:
                # v2.3：局部修复必须换检索式，否则命中首轮缓存 = 零新证据的空转
                retry_query, retry_mode = _retry_query_for(st, state)
                print(f"  -> [执行] 第 {retry_round} 次修复检索，换式：'{retry_query}'（{retry_mode}）")
            else:
                print(f"  -> [执行] 检索：{st}")
            context, sources, results, stats = search_with_evidence(st, query=retry_query)
            if retry_round > 0:
                stats = {**stats, "retry_query": retry_query,
                         "retry_mode": retry_mode, "retry_round": retry_round}
            per_query.append(stats)
            if context:
                all_contexts.append(f"=== 检索：{st} ===\n{context}")
                all_sources.extend(sources)
            all_results.extend(results)

    # 逐查询明细跨轮累计（`repair.cost.searches` 与"到底检索了几次"以它为准），
    # 但 raw/gated/elapsed 这类聚合量仍按**本轮**统计，保持与 v2 记录同口径
    round_per_query = list(per_query)
    prev_search = prev_stats.get("search") or {}
    all_per_query = list(prev_search.get("per_query") or []) + round_per_query

    # rewrite 分支的 context/sources 已由 build_evidence 装配好，这里不能再覆盖成空串
    if not rewrite_only:
        context = "\n\n".join(all_contexts)
        sources = list(dict.fromkeys(all_sources))

    # 量化统计：落盘到 data/runs/*.json，README 与论文表格的数字统一取自这里
    search_stats = {
        "per_query": all_per_query,
        # v2.5：本轮是否跳过检索（rewrite 路径）。per_query 变短是**设计**而不是检索失败，
        # 显式落一个布尔，避免事后把"没检索"读成"检索没结果"。
        "skipped": bool(rewrite_only),
        "raw": sum(s.get("raw", 0) for s in round_per_query),
        "gated": sum(s.get("gated", 0) for s in round_per_query),
        "filtered": sum(s.get("filtered", 0) for s in round_per_query),
        "evidence_chars": len(context),
        "sources": len(sources),
        "providers": sorted({p for s in round_per_query for p in s.get("providers", [])}),
        "cache": [s.get("cache") for s in round_per_query],
        "cache_sources": [s.get("cache_source", s.get("cache")) for s in round_per_query],
        "degraded": any(s.get("degraded") for s in round_per_query),
        # v2.2：过滤是否真的跑了（applied=False 表示 GLM 故障或开关关闭）
        "llm_filter": [s.get("llm_filter", {}) for s in round_per_query],
        "elapsed_s": round(sum(s.get("elapsed_s", 0) for s in round_per_query), 2),
        "evidence_path": "; ".join(
            s.get("evidence_path", "") for s in round_per_query if s.get("evidence_path")),
    }
    # 保留上一轮已有的 run_stats：repair.rounds 已按轮累加，这里只替换本轮检索统计
    run_stats = {**prev_stats, "search": search_stats,
                 "injected": prev_injected + injected}

    if rewrite_only:
        print(f"  -> [执行] 重写路径不检索：沿用累计 {len(all_per_query)} 次检索的既有证据")
    elif all_results:
        avg_summary = sum(len(r.get("summary") or "") for r in all_results) / len(all_results)
        avg_content = sum(len(r.get("content") or "") for r in all_results) / len(all_results)
        providers = sorted({r.get("provider", "?") for r in all_results})
        print(f"  -> [执行] 证据统计：检索 {len(all_results)} 条 / {len(sources)} 个链接 / "
              f"装配 {len(context)} 字（平均摘要 {avg_summary:.0f} 字，平均正文 {avg_content:.0f} 字，"
              f"provider={','.join(providers)}）")
    else:
        print("  -> [执行] 证据统计：0 条来源")

    # rewrite 不检索就没有"本轮检索失败"一说，不能拿来源清单（只有"标题 | URL"）
    # 去当资料塞进提示词
    if not context and prev_sources and not rewrite_only:
        print("  -> [检索] 本轮失败，复用历史来源")
        context = "\n".join(prev_sources)
        sources = prev_sources

    previous_errors = "\n".join(state.get("reflection_history", []))

    if rewrite_only:
        # 重写提示把字数口径写死成 REPAIR_MIN_CHARS：与局部修复、自由反思同一把尺子，
        # 否则"重写"又能靠刚好压线过关，三臂的正文字数就不可比了
        prompt = prompts.REWRITE_ONLY_PROMPT.format(
            topic=topic, context=context, min_chars=REPAIR_MIN_CHARS)
    elif context:
        if subtasks:
            prompt = prompts.GENERATE_COMPOSITE_PROMPT.format(topic=topic, context=context)
        else:
            prompt = prompts.GENERATE_SINGLE_PROMPT.format(topic=topic, context=context)
    else:
        print("  -> [生成] 无外部检索证据，走模型知识兜底并声明")
        prompt = prompts.GENERATE_FALLBACK_PROMPT.format(topic=topic)

    if previous_errors:
        prompt += f"\n注意：上一次的尝试因为以下原因被指出了问题：\n{previous_errors}\n请务必避免重蹈覆辙。"

    if retry_round > 0 and not rewrite_only:
        # 修复轮的字数/质量要求：只在**修复轮**加，首轮口径不变（首轮缺陷由 INJECT_FAULT 控制）。
        # 不加这一条的后果是"修复"退化成为凑门限的补丁式重写（C 组实测只有 127 字），
        # 于是 H/F 与 R 的差距里混进了"提示不同"这个与路径无关的变量。
        # rewrite 轮不加：它的提示词已经写死了同一套要求，重复贴一遍只会让提示更长。
        prompt += (
            f"\n【修复要求】这是一次修复性重写，不是打补丁：正文不少于 {REPAIR_MIN_CHARS} 字，"
            "必须把上一轮被指出的问题彻底补上（被截断的要把话写完、缺的对象要逐个讲清、"
            "该有引用的要给出可核验 URL），"
            "但不要靠「资料未提供 XX」这类声明去凑对象覆盖。"
        )

    if injected and _active_fault(state) == "one_side":
        # v2.4：注入要在**答案侧**成立。只砍检索不够——模型会用自身知识把另一侧补齐，
        # 于是"检索缺口"变成"其实答得挺全"（旧 48 次里 A/B 组 8/8 判卷合格）。
        # 这里把"只有一侧资料"如实告诉生成侧，答案才会真的缺对象，修复路径也才有事可做。
        others = [t for t in _object_terms(topic) if t != search_topics[0]]
        prompt += (
            f"\n【检索限制】本轮只检索到「{search_topics[0]}」这一侧的资料"
            + (f"，其他比较对象（{'、'.join(others)}）没有可用证据" if others else "")
            + "。请只基于已获得的资料作答：不要用你自己的知识补充其他对象，"
            "也不要在正文里提及或讨论其他对象（「缺少某侧资料」这类说明由系统判定，不必写）。")

    try:
        content = safe_llm_invoke(prompt, use_critic=False)
        # v2.3 受控缺陷注入（只作用于首轮）：让"修复路径"有确定的样本可比
        content, content_injected = _inject_fault_content(content, state)
        if content_injected:
            print(f"  -> [注入] 首轮受控缺陷：{content_injected}")
        # error_type 置空：清掉上一轮遗留的"系统繁忙"，否则反思节点会误判为执行侧仍故障
        return {"result": content, "sources": sources, "error_type": "",
                "replan_scope": "",   # P2-3：每轮执行后复位，路由不再读到上一轮的"全局"
                "repair_path": "",    # v2.3：路径标记同样按轮复位
                "evidence": all_results or state.get("evidence", []),
                "run_stats": {**run_stats,
                              "injected": prev_injected + injected + content_injected,
                              "generation": {
                                  "prompt_chars": len(prompt),
                                  "result_chars": len(content)}},
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


def _repair_outcome(rounds: list) -> str:
    """把"首轮 → 终态"压成修复结果四态（论文里修复率/无效重试率的分子分母来源）。

    pass_first_try : 首轮就通过（说明这道题不需要修复）
    fixed          : 首轮否决，最终严格通过
    partial        : 首轮否决，最终降级通过（可读但缺资料）
    degraded_unchanged : 首轮降级，最终仍降级（没变好也没变坏）
    failed         : 最终仍被否决（含撞上限）
    """
    if not rounds:
        return "unknown"
    first = _pass_kind(rounds[0].get("verdict") or "")
    final = _pass_kind(rounds[-1].get("verdict") or "")
    if first == "strict":
        return "pass_first_try"
    if final == "strict":
        return "fixed"
    if final == "degraded":
        return "partial" if first == "rejected" else "degraded_unchanged"
    return "failed"


def _repair_trace(repair: dict, verdict: str, path: str, gate: dict = None,
                  reason: str = "", decision_source: str = "",
                  repeat_hit: bool = False, rescued_from: str = "",
                  rule: str = "", facts: dict = None) -> dict:
    """v2.3：把本轮「错在哪 → 走哪条修复路径 → 这一轮的判定」追加进 run_stats.repair。

    为什么单独建一块：`run_stats.reflection` 只保留**最后一次**判定，跨臂比较时
    看不出"首轮错在哪、第几轮修好的、走了几条路"。这里按轮追加，成本用累计
    llm_calls 的差值算（eval 脚本里做差，不在节点里做，避免口径漂移）。
    """
    repair = dict(repair or {})
    rounds = list(repair.get("rounds") or [])
    rounds.append({
        "round": len(rounds),
        "verdict": verdict,
        "pass_kind": _pass_kind(verdict),
        "path": path or "",
        "decision_source": decision_source,
        "reason": (reason or "")[:160],
        "repeat_hit": bool(repeat_hit),
        "rescued_from": rescued_from,
        # v2.5：规则编号 + 结构化选路事实（不改变判定，只让"凭什么选这条路"可聚合）
        #   rule           本次走这条路的依据（R1~R5，见 REPAIR_RULES）
        #   answer_missing 答案侧没讲到的必答对象（与判卷同源 REQUIRED_OBJECTS）
        #   evidence_gap   其中证据里也没有的对象（只有这批需要换检索式）
        #   evidence_count 本轮证据条数（便于核对"无证据"判据是否成立）
        "rule": rule or "",
        "answer_missing": (facts or {}).get("missing", []),
        "evidence_gap": (facts or {}).get("evidence_gap", []),
        "evidence_count": (facts or {}).get("evidence_count"),
        # v2.4：没跑门控的轮次（如纯覆盖率判定的轮）必须写 null 而不是 0，
        # 否则"首轮正文 0 字""引用 0 条"会被误读成答案为空/没引用
        "gate_measured": bool(gate),
        "cited_count": (gate or {}).get("cited_count") if gate else None,
        "body_chars": (gate or {}).get("body_chars") if gate else None,
        "truncated": bool((gate or {}).get("truncated", False)) if gate else None,
        "llm_calls": llm_call_count(),
    })
    repair.update({
        "policy": REPAIR_POLICY,
        "inject_fault": INJECT_FAULT,
        "rounds": rounds,
        "first_verdict": rounds[0]["verdict"],
        "final_verdict": rounds[-1]["verdict"],
        "outcome": _repair_outcome(rounds),
    })
    return repair


def _reflection_stats(state: State, coverage, verdict: str, reason: str,
                      gated_by: str, gate: dict = None, repeat_hit: bool = False,
                      decision_source: str = None, rescue: tuple = None,
                      path: str = "", rule: str = "") -> dict:
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
    # v2.3：按轮留痕（policy / 路径 / 判定 / 累计调用），跨臂比较的唯一数据来源
    # v2.5：附上规则编号与选路事实（确定性计算，不调 LLM，也不改判定）
    run_stats["repair"] = _repair_trace(
        run_stats.get("repair"), verdict, path, gate=gate, reason=reason,
        decision_source=stats["decision_source"], repeat_hit=repeat_hit,
        rescued_from=(rescue[0] if rescue else ""),
        rule=rule, facts=_decision_facts(state))
    return {"run_stats": run_stats}


def _hard_content_facts(gate: dict) -> List[str]:
    """答案层的"硬事实"：与主题怎么拆无关，重拆也救不回，判定时应优先于语义结论。

    为什么单独抽出来：旧 48 次里 C 组（截断/过短）被裁判 LLM 判成「分解不合理」——
    因为提示词里"证据覆盖率 < 100% 即分解不合理"同时成立，而"截断"不在 LLM 的硬规则里。
    于是修复路径走向全局重拆，改完还是截断。这三条都能用确定性方式量出来，让它们先说话。
    """
    hits = []
    if not gate.get("len_ok", True):
        hits.append(f"正文仅 {gate.get('body_chars', 0)} 字")
    if ENABLE_TRUNCATION_RULE and gate.get("truncated"):
        hits.append("回答截断")
    if not gate.get("no_fabricated_url", True):
        hits.append("疑似编造 URL")
    return hits


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
        decision = _decide_next(state, "内容不合格", "字数太少")
        # 这条早退路径不落 repair trace（保持 v2.2 行为），规则编号也就不必进 state
        decision.pop("_rule", "")
        return decision

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

    if ENABLE_COVERAGE_RULE and evidence_items:
        # 证据侧覆盖率：v2.4 起只作**提示**与"该补哪个对象"的依据，不再单独做终判
        # 证据全文一次拼好：既有子任务口径要用，也要判断"缺的对象在证据里到底有没有"
        ev_blob = " ".join(
            " ".join([r.get("title") or "", r.get("summary") or "",
                      r.get("snippet") or "", (r.get("content") or "")[:200]])
            for r in evidence_items
        )
        if has_subtasks:
            # 已拆解：子任务字面量匹配证据 blob（保留 v2.1 口径，作为证据侧信号）
            hit = [st for st in subtasks if st in ev_blob]
            missing = [st for st in subtasks if st not in ev_blob]
            coverage = {"terms": subtasks, "hit": hit, "missing": missing,
                        "ratio": len(hit) / len(subtasks) if subtasks else 1.0,
                        "count": len(evidence_items)}
        else:
            coverage = evidence_coverage(topic, evidence_items)
        coverage["scope"] = "evidence"
        coverage_info = coverage
        balance_str = ""
        if coverage.get("per_term_counts"):
            # 偏侧检测：字面覆盖 100% 但证据全压在某一侧时，这条才是真正的"分解不合理"信号
            balance_str = (f"；偏侧检测：{coverage['per_term_counts']}，"
                           f"balance={coverage.get('balance', 1.0):.2f}")
        evidence_stats = (
            f"证据 {coverage['count']} 条；需覆盖对象 {coverage['terms'] or ['无']}；"
            f"已覆盖 {coverage['hit'] or ['无']}；未覆盖 {coverage['missing'] or ['无']}；"
            f"覆盖率 {coverage['ratio']:.0%}{balance_str}"
        )
        print(f"  -> [反思] {evidence_stats}")

        if REQUIRED_OBJECTS:
            # v2.4 终判改用**答案侧**对象覆盖，证据侧覆盖率只当提示。
            # 旧口径拿"子任务整串"去逐字匹配证据 blob，而这类串几乎不可能逐字出现：
            # 48 份旧记录里有 54/100 轮因此判「分解不合理」，其中 8 轮判卷其实合格
            # （F 臂"67% 误判"是**规则**缺陷，不是 fixed 路径的能力差）。
            answer_cov = answer_completeness(state.get("result") or "",
                                             list(REQUIRED_OBJECTS))
            answer_cov["scope"] = "answer"
            coverage_info = answer_cov
            if answer_cov["missing"]:
                miss = answer_cov["missing"]
                # 缺的对象在**证据里也没有**：文本重写救不回来，只能换检索式补证据。
                # 旧口径一律判"内容不合格"→局部重写，于是 A/B 组先白白重写一轮
                # （48 份旧记录里 H 臂 A/B 平均修复轮次 3.0，比 R 臂多一轮）。
                gap = [m for m in miss if m not in ev_blob]
                if gap:
                    error_type = "证据不足"
                    reason = (f"答案与证据均缺必答对象：{'、'.join(gap)}"
                              f"（属检索缺口，重写文本补不出内容，需换检索式补证据）")
                    print(f"  -> [反思] {reason}，判证据不足（答案侧+证据侧都缺）")
                else:
                    error_type = "内容不合格"
                    reason = (f"答案未覆盖 {len(miss)}/{len(answer_cov['terms'])} "
                              f"个必答对象：{'、'.join(miss)}"
                              f"（证据里已有这些对象，属生成缺口，局部重写即可）")
                    print(f"  -> [反思] {reason}，判内容不合格（证据够、答案漏）")
                # 这一轮没走语义裁判，但门控事实仍要量出来落盘：
                # 否则对照表里这一轮记为"正文 null / 引用 null"，看起来像没测
                gate = content_gate(state.get("result", ""), state.get("sources", []),
                                    evidence_items)
                decision = _decide_next(state, error_type, reason)
                repeat_hit = decision.pop("_repeat_hit", False)
                rule = decision.pop("_rule", "")         # 规则编号只进 trace，不进 state
                path = decision.get("repair_path", "")   # 不 pop：路由要读这个字段
                return {**decision,
                        **_reflection_stats(state, answer_cov, error_type, reason,
                                            "coverage_rule", gate=gate,
                                            repeat_hit=repeat_hit, path=path,
                                            rule=rule)}
        elif has_subtasks and coverage["ratio"] < 1.0:
            reason = f"子任务覆盖率仅 {coverage['ratio']:.0%}，未覆盖：{'、'.join(coverage['missing'])}"
            print("  -> [反思] 已拆解但子任务覆盖率 < 100%，判分解不合理")
            decision = _decide_next(state, "分解不合理", reason)
            repeat_hit = decision.pop("_repeat_hit", False)
            rule = decision.pop("_rule", "")
            path = decision.get("repair_path", "")
            return {**decision,
                    **_reflection_stats(state, coverage, "分解不合理", reason,
                                        "coverage_rule", repeat_hit=repeat_hit,
                                        path=path, rule=rule)}
        elif (not has_subtasks and is_composite and coverage.get("terms")
              and coverage["ratio"] < 1.0):
            print("  -> [反思] 复合主题证据覆盖率 < 100%，跳过 LLM 直接判分解不合理")
            reason = f"证据覆盖率仅 {coverage['ratio']:.0%}，未覆盖：{'、'.join(coverage['missing'])}"
            decision = _decide_next(state, "分解不合理", reason)
            repeat_hit = decision.pop("_repeat_hit", False)
            rule = decision.pop("_rule", "")
            path = decision.get("repair_path", "")
            return {**decision,
                    **_reflection_stats(state, coverage, "分解不合理", reason,
                                        "coverage_rule", repeat_hit=repeat_hit,
                                        path=path, rule=rule)}
    else:
        if not ENABLE_COVERAGE_RULE:
            evidence_stats = (f"证据 {len(evidence_items)} 条"
                              f"（覆盖率规则已关闭：ENABLE_COVERAGE_RULE=False）")
        elif is_composite:
            evidence_stats = "证据 0 条（本轮未获得证据，覆盖率检查跳过）"
        else:
            evidence_stats = f"证据 {len(evidence_items)} 条（单主题，无需比较对象覆盖率检查）"
        print(f"  -> [反思] {evidence_stats}")

    # 覆盖率口径随 REQUIRED_OBJECTS 切换：有必答对象就按答案侧讲，没有就保持历史口径
    coverage_rule_hint = (prompts.COVERAGE_RULE_HINT_ANSWER if REQUIRED_OBJECTS
                          else prompts.COVERAGE_RULE_HINT_LEGACY)
    prompt = prompts.REFLECTION_PROMPT.format(
        topic=topic, result=state['result'],
        sources="\n".join(state.get('sources', [])) or "（无）",
        evidence_stats=evidence_stats, composite_check_hint=composite_check_hint,
        coverage_rule_hint=coverage_rule_hint,
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
    # v2.4：答案层硬事实优先于「分解不合理」（见 _hard_content_facts 的说明）
    pre_gate = ""
    if llm_error_type == "分解不合理" and not DISABLE_CONTENT_GATE:
        hard = _hard_content_facts(gate)
        if hard:
            pre_gate = "，".join(hard)
            print(f"  -> [门控] 确定性内容事实（{pre_gate}）优先于「分解不合理」，改判「内容不合格」")
            llm_error_type, reason = "内容不合格", f"确定性内容事实：{pre_gate}"
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

    gated_by = "llm" if (error_type == llm_error_type and not pre_gate) else "gate"
    decision = _decide_next(state, error_type, reason)
    repeat_hit = decision.pop("_repeat_hit", False)
    rule = decision.pop("_rule", "")
    path = decision.get("repair_path", "")
    return {**decision,
            **_reflection_stats(state, coverage_info, error_type, reason, gated_by,
                                gate, repeat_hit=repeat_hit, rescue=rescue,
                                path=path, rule=rule)}

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
        rule = decision.pop("_rule", "")
        path = decision.get("repair_path", "")
        return {**decision,
                **_reflection_stats(state, coverage, error_type, reason_full,
                                    "rule_fallback", repeat_hit=repeat_hit,
                                    path=path, rule=rule)}

    # 兜底路径复用同一套确定性门控：没有可核验引用，同样不能记「通过」
    gate = content_gate(result, state.get("sources", []), state.get("evidence", []))
    error_type, reason_full, gate_hit = _apply_pass_gate(state, "通过", reason_full, gate)
    if gate_hit:
        print(f"  -> [门控] 规则兜底后 {gate_hit}，判定「{error_type}」")

    gated_by = "gate" if gate_hit else "rule_fallback"
    decision = _decide_next(state, error_type, reason_full)
    repeat_hit = decision.pop("_repeat_hit", False)
    rule = decision.pop("_rule", "")
    path = decision.get("repair_path", "")
    return {**decision,
            **_reflection_stats(state, coverage, error_type, reason_full, gated_by,
                                gate, repeat_hit=repeat_hit, path=path, rule=rule)}

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


_PATH_LABEL = {"local": "局部修复", "global": "全局重拆", "free": "自由反思",
               "rewrite": "定向重写"}
# 旧口径（中文 scope）→ 新路径枚举；保留中文标签是为了不改既有记录/文档用语
_SCOPE_PATH = {"局部": "local", "全局": "global"}

# ================= v2.5 修复路径决策规则表 =================
# 为什么要有规则编号：论文要回答"混合策略凭什么选这条路"，靠 reason 字符串没法聚合，
# 也看不出"哪条规则触发了几次、各自修好几题"。这里只给既有分支**命名**：
# 第一步（本版）不改任何判定 —— experiments/replay_rules.py 用 v2.4 的 48 份记录复算，
# 46 个决策轮零翻转；第二步才把 R2 从 local 改成 rewrite（改在 _rule_and_desired 一处）。
REPAIR_RULES = {
    "R1_evidence_gap": "证据缺口：缺的必答对象在证据里也没有（或本轮无证据）→ 换检索式补证据",
    "R2_answer_gap": "答案缺口：证据里有缺的对象 / 过短 / 截断 / 缺引用 → 定向重写答案",
    "R3_structure": "结构缺口：分解不合理（复合主题覆盖不足）→ 全局重拆",
    "R4_repeat": "同类错误重复：上一轮同类型修复没带来新信息 → 换检索式重写",
    "R5_infra": "基础设施故障：系统繁忙 / 执行崩溃 → 同路重试（由 main.py 路由处理）",
}


def _decision_facts(state: State) -> dict:
    """选路需要的事实（确定性、可复算、不调 LLM）：只用于选路与留痕，不参与判定。

    missing      答案侧没讲到的必答对象（与判卷同源：REQUIRED_OBJECTS）
    evidence_gap 其中连证据里也没有的对象 —— 只有这批靠换检索式才救得回来；
                 证据里已有的那批属于"生成漏了"，重写答案即可（R2）
    evidence_covers_required
                 有必答对象、且缺的对象在证据里**都有**（True → 该修答案，不该换检索式）

    为什么要这份事实：v2.4 已经用「缺的对象在不在证据里」区分证据缺口/答案缺口
    （见 reflection_node 的 coverage_rule 分支），但那是一次性的局部变量，
    run_stats 里只留下一句话。抽出来才能让"命名规则"与"落盘事实"同源。
    """
    evidence = state.get("evidence") or []
    ev_blob = " ".join(
        " ".join([r.get("title") or "", r.get("summary") or "",
                  r.get("snippet") or "", (r.get("content") or "")[:200]])
        for r in evidence)
    required = [t for t in REQUIRED_OBJECTS if t]
    missing, gap = [], []
    if required:
        comp = answer_completeness(state.get("result") or "", required)
        missing = comp.get("missing") or []
        gap = [m for m in missing if m not in ev_blob]
    return {"evidence_count": len(evidence), "required": required,
            "missing": missing, "evidence_gap": gap,
            "evidence_covers_required": bool(required) and not gap}


def _rule_and_desired(state, error_type: str, reason: str, repeat_fired: bool,
                      facts: dict = None):
    """按规则表给出「需要走哪条路」：返回 (desired, rule, facts)。

    与 v2.2~v2.4 的 if/else **逐条等价**，除 R2 一处（v2.5 第二步）：
        R4 重复 → free ；R3 分解不合理 → global ；
        内容不合格 且有证据 → rewrite（R2，v2.4 是 local）；其余（无证据 / 无类型）→ free（R1）。
    R2 改走 rewrite 的依据就是 facts：缺的必答对象在证据里**都有**（evidence_gap 为空），
    再检索一次只会拿回同一批资料，缺的是"写"而不是"找"。
    判断顺序一个没动，关掉 ENABLE_REWRITE_PATH 即回到 v2.4 的 local。
    """
    facts = facts or _decision_facts(state)
    no_evidence = (facts.get("evidence_count", 0) == 0) or any(
        hint in (reason or "") for hint in _NO_EVIDENCE_HINTS)
    if repeat_fired:
        return "free", "R4_repeat", facts
    if error_type == "分解不合理":
        return "global", "R3_structure", facts
    scope = _SCOPE_PATH.get(_REPLAN_TYPES.get(error_type, ""), "")
    if scope == "local" and not no_evidence:
        # v2.5 第二步：同一份证据重写答案，不再"换检索式再检索一遍"
        return ("rewrite" if ENABLE_REWRITE_PATH else "local"), "R2_answer_gap", facts
    return "free", "R1_evidence_gap", facts


def _apply_repair_policy(desired: str) -> str:
    """把「这条错误该走哪条路」映射成「策略允许走哪条路」——三臂对照的唯一切换点。

    fixed ：不允许自由反思，判断不清就退回定向修复（无类型的错 = 局部重试一次），
            这正是"按类型硬修"的旧做法，用来量化"只按分类修"的代价；
    free  ：不做定向修复，一律补检索重写，用来量化"不看依据、只换检索式"的代价；
    hybrid：有确定性依据走定向、判断不清走自由反思（v2.2 默认行为）。

    v2.5 第二步：fixed 只剩"不许转自由反思"这一条，不再把 R2 的 rewrite 压回 local ——
    rewrite 是"用已有证据重写答案"，全臂共用才能把"换检索式"与"改答案"两个变量分开；
    F 臂因此从"旧做法参照"变成"hybrid 去掉自由反思"的消融项。
    （free 臂仍是定义上的一律 free：它检验的正是"不看依据、只换检索式"，
    若 R2 也给它 rewrite，这条消融项就没有对照物了。）

    注意：这里只改"走哪条路"，不改判定标准，所以 R/L/C 二分类指标不受影响。
    """
    policy = (REPAIR_POLICY or "hybrid").strip().lower()
    if policy == "free":
        return "free"
    if policy == "fixed":
        return "local" if desired == "free" else desired
    return desired


def _decide_next(state: State, error_type: str, reason: str, facts: dict = None):
    """判定 → 修复路径决策。

    返回里含三个临时/权威键：
    - `repair_path`（local/global/free）：写进 state，路由函数**只认它**，不再靠 error_type 猜；
    - `_repeat_hit`：临时键，由调用方 pop 后写进 run_stats.reflection；
    - `_rule`（v2.5）：本次选路依据的规则编号（见 REPAIR_RULES），由调用方 pop 后
      写进 run_stats.repair.rounds[].rule —— 它只是**留痕**，不参与任何路由。

    分两步，是为了让"该走哪条路"（证据判断）与"允许走哪条路"（实验策略）可分离：
    第一步 `_rule_and_desired` 给依据与编号，第二步 `_apply_repair_policy` 按臂分叉。
    P1-1 的重复检测仍然只看历史，第 2 次起说明重规划没带来新信息，应转自由反思。
    """
    raw_type = (error_type or "").strip()
    error_type = _norm_type(raw_type)      # 存储 / 匹配同形，见 _TYPE_ALIASES
    if error_type in ("通过", "降级通过"):
        return {"error_type": error_type, "repair_path": "",
                "_repeat_hit": False, "_rule": ""}

    current_count = state.get("replan_count", 0)
    history = state.get("reflection_history", [])

    # 前缀匹配，与历史写入格式 f"[{error_type}] {reason}" 对齐
    repeat_times = sum(1 for h in history
                       if isinstance(h, str) and h.startswith(f"[{error_type}]"))
    repeat_hit = repeat_times > 0
    # `_repeat_hit` 的语义保持 v2.2：指"重复规则真的被触发"，规则关掉时记 False
    repeat_fired = bool(repeat_hit and ENABLE_REPEAT_RULE)

    # 归一发生了（如 检索失败→证据不足）时保留原判，便于论文里追溯裁判措辞
    note = f"原判{raw_type}：" if raw_type != error_type else ""

    # 第一步：这条错误"需要"哪种修复（与 policy 无关的纯证据判断）+ 规则编号
    desired, rule, facts = _rule_and_desired(state, error_type, reason,
                                             repeat_fired, facts)
    no_evidence = ((facts.get("evidence_count", 0) == 0)
                   or any(hint in (reason or "") for hint in _NO_EVIDENCE_HINTS))

    # 第二步：策略决定"允许"走哪条（v2.3 三臂对照在这里分叉）
    path = _apply_repair_policy(desired)
    if path != desired:
        print(f"  -> [修复策略] policy={REPAIR_POLICY}：本应走【{_PATH_LABEL[desired]}】，"
              f"改为【{_PATH_LABEL[path]}】（规则 {rule}）")

    if path == "free":
        tag = (f"{error_type}重复{repeat_times + 1}次"
               if repeat_fired else f"{note}{reason}")
        print(f"  -> 转【自由反思】换检索式（原判 {error_type}，规则 {rule}）")
        return {"error_type": "证据不足", "replan_scope": "", "repair_path": "free",
                "replan_count": current_count + 1,
                "reflection_history": history + [f"[证据不足] {tag}"],
                "_repeat_hit": repeat_fired, "_rule": rule}

    print(f"  -> 触发【{_PATH_LABEL[path]}】（error_type={error_type}，"
          f"规则={rule}，重复={repeat_hit}，无证据={no_evidence}）")
    return {"error_type": error_type,
            "replan_scope": "全局" if path == "global" else "局部",
            "repair_path": path,
            "replan_count": current_count + 1,
            "reflection_history": history + [f"[{error_type}] {note}{reason}"],
            "_repeat_hit": repeat_fired, "_rule": rule}

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

    # v2.4：重写长度必须**高于**门控阈值，否则重写刚生成就被自己的字数门否掉。
    # 旧 prompt 写"100 字左右"，而 PASS_MIN_CHARS=120 —— 20 次重写的失败样本正文
    # 是 108/108/108/111/119 字，全部差 1~12 字，纯粹是prompt 与阈值互斥造成的假失败。
    # 口径与执行侧修复轮共用 REPAIR_MIN_CHARS（见文件头注释），两条路的字数要求必须一致。
    min_chars = REPAIR_MIN_CHARS

    if context:
        print("  -> [补检索] 成功，基于新证据重写")
        prompt = prompts.FREE_REFLECT_SUCCESS_PROMPT.format(
            topic=topic, context=context, min_chars=min_chars)
        used_fallback = False
    else:
        print("  -> [补检索] 仍失败，走模型知识兜底并声明")
        prompt = prompts.FREE_REFLECT_FAIL_PROMPT.format(topic=topic, min_chars=min_chars)
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
        # v2.3：补检索/重写这一轮也入轨迹，三臂的"路径序列"才完整
        # free 轮不传 facts，第三步若需按轮比较证据条数，需在这里补 facts=_decision_facts(state)
        run_stats["repair"] = _repair_trace(
            run_stats.get("repair"), verdict, "free", gate=gate,
            reason=final_reason, decision_source=gated_by,
            repeat_hit=bool(prev_refl.get("repeat_hit", False)))
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
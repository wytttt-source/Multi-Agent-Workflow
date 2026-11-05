# -*- coding: utf-8 -*-
"""反思判定门控的离线单测：不调用真实 LLM、不联网、不写 data/runs/。

对应用例编号见 docs/反思判定状态机-改动清单.md 第 7 节（C1~C7）。
运行方式（任选其一）：
    python -m unittest discover -s tests -v
    python tests/test_reflection_gate.py
"""
import unittest
from unittest import mock

import agents.nodes as nodes
from tools import _clean_for_coverage, evidence_coverage
from utils import content_gate

EVIDENCE = [{"title": "细胞结构", "url": "https://a.com/1",
             "summary": "细胞膜与细胞核", "content": "正文节选"}]
SOURCES = ["细胞结构 | https://a.com/1"]
# 超过 PASS_MIN_CHARS=120 的长文本
LONG = ("量子计算利用叠加态的并行性，在特定问题上相对经典计算有指数级加速，"
        "但并非全面取代；经典计算在处理确定性任务时仍然更稳定、成本更低。") * 3
WITH_SOURCE = LONG + "\n【来源】细胞结构 https://a.com/1"


def _state(result=LONG, sources=None, evidence=None, used_fallback=False, extra=None):
    state = {
        "topic": "生物细胞的结构",
        "result": result,
        "sources": SOURCES if sources is None else sources,
        "evidence": EVIDENCE if evidence is None else evidence,
        "used_fallback": used_fallback,
        "reflection_history": [],
        "replan_count": 0,
        "llm_fail_count": 0,
        "subtasks": [],
        "run_stats": {"search": {"cache": "network"},
                      "reflection": {"coverage": {"missing": ["Java并发编程"]}}},
    }
    state.update(extra or {})
    return state


class ContentGateTest(unittest.TestCase):
    """门控只报事实，判定由 _apply_pass_gate 做。"""

    def test_c1_no_source_line_is_not_pass(self):
        gate = content_gate(LONG, SOURCES, EVIDENCE)
        self.assertTrue(gate["len_ok"])
        self.assertTrue(gate["evidence_nonempty"])
        self.assertFalse(gate["has_source_line"])
        self.assertEqual(gate["cited_count"], 0)

        verdict, reason, hit = nodes._apply_pass_gate(_state(), "通过", "LLM 判通过", gate)
        self.assertEqual(verdict, "降级通过")
        self.assertEqual(hit, "无引用支撑")
        self.assertIn("可核验引用 0 条", reason)

    def test_c2_source_line_with_verified_url_passes(self):
        gate = content_gate(WITH_SOURCE, SOURCES, EVIDENCE)
        self.assertTrue(gate["has_source_line"])
        self.assertEqual(gate["cited_count"], 1)
        verdict, _, hit = nodes._apply_pass_gate(_state(WITH_SOURCE), "通过",
                                                 "LLM 判通过", gate)
        self.assertEqual((verdict, hit), ("通过", ""))

    def test_c3_fabricated_url_rejected(self):
        text = LONG + "\n【来源】随便编的 https://fake.example.com/x"
        gate = content_gate(text, SOURCES, EVIDENCE)
        self.assertFalse(gate["no_fabricated_url"])
        verdict, _, hit = nodes._apply_pass_gate(_state(text), "通过", "LLM 判通过", gate)
        self.assertEqual(verdict, "内容不合格")
        self.assertEqual(hit, "疑似编造 URL")

    def test_c3b_truncated_query_url_is_verified(self):
        # 实跑案例（生物细胞的结构）：模型把带 query 的 URL 截断后引用，
        # 精确匹配会误判"编造"，从而白白触发一次局部重规划。
        sources = ["全屏显示 | https://mooc1.chaoxing.com/mooc-ans/"
                               "visitnodedetail?courseId=200901994&knowledgeId=116677997"]
        text = LONG + "\n【来源】全屏显示 https://mooc1.chaoxing.com/mooc-ans/visitnodedetail"
        gate = content_gate(text, sources, EVIDENCE)
        self.assertTrue(gate["no_fabricated_url"])
        self.assertEqual(gate["cited_count"], 1)
        self.assertEqual(len(gate["cited_loose"]), 1)
        verdict, _, hit = nodes._apply_pass_gate(_state(text), "通过", "LLM 判通过", gate)
        self.assertEqual((verdict, hit), ("通过", ""))

    def test_domain_root_only_is_not_verified(self):
        # 只写域名根不算引用某条来源
        text = LONG + "\n【来源】首页 https://a.com"
        gate = content_gate(text, SOURCES, EVIDENCE)
        self.assertEqual(gate["cited_count"], 0)
        self.assertFalse(gate["no_fabricated_url"])

    def test_url_normalization_anchor_and_trailing_punct(self):
        # 锚点与中文句号不应被当成"编造来源"（旧实现会误判）
        text = WITH_SOURCE.replace("https://a.com/1", "https://a.com/1#part2。")
        gate = content_gate(text, SOURCES, EVIDENCE)
        self.assertTrue(gate["no_fabricated_url"])
        self.assertEqual(gate["cited_count"], 1)


class ReflectionPathTest(unittest.TestCase):
    """三条判定路径都必须过同一套门控。"""

    def test_c4_rule_fallback_without_citation_is_degraded(self):
        out = nodes.rule_based_reflection(_state(), "反思节点LLM不可用: Request timed out.")
        self.assertEqual(out["error_type"], "降级通过")
        refl = out["run_stats"]["reflection"]
        self.assertEqual(refl["gated_by"], "gate")
        self.assertEqual(refl["pass_kind"], "degraded")
        self.assertEqual(refl["cited_count"], 0)

    def test_rule_fallback_with_citation_passes(self):
        out = nodes.rule_based_reflection(_state(WITH_SOURCE), "反思节点LLM不可用")
        self.assertEqual(out["error_type"], "通过")
        self.assertEqual(out["run_stats"]["reflection"]["gated_by"], "rule_fallback")
        self.assertEqual(out["run_stats"]["reflection"]["pass_kind"], "strict")

    def test_c5_free_reflection_success_without_citation_is_degraded(self):
        out, _ = self._run_free_reflection(LONG)
        self.assertEqual(out["error_type"], "降级通过")
        self.assertEqual(out["run_stats"]["reflection"]["pass_kind"], "degraded")
        self.assertEqual(out["run_stats"]["reflection"]["gated_by"], "free_reflection")

    def test_c5b_free_reflection_success_with_citation_passes(self):
        out, _ = self._run_free_reflection(WITH_SOURCE)
        self.assertEqual(out["error_type"], "通过")

    def test_c6_retry_query_differs_from_first_round(self):
        out, call = self._run_free_reflection(WITH_SOURCE)
        retry_query = call.call_args.kwargs.get("query")
        self.assertTrue(retry_query)
        self.assertNotEqual(retry_query, "生物细胞的结构 lang:zh")
        self.assertIn("Java并发编程", retry_query)   # 首轮未覆盖对象被追加
        self.assertEqual(out["run_stats"]["search"]["retry_query"], retry_query)
        self.assertTrue(out["run_stats"]["search"]["retry_mode"].startswith("missing_term"))

    def test_retry_mode_same_keeps_original_query(self):
        with mock.patch.object(nodes, "RETRY_QUERY_MODE", "same"):
            query, mode = nodes._build_retry_query(_state())
        self.assertEqual(mode, "same")
        self.assertEqual(query, "生物细胞的结构 lang:zh")

    def _run_free_reflection(self, content):
        state = _state(result="上一轮结果", used_fallback=True)
        with mock.patch.object(nodes, "search_with_evidence",
                               return_value=("证据块", SOURCES, EVIDENCE,
                                             {"cache": "network"})) as patched, \
                mock.patch.object(nodes, "safe_llm_invoke", return_value=content):
            out = nodes.free_reflection_node(state)
        return out, patched


class CoverageTermTest(unittest.TestCase):
    """C7：主题切分回归（防止覆盖率规则静默退化成误判）。"""

    def test_clean_for_coverage(self):
        cases = {
            "比较Python和Java在并发编程与内存管理上的差异": "Python和Java",
            "比较Rust和Go在内存安全与并发模型上的差异": "Rust和Go",
        }
        for topic, expected in cases.items():
            self.assertEqual(_clean_for_coverage(topic), expected)

    def test_composite_terms(self):
        coverage = evidence_coverage("比较Python和Java在并发编程与内存管理上的差异",
                                     EVIDENCE)
        self.assertEqual(coverage["terms"], ["Python", "Java"])
        # 对称句：无"在…上的"状语，剥掉"的区别"后按"和"切分
        coverage = evidence_coverage("量子计算和经典计算的区别", EVIDENCE)
        self.assertEqual(coverage["terms"], ["量子计算", "经典计算"])

    def test_non_composite_topic_terms_not_empty(self):
        coverage = evidence_coverage("生物细胞的结构", EVIDENCE)
        self.assertEqual(coverage["terms"], ["生物细胞"])

    def test_p1_2_suffix_and_question_tails(self):
        # docs/覆盖率的边界用例.md 第 4 节的三条"仍会切错"的句式
        cases = {
            "Python和Java的并发性能": ["Python", "Java"],
            "Rust与Go哪个更适合写后端": ["Rust", "Go"],
            "对比Kafka与RabbitMQ在吞吐上的表现": ["Kafka", "RabbitMQ"],
        }
        for topic, expected in cases.items():
            with self.subTest(topic=topic):
                self.assertEqual(evidence_coverage(topic, EVIDENCE)["terms"], expected)

    def test_hit_is_case_and_space_insensitive(self):
        results = [{"title": "Python 与 Java 的并发模型", "summary": ""}]
        coverage = evidence_coverage("比较python和JAVA的差异", results)
        self.assertEqual(coverage["terms"], ["python", "JAVA"])
        self.assertEqual(coverage["ratio"], 1.0)

    def test_p1_3_balance_uses_summary_not_only_title(self):
        results = [
            {"title": "并发模型对比", "summary": "Java 线程池与协程的取舍"},
            {"title": "并发模型对比", "summary": "Java 与 Python 的并发差异"},
        ]
        coverage = evidence_coverage("Python和Java的并发性能", results)
        self.assertEqual(coverage["per_term_counts"], {"Python": 1, "Java": 2})
        self.assertEqual(coverage["balance"], 0.5)


class RepeatRuleTest(unittest.TestCase):
    """C8（P1-1）：任意类型的否决连发 2 次 → 证据不足，不再原地重规划。"""

    def test_c8_second_content_failure_goes_free_reflection(self):
        state = _state()
        state["reflection_history"] = ["[内容不合格] 回答过短（80 字）"]
        state["replan_count"] = 1
        decision = nodes._decide_next(state, "内容不合格", "回答过短（80 字）")
        self.assertEqual(decision["error_type"], "证据不足")
        self.assertTrue(decision["_repeat_hit"])
        self.assertEqual(decision["replan_count"], 2)
        self.assertIn("内容不合格重复2次", decision["reflection_history"][-1])

    def test_first_occurrence_still_replans_locally(self):
        decision = nodes._decide_next(_state(), "内容不合格", "偏题")
        self.assertEqual(decision["error_type"], "内容不合格")
        self.assertEqual(decision["replan_scope"], "局部")
        self.assertFalse(decision["_repeat_hit"])

    def test_e5_repeat_rule_can_be_disabled(self):
        state = _state(extra={"reflection_history": ["[内容不合格] 偏题"],
                              "replan_count": 1})
        with mock.patch.object(nodes, "ENABLE_REPEAT_RULE", False):
            decision = nodes._decide_next(state, "内容不合格", "偏题")
        self.assertEqual(decision["error_type"], "内容不合格")
        self.assertEqual(decision["replan_scope"], "局部")
        self.assertFalse(decision["_repeat_hit"])

    def test_repeat_hit_recorded_in_run_stats(self):
        out = nodes.rule_based_reflection(
            _state(result="太短", extra={"reflection_history": ["[内容不合格] 字数过少(2字)"]}),
            "格式解析错误")
        self.assertEqual(out["error_type"], "证据不足")
        self.assertTrue(out["run_stats"]["reflection"]["repeat_hit"])
        self.assertEqual(out["run_stats"]["reflection"]["decision_source"], "rule_fallback")


class ExperimentSwitchTest(unittest.TestCase):
    """第 5 节的 4 个开关：改判定分布，因此必须显式可关、可复现。"""

    def test_e1_disabled_reflection_passes_directly(self):
        with mock.patch.object(nodes, "ENABLE_REFLECTION", False):
            out = nodes.reflection_node(_state(result="太短"))   # 连字数都不看
        self.assertEqual(out["error_type"], "通过")
        refl = out["run_stats"]["reflection"]
        self.assertEqual(refl["decision_source"], "baseline")
        self.assertEqual(refl["pass_kind"], "strict")

    def test_e2_disabled_gate_skips_deterministic_veto(self):
        with mock.patch.object(nodes, "DISABLE_CONTENT_GATE", True):
            out = nodes.rule_based_reflection(_state(result=LONG), "反思LLM不可用")
        self.assertEqual(out["error_type"], "通过")

    def test_e4_coverage_rule_switch(self):
        state = _state(result=WITH_SOURCE)
        state["topic"] = "比较Python和Java的并发性能"   # 证据里只有"细胞结构"→ 覆盖 0
        llm_json = '{"error_type": "通过", "reason": "覆盖不足但语义可读"}'
        with mock.patch.object(nodes, "safe_llm_invoke", return_value=llm_json):
            on = nodes.reflection_node(dict(state))
        self.assertEqual(on["error_type"], "分解不合理")
        self.assertEqual(on["run_stats"]["reflection"]["decision_source"], "coverage_rule")

        with mock.patch.object(nodes, "ENABLE_COVERAGE_RULE", False), \
                mock.patch.object(nodes, "safe_llm_invoke", return_value=llm_json):
            off = nodes.reflection_node(dict(state))
        self.assertEqual(off["error_type"], "通过")

    def test_e6_inject_llm_fail_then_recovers(self):
        import config
        import utils
        utils.reset_llm_calls()
        with mock.patch.object(config, "INJECT_LLM_FAIL", 2):
            for _ in range(2):
                with self.assertRaises(utils.LLMInfraError):
                    utils.safe_llm_invoke("prompt")
            with mock.patch.object(config, "llm_generator",
                                   mock.Mock(invoke=mock.Mock(
                                       return_value=mock.Mock(content="ok")))):
                self.assertEqual(utils.safe_llm_invoke("prompt"), "ok")
        self.assertEqual(utils.llm_call_count(), 3)
        utils.reset_llm_calls()



if __name__ == "__main__":
    unittest.main(verbosity=2)


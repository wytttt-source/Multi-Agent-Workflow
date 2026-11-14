# -*- coding: utf-8 -*-
"""v2.3 修复路径三臂（fixed / free / hybrid）的离线单测。

覆盖四组：
    RepairPolicyTest       —— 策略只改"走哪条路"，不改判定（R1~R4）
    InjectFaultTest        —— 首轮受控缺陷只作用于第一轮，且形态可被门控识别（F1~F4）
    RepairTraceTest        —— 按轮轨迹与修复结果四态（T1~T2）
    AnswerCompletenessTest —— 补检索重写后的完整性复核（K1~K3）

不调用真实 LLM、不联网、不写 data/。
运行方式：
    python -m unittest discover -s tests -v
    python tests/test_repair_policy.py
"""
import unittest
from unittest import mock

import agents.nodes as nodes
from utils import answer_completeness, _looks_truncated

EVIDENCE = [{"title": "细胞结构", "url": "https://a.com/1",
             "summary": "细胞膜与细胞核", "content": "正文节选"}]
SOURCES = ["细胞结构 | https://a.com/1"]
# 超过 PASS_MIN_CHARS=120 的长文本
LONG = ("量子计算利用叠加态的并行性，在特定问题上相对经典计算有指数级加速，"
        "但并非全面取代；经典计算在处理确定性任务时仍然更稳定、成本更低。") * 3
TWO_OBJECTS = ("Python 的并发以 GIL 与 asyncio 为主，Java 则依赖线程池与 JUC 包；"
               "两者在内存管理上分别走引用计数加 GC 与分代 GC。") * 3


def _state(**extra):
    state = {
        "topic": "生物细胞的结构",
        "result": LONG,
        "sources": SOURCES,
        "evidence": EVIDENCE,
        "used_fallback": False,
        "reflection_history": [],
        "replan_count": 0,
        "llm_fail_count": 0,
        "subtasks": [],
        "run_stats": {},
    }
    state.update(extra)
    return state


class RepairPolicyTest(unittest.TestCase):
    """策略是"允许走哪条路"的开关，不参与"错在哪"的判断。"""

    def test_r1_fixed_never_falls_back_to_free(self):
        # 无证据 + 内容不合格：hybrid 会转自由反思，fixed 只能局部重试
        with mock.patch.object(nodes, "REPAIR_POLICY", "fixed"):
            decision = nodes._decide_next(_state(evidence=[]), "内容不合格", "证据缺失")
        self.assertEqual(decision["repair_path"], "local")
        self.assertEqual(decision["replan_scope"], "局部")
        # error_type 不改成「证据不足」——否则路由器会把 fixed 臂送去自由反思
        self.assertEqual(decision["error_type"], "内容不合格")

    def test_r2_fixed_keeps_global_for_decomposition(self):
        with mock.patch.object(nodes, "REPAIR_POLICY", "fixed"):
            decision = nodes._decide_next(_state(), "分解不合理", "只讲了一个对象")
        self.assertEqual(decision["repair_path"], "global")
        self.assertEqual(decision["replan_scope"], "全局")

    def test_r3_free_policy_always_reflects(self):
        with mock.patch.object(nodes, "REPAIR_POLICY", "free"):
            decision = nodes._decide_next(_state(), "内容不合格", "回答过短")
        self.assertEqual(decision["repair_path"], "free")
        self.assertEqual(decision["error_type"], "证据不足")
        self.assertEqual(decision["replan_scope"], "")

    def test_r4_hybrid_matches_v22_behavior(self):
        with mock.patch.object(nodes, "REPAIR_POLICY", "hybrid"):
            local = nodes._decide_next(_state(), "内容不合格", "回答过短")
            free = nodes._decide_next(_state(evidence=[]), "内容不合格", "证据缺失")
        self.assertEqual((local["repair_path"], local["replan_scope"]), ("local", "局部"))
        self.assertEqual((free["repair_path"], free["error_type"]), ("free", "证据不足"))


class InjectFaultTest(unittest.TestCase):
    """注入只为"制造需要修复的样本"，绝不能把修复轮次一起打断。"""

    def test_f1_fault_only_on_first_round(self):
        with mock.patch.object(nodes, "INJECT_FAULT", "short"):
            first, marks = nodes._inject_fault_content(LONG, _state())
            second, marks2 = nodes._inject_fault_content(LONG, _state(replan_count=1))
        self.assertEqual(marks, ["short"])
        self.assertLess(len(first), len(LONG))
        self.assertEqual((second, marks2), (LONG, []))

    def test_f2_no_cite_strips_source_block(self):
        text = LONG + "\n【来源】细胞结构 https://a.com/1"
        with mock.patch.object(nodes, "INJECT_FAULT", "no_cite"):
            injected, marks = nodes._inject_fault_content(text, _state())
        self.assertNotIn("【来源】", injected)
        self.assertEqual(marks, ["no_cite"])

    def test_f3_truncate_hits_deterministic_rule(self):
        with mock.patch.object(nodes, "INJECT_FAULT", "truncate"):
            injected, _ = nodes._inject_fault_content(LONG * 2, _state())
        self.assertTrue(_looks_truncated(injected)["truncated"])

    def test_f4_unknown_fault_is_noop(self):
        with mock.patch.object(nodes, "INJECT_FAULT", "none"):
            text, marks = nodes._inject_fault_content(LONG, _state())
            self.assertFalse(nodes._active_fault(_state()))
        self.assertEqual((text, marks), (LONG, []))


class RepairTraceTest(unittest.TestCase):
    """轨迹要能回答：首轮错在哪、走了哪条路、最后修好没有。"""

    def test_t1_outcome_four_states(self):
        cases = {
            ("通过",): "pass_first_try",
            ("内容不合格", "通过"): "fixed",
            ("内容不合格", "降级通过"): "partial",
            ("降级通过", "降级通过"): "degraded_unchanged",
            ("内容不合格", "内容不合格"): "failed",
        }
        for verdicts, expected in cases.items():
            rounds = [{"verdict": v} for v in verdicts]
            self.assertEqual(nodes._repair_outcome(rounds), expected, verdicts)

    def test_t2_rounds_accumulate_and_outcome_updates(self):
        state = _state()
        out1 = nodes._reflection_stats(state, None, "内容不合格", "回答过短",
                                       "gate", path="local")
        state2 = {**state, "run_stats": out1["run_stats"]}
        out2 = nodes._reflection_stats(state2, None, "通过", "修好了", "llm", path="")
        repair = out2["run_stats"]["repair"]
        self.assertEqual(len(repair["rounds"]), 2)
        self.assertEqual(repair["rounds"][0]["path"], "local")
        self.assertEqual(repair["first_verdict"], "内容不合格")
        self.assertEqual(repair["final_verdict"], "通过")
        self.assertEqual(repair["outcome"], "fixed")
        self.assertEqual(repair["policy"], nodes.REPAIR_POLICY)


class AnswerCompletenessTest(unittest.TestCase):
    """补检索重写后最容易"补上引用、只讲一个对象"，字数门看不出来。"""

    def test_k1_all_objects_covered(self):
        info = answer_completeness(TWO_OBJECTS, ["Python", "Java"])
        self.assertTrue(info["complete"])
        self.assertEqual(info["ratio"], 1.0)
        self.assertEqual(info["missing"], [])

    def test_k2_missing_object_is_incomplete(self):
        info = answer_completeness(TWO_OBJECTS, ["Python", "Java", "Rust"])
        self.assertFalse(info["complete"])
        self.assertEqual(info["missing"], ["Rust"])
        self.assertAlmostEqual(info["ratio"], 0.667, places=3)

    def test_k3_truncated_answer_is_incomplete(self):
        info = answer_completeness(LONG + "以及", ["量子计算"])
        self.assertTrue(info["truncated"])
        self.assertFalse(info["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

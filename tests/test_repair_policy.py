# -*- coding: utf-8 -*-
"""v2.3 修复路径三臂（fixed / free / hybrid）的离线单测。

覆盖七组：
    RepairPolicyTest       —— 策略只改"走哪条路"，不改判定（R1~R4）
    RepairRuleTest         —— 规则编号只做命名，不改变任何路径选择（N1~N4）
    InjectFaultTest        —— 首轮受控缺陷只作用于第一轮，且形态可被门控识别（F1~F7）
    RepairTraceTest        —— 按轮轨迹与修复结果四态（T1~T2）
    AnswerCompletenessTest —— 补检索重写后的完整性复核（K1~K3）
    CoverageVerdictTest    —— 覆盖率终判看答案侧，证据侧只作提示（K4~K7）
    GatePrecedenceTest     —— 截断/过短等硬事实优先于「分解不合理」（G1~G2）
    CoverageHintTest       —— 裁判提示词的覆盖率口径与判定口径同步（M1~M2）
    FreeReflectPromptTest  —— 重写长度口径高于门控阈值（L1）

不调用真实 LLM、不联网、不写 data/。
运行方式：
    python -m unittest discover -s tests -v
    python tests/test_repair_policy.py
"""
import unittest
from unittest import mock

import agents.nodes as nodes
from utils import answer_completeness, _looks_truncated, content_gate

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

    def test_r4_hybrid_rewrites_when_evidence_exists(self):
        # v2.5：证据已在手 → 定向重写（rewrite）；证据为空仍走自由反思
        with mock.patch.object(nodes, "REPAIR_POLICY", "hybrid"):
            rewrite = nodes._decide_next(_state(), "内容不合格", "回答过短")
            free = nodes._decide_next(_state(evidence=[]), "内容不合格", "证据缺失")
        self.assertEqual((rewrite["repair_path"], rewrite["replan_scope"]),
                         ("rewrite", "局部"))
        self.assertEqual((free["repair_path"], free["error_type"]), ("free", "证据不足"))


class RepairRuleTest(unittest.TestCase):
    """v2.5 第一步：规则编号（R1~R5）只做命名，不得改变任何路径选择（N1~N4）。

    验收依据：experiments/replay_rules.py 用 v2.4 的 48 份记录离线复算，
    46 个决策轮零翻转；下面这四条锁住"改名"这件事本身。
    """

    def _decide(self, error_type, reason="回答过短", policy="hybrid", **state_kw):
        with mock.patch.object(nodes, "REPAIR_POLICY", policy):
            return nodes._decide_next(_state(**state_kw), error_type, reason)

    def test_n1_rule_table_reproduces_v24_paths(self):
        # R1/R3/R4 与 v2.4 逐条一致；R2 是第二步唯一改动的分支（local → rewrite，全臂共用）
        cases = [
            (("内容不合格", "回答过短"), {}, "hybrid", "rewrite", "R2_answer_gap"),
            (("内容不合格", "证据 0 条"), {"evidence": []}, "hybrid", "free",
             "R1_evidence_gap"),
            (("分解不合理", "只讲了一个对象"), {}, "hybrid", "global", "R3_structure"),
            (("内容不合格", "回答过短"), {}, "fixed", "rewrite", "R2_answer_gap"),
            (("分解不合理", "只讲了一个对象"), {}, "fixed", "global", "R3_structure"),
            (("内容不合格", "回答过短"), {}, "free", "free", "R2_answer_gap"),
        ]
        for (etype, reason), kw, policy, want_path, want_rule in cases:
            with self.subTest(etype=etype, policy=policy, reason=reason):
                decision = self._decide(etype, reason, policy, **kw)
                self.assertEqual(decision["repair_path"], want_path)
                self.assertEqual(decision["_rule"], want_rule)

    def test_n2_repeat_overrides_to_free(self):
        # 同类错误第 2 次出现：说明上一轮定向修复没带来新信息 → 换检索式（R4）
        state = _state(reflection_history=["[内容不合格] 上一轮回答过短"])
        with mock.patch.object(nodes, "REPAIR_POLICY", "hybrid"), \
                mock.patch.object(nodes, "ENABLE_REPEAT_RULE", True):
            decision = nodes._decide_next(state, "内容不合格", "回答过短")
        self.assertEqual((decision["repair_path"], decision["_rule"]),
                         ("free", "R4_repeat"))

    def test_n3_facts_are_llm_free_and_split_gap(self):
        # 选路依据必须确定性可复算：_decision_facts 不许调 LLM
        with mock.patch.object(nodes, "REQUIRED_OBJECTS", ["细胞膜", "细胞核"]), \
                mock.patch.object(nodes, "safe_llm_invoke",
                                  side_effect=AssertionError("选路不得调 LLM")):
            facts = nodes._decision_facts(_state(result="细胞膜控制物质进出。" * 12))
        self.assertEqual(facts["missing"], ["细胞核"])
        # 细胞核 在证据摘要里出现过 → 属于"生成漏了"，不是检索缺口（否则会白换检索式）
        self.assertEqual(facts["evidence_gap"], [])
        # 缺的对象证据里都有 → 该修答案（R2），不该换检索式（R1）
        self.assertTrue(facts["evidence_covers_required"])

    def test_n4_rule_reaches_repair_trace(self):
        stats = nodes._reflection_stats(
            _state(), None, "内容不合格", "回答过短", "llm",
            gate=content_gate(LONG, SOURCES, EVIDENCE),
            path="local", rule="R2_answer_gap")
        last = stats["run_stats"]["repair"]["rounds"][-1]
        self.assertEqual(last["rule"], "R2_answer_gap")
        # 事实与规则同源落盘：缺哪些对象、其中哪些证据里也没有、本轮几条证据
        self.assertIn("answer_missing", last)
        self.assertIn("evidence_gap", last)
        self.assertEqual(last["evidence_count"], len(EVIDENCE))


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

    def test_f5_truncate_cuts_body_not_source_block(self):
        # 旧实现按"全文"算截断点：来源清单越长切点越靠后，甚至落进 URL 里，
        # 而 content_gate 只认【来源】之前的正文 → 截断信号丢失（P1）。
        body = LONG * 2
        text = body + "\n【来源】" + "\n".join(
            f"细胞结构 | https://a.com/{i}" for i in range(1, 21))
        with mock.patch.object(nodes, "INJECT_FAULT", "truncate"):
            injected, marks = nodes._inject_fault_content(text, _state())
        body_out, source_block = nodes._split_body(injected)
        self.assertEqual(marks, ["truncate"])
        self.assertLess(len(body_out), len(body))          # 切在正文里
        self.assertIn("【来源】", source_block)             # 来源段保留（缺陷只落在内容）
        gate = content_gate(injected, SOURCES, EVIDENCE)
        self.assertTrue(gate["truncated"])
        self.assertEqual(gate["cited_count"], 1)

    def test_f6_one_side_degrades_answer_side(self):
        # 首轮只检索第一个对象，模型靠自身知识把另一侧补上 → 缺陷到不了答案层（P3）。
        # 修复后答案里"只谈另一侧"的句子也要删掉。
        mixed = ("Python 的并发以 GIL 与 asyncio 为主，协程切换开销小。" * 2
                 + "Java 则依赖线程池与 JUC 包，线程模型更重。"
                 + "两者在内存管理上分别走引用计数加 GC 与分代 GC。" * 2
                 + "Python 的性能瓶颈常出现在 CPU 密集任务。" * 2)
        text = mixed + "\n\n【来源】细胞结构 | https://a.com/1"
        topic = "比较Python和Java的并发性能"
        with mock.patch.object(nodes, "REQUIRED_OBJECTS", ["Python", "Java"]), \
                mock.patch.object(nodes, "INJECT_FAULT", "one_side"):
            injected, marks = nodes._inject_fault_content(text, _state(topic=topic))
        self.assertIn("Python", injected)
        self.assertNotIn("Java", injected)
        self.assertEqual(marks, ["one_side:answer_drop:Java"])
        self.assertEqual(answer_completeness(injected, ["Python", "Java"])["missing"],
                         ["Java"])

    def test_f7_one_side_is_honest_when_undroppable(self):
        # 删不动（剩余句子不足 3 句）时如实报告"未生效"，不能假装注入成功
        text = "Python 与 Java 都是主流语言。"
        with mock.patch.object(nodes, "REQUIRED_OBJECTS", ["Python", "Java"]), \
                mock.patch.object(nodes, "INJECT_FAULT", "one_side"):
            injected, marks = nodes._inject_fault_content(
                text, _state(topic="比较Python和Java的并发性能"))
        self.assertEqual((injected, marks), (text, []))


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

    def test_k9_substantive_mode_ignores_material_disclaimers(self):
        # 判卷口径（substantive=True）：句子只是声明"资料未提供 X"，没有讲 X → 不算覆盖
        answer = ("围绕Kafka与RabbitMQ的吞吐对比，现有资料仅覆盖RabbitMQ。"
                  "RabbitMQ 在仲裁队列上有改进，吞吐受队列类型影响。"
                  "但资料未提供Kafka的吞吐或基准测试数据，需补充证据后再作比较。")
        lenient = answer_completeness(answer, ["Kafka", "RabbitMQ"])
        strict = answer_completeness(answer, ["Kafka", "RabbitMQ"], substantive=True)
        self.assertEqual(lenient["missing"], [])          # 旧口径：提到名字即算讲到
        self.assertEqual(strict["missing"], ["Kafka"])    # 判卷口径：只有声明不算
        self.assertTrue(strict["substantive"])

    def test_k11_substantive_mode_ignores_refusal_and_plan(self):
        # 实跑案例（P-F 的 B4 比较 Docker/Podman/containerd）：重试检索失败后，
        # 模型给的是一份"拒答 + 下一步检索计划"，句子里三个对象名全都出现，
        # 但只有 Docker/containerd 讲了内容，Podman 只出现在"没查到/该补什么"里。
        # 旧口径会把这份拒答判成三对象全覆盖 → 系统判否决反被记成"误否决"。
        answer = (
            "在严格遵循“只依据所给参考资料”的前提下，我不能把现有材料改写成 "
            "Docker、Podman 与 containerd 三者镜像管理的比较摘要。"
            "当前可核验资料只有 Docker Desktop 关于 containerd 镜像存储的页面；"
            "它能说明 Docker Engine 已在底层使用 containerd 创建、启动和停止容器。"
            "该资料没有提供 Podman 镜像拉取、镜像存储、rootless 或多架构的可核验材料。"
            "修复该任务需补充三类官方资料：Docker 镜像存储文档，containerd 的 images 文档，"
            "Podman 的 pull/build/image 与 storage.conf 文档。"
            "可按“site:docs.podman.io podman image storage.conf”等检索式补齐。"
        )
        info = answer_completeness(answer, ["Docker", "Podman", "containerd"], substantive=True)
        self.assertEqual(info["missing"], ["Podman"])
        # 反向保护：三个对象名都在"实际讲了内容"的句子里时不许误删
        ok = answer_completeness(
            "Docker 用 containerd 管理镜像存储，Podman 无守护进程、用 storage.conf 配置存储。",
            ["Docker", "Podman", "containerd"], substantive=True)
        self.assertEqual(ok["missing"], [])

    def test_k10_substantive_mode_keeps_negative_findings(self):
        # 反向保护：'X 未提供某特性' 是**实质结论**，剔除后不能把对象判成没讲
        answer = ("Redis 未提供事务回滚，只能靠 MULTI/EXEC 做有限组合；"
                  "PostgreSQL 支持保存点回滚。") * 3
        info = answer_completeness(answer, ["Redis", "PostgreSQL"], substantive=True)
        self.assertEqual(info["missing"], [])


CELL_ANSWER = ("细胞膜是选择透过性屏障，控制物质进出并参与信号识别。"
               "细胞核储存遗传信息，通过转录调控蛋白合成。") * 4
# 子任务串在证据里几乎不可能逐字出现：旧口径据此判"分解不合理"（P2 的规则缺陷）
NEVER_LITERAL_SUBTASKS = ["细胞膜的选择透过性与信号识别机制",
                          "细胞核的转录调控与染色体结构"]


class CoverageVerdictTest(unittest.TestCase):
    """覆盖率终判要看**答案**有没有讲到对象，证据侧覆盖只作提示（P2）。"""

    def _run(self, required: list, subtasks: list = None):
        state = _state(topic="生物细胞的结构",
                       result=CELL_ANSWER + "\n【来源】细胞结构 https://a.com/1",
                       subtasks=subtasks if subtasks is not None else NEVER_LITERAL_SUBTASKS,
                       evidence=EVIDENCE + [{"title": "细胞核", "url": "https://a.com/2",
                                             "summary": "细胞核储存遗传信息",
                                             "content": "正文节选"}])
        with mock.patch.object(nodes, "REQUIRED_OBJECTS", required), \
                mock.patch.object(nodes, "safe_llm_invoke",
                                  return_value='{"error_type": "通过", "reason": "内容完整"}'):
            return nodes.reflection_node(state)

    def test_k4_answer_coverage_match_not_judged_as_bad_decomposition(self):
        out = self._run(["细胞膜", "细胞核"])
        self.assertEqual(out["error_type"], "通过")
        self.assertEqual(out["run_stats"]["reflection"]["coverage"]["scope"], "answer")

    def test_k5_legacy_rule_still_fires_without_required_objects(self):
        # 留空 REQUIRED_OBJECTS = 沿用旧口径：子任务串匹配不上直接判分解不合理
        out = self._run([])
        self.assertEqual(out["error_type"], "分解不合理")
        self.assertEqual(out["run_stats"]["reflection"]["decision_source"], "coverage_rule")

    def test_k6_missing_object_absent_from_evidence_routes_to_retrieval(self):
        # 答案缺 Rust，且证据里也没有 Rust → 检索缺口：局部重写补不出内容，应补检索
        state = _state(topic="比较Python和Java的并发性能",
                       result=TWO_OBJECTS + "\n【来源】细胞结构 https://a.com/1")
        with mock.patch.object(nodes, "REQUIRED_OBJECTS", ["Python", "Java", "Rust"]), \
                mock.patch.object(nodes, "REPAIR_POLICY", "hybrid"), \
                mock.patch.object(nodes, "safe_llm_invoke",
                                  return_value='{"error_type": "通过", "reason": "完整"}'):
            out = nodes.reflection_node(state)
        self.assertEqual(out["error_type"], "证据不足")
        self.assertIn("Rust", out["run_stats"]["reflection"]["reason"])
        # 混合策略：证据不足 → 自由反思（换检索式补证据），不该先白重写一轮
        self.assertEqual(out["run_stats"]["repair"]["rounds"][0]["path"], "free")

    def test_k8_missing_object_present_in_evidence_goes_to_rewrite(self):
        # 证据里有 Rust，只是答案没写 → 生成缺口：同一份证据重写即可（rewrite），不必重检索
        ev = EVIDENCE + [{"title": "Rust 并发", "url": "https://a.com/2",
                          "summary": "Rust 用所有权避免数据竞争", "content": "正文节选"}]
        state = _state(topic="比较Python和Java的并发性能", evidence=ev,
                       result=TWO_OBJECTS + "\n【来源】细胞结构 https://a.com/1")
        with mock.patch.object(nodes, "REQUIRED_OBJECTS", ["Python", "Java", "Rust"]), \
                mock.patch.object(nodes, "REPAIR_POLICY", "hybrid"), \
                mock.patch.object(nodes, "safe_llm_invoke",
                                  return_value='{"error_type": "通过", "reason": "完整"}'):
            out = nodes.reflection_node(state)
        self.assertEqual(out["error_type"], "内容不合格")
        self.assertIn("Rust", out["run_stats"]["reflection"]["reason"])
        self.assertEqual(out["run_stats"]["repair"]["rounds"][0]["path"], "rewrite")

    def test_k7_gate_facts_recorded_on_coverage_round(self):
        # 覆盖率轮不调用裁判 LLM，但门控事实仍要落盘：
        # 否则对照表里这一轮是"正文 null / 引用 null"，看起来像没测量
        state = _state(topic="比较Python和Java的并发性能",
                       result=TWO_OBJECTS + "\n【来源】细胞结构 https://a.com/1")
        with mock.patch.object(nodes, "REQUIRED_OBJECTS", ["Python", "Java", "Rust"]), \
                mock.patch.object(nodes, "safe_llm_invoke",
                                  return_value='{"error_type": "通过", "reason": "完整"}'):
            out = nodes.reflection_node(state)
        round0 = out["run_stats"]["repair"]["rounds"][0]
        self.assertTrue(round0["gate_measured"])
        self.assertGreater(round0["body_chars"], 0)
        self.assertEqual(round0["decision_source"], "coverage_rule")


class GatePrecedenceTest(unittest.TestCase):
    """答案层硬事实优先于语义判「分解不合理」（P2 续）。

    旧 48 次里 C 组（截断/过短）同时命中"证据覆盖率 < 100%"与"截断"，
    裁判 LLM 只看前者，判了「分解不合理」→ 全局重拆 → 改完还是截断，白跑一轮。
    截断/过短/编造 URL 都能确定性量出来，应当先说话。
    """

    def _run(self, result: str):
        state = _state(topic="量子计算的优势", result=result)
        with mock.patch.object(nodes, "REQUIRED_OBJECTS", []), \
                mock.patch.object(nodes, "REPAIR_POLICY", "hybrid"), \
                mock.patch.object(nodes, "safe_llm_invoke",
                                  return_value='{"error_type": "分解不合理", '
                                               '"reason": "证据覆盖不全"}'):
            return nodes.reflection_node(state)

    def test_g1_truncation_beats_bad_decomposition(self):
        out = self._run(LONG + "以及")
        self.assertEqual(out["error_type"], "内容不合格")
        self.assertEqual(out["run_stats"]["reflection"]["decision_source"], "gate")
        # 截断是答案层问题 → 用已有证据重写，不该花代价重拆主题、也不该再检索一次
        self.assertEqual(out["run_stats"]["repair"]["rounds"][0]["path"], "rewrite")

    def test_g2_pure_decomposition_verdict_is_kept(self):
        # 没有硬事实时不得抢裁判的话：分解不合理仍然路由到全局重拆
        out = self._run(LONG + "\n【来源】细胞结构 https://a.com/1")
        self.assertEqual(out["error_type"], "分解不合理")
        self.assertEqual(out["run_stats"]["reflection"]["decision_source"], "llm")
        self.assertEqual(out["run_stats"]["repair"]["rounds"][0]["path"], "global")


class CoverageHintTest(unittest.TestCase):
    """裁判提示词里的覆盖率口径必须与判定口径同步。

    旧提示词写着"覆盖率 < 100% 就判分解不合理"，而 v2.4 的判定改看答案侧，
    两边不一致会自相矛盾：答案已经补齐两侧，系统却因为证据覆盖 0% 送去全局重拆。
    """

    def _prompt(self, required: list) -> str:
        captured = {}

        def _fake(prompt, **kwargs):
            captured["prompt"] = prompt
            return '{"error_type": "通过", "reason": "内容完整"}'

        # 回答里两个对象都在 → 不会在答案侧覆盖率那步提前返回，才会走到裁判 LLM
        state = _state(result=CELL_ANSWER + "\n【来源】细胞结构 https://a.com/1")
        with mock.patch.object(nodes, "REQUIRED_OBJECTS", required), \
                mock.patch.object(nodes, "safe_llm_invoke", side_effect=_fake):
            nodes.reflection_node(state)
        return captured["prompt"]

    def test_m1_answer_side_hint_when_required_objects(self):
        self.assertIn("不等于分解不合理", self._prompt(["细胞膜", "细胞核"]))

    def test_m2_legacy_hint_without_required_objects(self):
        self.assertIn('应判"分解不合理"', self._prompt([]))


class FreeReflectPromptTest(unittest.TestCase):
    """重写长度口径 ≫ 门控阈值（P4：prompt 写 100 字、门控要 120 字）。"""

    def test_l1_prompt_demands_above_gate_threshold(self):
        import prompts
        min_chars = max(nodes.PASS_MIN_CHARS * 2, 200)
        for tpl in (prompts.FREE_REFLECT_SUCCESS_PROMPT, prompts.FREE_REFLECT_FAIL_PROMPT):
            self.assertNotIn("100字左右", tpl)
            rendered = tpl.format(topic="细胞", context="资料", min_chars=min_chars)
            self.assertIn(str(min_chars), rendered)
        self.assertGreater(min_chars, nodes.PASS_MIN_CHARS)


if __name__ == "__main__":
    unittest.main(verbosity=2)

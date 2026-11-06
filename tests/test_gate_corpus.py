# -*- coding: utf-8 -*-
"""判决器对照实验的离线校验：不调 LLM、不联网、不写文件。

对应 docs/规则vsLLM-混淆矩阵.md 的附录 A（样本集）与第 1 节结论：
- 18 条样本必须字段齐备、id 唯一、gold 取值合法
- 规则列（把 LLM 打成不可用）与真值的二分类偏差只允许落在白名单里
  （S05 字数注水 / S14 偏题）；这两个缺口本身是实验结论的一部分，不能被悄悄改掉

运行方式：
    python -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "experiments")):
    if path not in sys.path:
        sys.path.insert(0, path)

import agents.nodes as nodes  # noqa: E402
import gate_vs_llm as gvl  # noqa: E402

GOLD_LABELS = ("通过", "降级通过", "内容不合格", "分解不合理")
# 规则列与真值不一致的样本：S05（来源行撑字数）、S14（偏题无字面信号）
RULE_GAP = {"S05", "S14"}


class GateCorpusTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(gvl.CASES_PATH, encoding="utf-8") as f:
            cls.corpus = json.load(f)

    def test_corpus_shape_and_gold(self):
        cases = self.corpus["cases"]
        self.assertEqual(len(cases), 18)
        self.assertEqual(len({c["id"] for c in cases}), 18)
        for c in cases:
            self.assertIn(c["gold"], GOLD_LABELS, c["id"])
            self.assertIn(c.get("gold_alt") or c["gold"], GOLD_LABELS, c["id"])
            for key in ("id", "trap", "derivation", "gold_note", "state"):
                self.assertTrue(c.get(key), "%s 缺少 %s" % (c["id"], key))
            st = c["state"]
            for key in ("topic", "result", "sources", "evidence"):
                self.assertIn(key, st, c["id"])
            self.assertIsInstance(st["sources"], list)
            self.assertIsInstance(st["evidence"], list)
            self.assertTrue(st["result"].strip(), c["id"])

    def test_rule_column_gap_is_whitelisted(self):
        # 逐条跑真实节点（safe_llm_invoke 强制抛 LLMInfraError → 走规则路径）
        gaps = []
        for c in self.corpus["cases"]:
            out = gvl._run(
                gvl._state_of(c), gate_off=False, coverage_on=True, repeat_on=True,
                invoke=mock.Mock(side_effect=nodes.LLMInfraError("单测：不调用 LLM")))
            verdict = gvl._summarize(out)["verdict"]
            if not gvl._ok_binary(c["gold"], verdict):
                gaps.append(c["id"])
        self.assertEqual(
            sorted(gaps), sorted(RULE_GAP),
            "规则列的偏差变了：%s。若是有意改动，请同步 docs/规则vsLLM-混淆矩阵.md 与白名单"
            % gaps)


if __name__ == "__main__":
    unittest.main()
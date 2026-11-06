# -*- coding: utf-8 -*-
"""规则 vs LLM 判决器对照实验（混淆矩阵）。

问题：「该不该判合格」这件事，纯语义裁判（E2）与确定性门控（E3）各错在哪？
做法：18 条带真值样本 × 3 个判决器，配对设计（每条样本只花 1 次 critic 调用）：

    R  规则单独 = LLM 完全不可用时的形态（等价 E6 产线）：把 safe_llm_invoke 打成
       LLMInfraError，走 nodes 的覆盖率前置否决 + rule_based_reflection（内含门控）
    L  纯 LLM   = E2 配置（DISABLE_CONTENT_GATE=1 + 关覆盖率/重复规则），真实调用 GLM
    C  组合     = E3 默认开关，**回放 L 列那次调用的原始返回**，差异只来自确定性门控

样本由 data/runs_v2/E3/*.json 的真实输出做受控损坏得到（--build 可重建），
真值按 v2 客观口径给出：证据是否为空 / 字数 ≥120 / URL 能否在来源清单核验 / 是否偏题。

用法：
    python experiments/gate_vs_llm.py --build      # 从真实记录派生样本集
    python experiments/gate_vs_llm.py --list       # 只打印样本集，人工复核真值
    python experiments/gate_vs_llm.py              # 跑三判决器 + 生成 docs/规则vsLLM-混淆矩阵.md
    python experiments/gate_vs_llm.py --repeat 3   # 重复 3 次，量化 LLM 自身抖动
    python experiments/gate_vs_llm.py --mock data/gate_vs_llm/<ts>.json  # 离线复算（不调 LLM）
"""
import argparse
import contextlib
import glob
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import agents.nodes as nodes  # noqa: E402  （必须在 sys.path 之后导入）
import config as cfg  # noqa: E402
from utils import _body_chars  # noqa: E402  （字数口径：正文，不含【来源】清单）

CASES_PATH = os.path.join(ROOT, "experiments", "samples", "gate_cases.json")
RUN_DIR = os.path.join(ROOT, "data", "runs_v2")
OUT_DIR = os.path.join(ROOT, "data", "gate_vs_llm")
OUT_DOC = os.path.join(ROOT, "docs", "规则vsLLM-混淆矩阵.md")

# 捕获真实实现：run 时 nodes.safe_llm_invoke 会被 mock 掉，避免自己递归自己
_REAL_INVOKE = nodes.safe_llm_invoke

# 论文口径的三态（降级 ≠ 失败，要单独成列；其余一切否决压成「否决」）
BUCKETS = ("通过", "降级通过", "否决")


def _bucket(verdict):
    if verdict == "通过":
        return "通过"
    if verdict == "降级通过":
        return "降级通过"
    return "否决"


def _ok_binary(gold, pred):
    # 二分类口径：只区分「合格(通过) / 非合格(降级或否决)」。
    # 降级与否决的差别是分类粒度，不该和「把不合格判成合格」混为一谈。
    return (_bucket(gold) == "通过") == (_bucket(pred) == "通过")


_CITED_LINE_RE = re.compile(r"^【来源】.*$", re.M)


def _load_record(topic_kw, group="E3"):
    """取该主题最新一份运行记录（按 mtime）；找不到返回 (None, None)。"""
    best = None
    for path in glob.glob(os.path.join(RUN_DIR, group, "*.json")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if topic_kw not in (rec.get("topic") or ""):
            continue
        key = os.path.getmtime(path)
        if best is None or key > best[0]:
            best = (key, path, rec)
    if best is None:
        return None, None
    return best[1], best[2]


def _evidence_from_sources(sources):
    """把记录里的 sources（title | url）还原成门控所需的最小 evidence 结构。

    运行记录不回传 evidence 原文，但门控只用到「非空」与 title/summary 的字面命中，
    因此用标题回填摘要即可等价复现（摘要长度不参与任何判定）。
    """
    items = []
    for s in sources or []:
        if " | " in s:
            title, url = s.split(" | ", 1)
        else:
            title, url = s, ""
        title, url = title.strip(), url.strip()
        if not title and not url:
            continue
        items.append({"title": title, "url": url, "summary": title, "content": ""})
    return items


def _strip_citation(text):
    return _CITED_LINE_RE.sub("", text or "").strip()


def _with_citation(text, line):
    return _strip_citation(text) + "\n" + line


def _truncate(text, n=60):
    body = re.sub(r"\s+", "", _strip_citation(text))
    return body[:n].rstrip("，、；：,;。") + "。"


def _url_host(url):
    return (url or "").split("://", 1)[-1].split("/", 1)[0]

def _case(cid, trap, derivation, gold, note, state, subjective=False,
          definition_sensitive=False, gold_alt=None):
    # gold_alt = 按当前代码实现口径的真值，只用于口径敏感性分析
    return {"id": cid, "trap": trap, "derivation": derivation, "gold": gold,
            "gold_note": note, "subjective": subjective,
            "definition_sensitive": definition_sensitive,
            "gold_alt": gold_alt or gold, "state": state}


def build_cases():
    """从真实记录派生 18 条带真值样本（受控损坏，全部可复现）。"""
    path_a, base_a = _load_record("生物细胞的结构")
    path_b, base_b = _load_record("比较Python和Java")
    path_c, base_c = _load_record("量子计算和经典计算的区别")
    if not (base_a and base_b and base_c):
        raise SystemExit("缺少基底记录：先跑 python experiments/run_matrix.py --groups E3")

    src_a = base_a.get("sources") or []
    src_c = base_c.get("sources") or []
    ev_a = _evidence_from_sources(src_a)
    ev_c = _evidence_from_sources(src_c)
    body_a = _strip_citation(base_a["result"])
    topic_a = base_a["topic"]

    pairs_a = [(s.split(" | ", 1)[0].strip(), s.split(" | ", 1)[1].strip())
               for s in src_a if " | " in s]
    title_a, url_a = pairs_a[0]
    url_a_plain = url_a.split("#", 1)[0]
    url_cx = next((u for _, u in pairs_a if "chaoxing" in u), url_a)
    # 落在路径边界上的截断写法（真实案例：模型把长 URL 截到 /ztnodedetailcontroller）
    url_cx_prefix = url_cx.split("?", 1)[0].rsplit("/", 1)[0]

    def st(result, sources=None, evidence=None, used_fallback=False,
           subtasks=None, topic=None):
        return {"topic": topic or topic_a, "result": result,
                "sources": list(src_a if sources is None else sources),
                "evidence": list(ev_a if evidence is None else evidence),
                "used_fallback": used_fallback, "subtasks": list(subtasks or [])}

    cases = []

    cases.append(_case(
        "S01", "无【来源】行", "删除真实回答末行的【来源】，其余原样",
        "降级通过", "有 5 条证据但 0 条可核验引用；v2 口径只能降级，不计入合格样本",
        st(body_a)))

    cases.append(_case(
        "S02", "编造 URL（外部域名）",
        "把【来源】的 URL 换成不存在的域名 fake.example.com",
        "内容不合格", "URL 不在检索来源清单里 → 按 v2 口径视为编造",
        st(_with_citation(body_a, "【来源】细胞的基本结构 https://fake.example.com/x"))))

    cases.append(_case(
        "S03", "编造 URL（同域不同路径）",
        "保留真实域名，只把路径段改掉（1265003 → 9999999）",
        "内容不合格", "同域但路径不匹配、也不是来源 URL 的前缀 → 编造",
        st(_with_citation(
            body_a, f"【来源】{title_a} https://{_url_host(url_a)}/gaokao/20200225/9999999.html"))))

    cases.append(_case(
        "S04", "过短且无来源", "正文截断到 60 字并去掉【来源】",
        "内容不合格", "60 字 < PASS_MIN_CHARS=120，字数门控先命中",
        st(_truncate(body_a, 60))))

    cases.append(_case(
        "S05", "正文过短但来源行很长",
        "正文截断到 60 字，【来源】保留真实 URL（来源行本身 ~90 字）",
        "内容不合格",
        "正文只有 60 字，【来源】把它撑到 154 字。字数门已改为按「正文」计（剔除【来源】"
        "及纯 URL 行），规则列现在判内容不合格；旧口径按全文计时会被注水放过 —— 那正是"
        "R 列最初缺口的成因，保留本样本用于回归",
        st(_with_citation(_truncate(body_a, 60), f"【来源】{title_a} {url_a}")),
        definition_sensitive=True, gold_alt="通过"))
    cases.append(_case(
        "S06", "URL 被全角括号+句号包裹",
        "【来源】写成「标题（URL）。」——真实模型常见写法",
        "通过", "URL 可核验，规则不应被中文标点绊倒（规则误伤风险点）",
        st(_with_citation(body_a, f"【来源】{title_a}（{url_a}）。"))))

    cases.append(_case(
        "S07", "URL 带锚点",
        "引用时追加 #section2（来源存的是 #1）",
        "通过", "_norm_url 去锚点后应精确命中（规则误伤风险点）",
        st(_with_citation(body_a, f"【来源】{title_a} {url_a_plain}#section2"))))

    cases.append(_case(
        "S08", "URL 被截断（prefix）",
        "把 chaoxing 长 URL 截到路径边界 /ztnodedetailcontroller（实跑案例）",
        "通过", "截断写法应判为可核验前缀，不能当编造（规则误伤风险点）",
        st(_with_citation(body_a, f"【来源】全屏显示 {url_cx_prefix}"))))

    cases.append(_case(
        "S09", "只写域名根",
        "【来源】只给 https://域名，不给具体页面（域名真实存在于来源清单）",
        "降级通过",
        "引用不完整：域名根不指向具体页面 → 不算可核验引用（应降级）。"
        "但门控把它并进「编造 URL」→ 判内容不合格，属规则过严（定义敏感项）",
        st(_with_citation(body_a, f"【来源】{title_a} https://{_url_host(url_a)}")),
        definition_sensitive=True, gold_alt="内容不合格"))

    cases.append(_case(
        "S10", "证据 0 条",
        "回答与来源保持原样，把 state.evidence 置空（模拟检索层没拿到东西）",
        "降级通过", "门控第 1 条：证据为空 → 降级，不能计入合格样本",
        st(body_a + f"【来源】{title_a} {url_a}", evidence=[])))
    cases.append(_case(
        "S11", "兜底未声明",
        "used_fallback=True（补检索失败、模型知识兜底）但回答没写【声明】",
        "降级通过", "构造项：覆盖门控第 2 条「无资料兜底」，真实链路少见",
        st(body_a, used_fallback=True)))

    cases.append(_case(
        "S12", "6 条来源只引 2 条",
        "来源清单 6 条（真实 5 条 + 叠加 1 条），回答只引用其中 2 条（引用率分母封顶 EVIDENCE_TOP_K=3）",
        "通过", "分母封顶后 2/3 已满足门槛；不能因「没全引」判不合格",
        st(_with_citation(body_a, "【来源】" + "；".join(
            f"《{t}》{u}" for t, u in pairs_a[:2])),
           sources=src_a + [src_c[0]])))

    cases.append(_case(
        "S13", "正文提及非来源链接",
        "正文插入「延伸阅读：https://other.example.com/blog/1」，其余（含【来源】）保持原样",
        "内容不合格",
        "按 v2 口径任何不可核验 URL 都视为风险；语义上回答可读，属规则假阳性（见文档局限）",
        st(_with_citation(
            body_a.replace("。", "（延伸阅读：https://other.example.com/blog/1）。", 1),
            f"【来源】{title_a} {url_a}")),
        definition_sensitive=True))

    cases.append(_case(
        "S14", "偏题（换主题回答）",
        "主题=生物细胞的结构，result/sources/evidence 全换成「量子计算和经典计算」的真实记录",
        "内容不合格",
        "格式与引用都合格，只有语义能发现「答非所问」——规则在这里必然失手",
        st(base_c["result"], sources=src_c, evidence=ev_c), subjective=True))

    py_topic = "比较Python和Java在并发编程与内存管理上的差异"
    py_sources = [
        "Python 并发编程与内存管理（派生自真实证据） | https://docs.python.org/3/library/asyncio.html",
        "Python GIL 与多进程实践（派生） | https://realpython.com/python-gil/",
        "Python 内存管理与引用计数（派生） | https://docs.python.org/3/c-api/memory.html",
    ]
    py_ev = _evidence_from_sources(py_sources)
    py_cite = "；".join("《%s》%s" % (s.split(" | ")[0], s.split(" | ")[1])
                        for s in py_sources[:2])
    cases.append(_case(
        "S15", "复合主题证据单侧",
        "topic=比较Python和Java…，subtasks=[Python, Java]，证据/来源只保留 Python 一侧（3 条标题改写）",
        "分解不合理",
        "子任务 Java 在证据里 0 命中 → 覆盖率 50%，规则前置否决，LLM 根本不参与",
        {"topic": py_topic,
         "result": _with_citation(_strip_citation(base_b["result"]), "【来源】" + py_cite),
         "sources": py_sources, "evidence": py_ev,
         "used_fallback": False, "subtasks": ["Python", "Java"]}))

    cases.append(_case(
        "S16", "无证据但已声明",
        "回答前置【声明】（基于模型知识），evidence/sources 置空、used_fallback=True",
        "降级通过", "降级 ≠ 不合格；关键是它不能计入合格样本",
        st("【声明】本文未获取到外部检索证据，内容基于模型内部知识，仅供参考。\n" + body_a,
           sources=[], evidence=[], used_fallback=True)))

    cases.append(_case(
        "S17", "真 URL + 编造 URL 混排",
        "【来源】里同时给真实可核验 URL 和 fake.example.com",
        "内容不合格", "一条真引用不能冲抵一条编造引用",
        st(_with_citation(
            body_a, f"【来源】{title_a} {url_a}；另一份资料 https://fake.example.com/y"))))

    cases.append(_case(
        "S18", "正常合格（对照组）",
        "E3 真实记录原样（含【来源】与 3 条可核验 URL）",
        "通过", "对照组：三列都应该判通过",
        st(base_a["result"])))

    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"样本 id 重复：{ids}")
    bad = [c["id"] for c in cases if c["gold"] not in cfg_gold_labels()]
    if bad:
        raise SystemExit(f"gold 取值非法：{bad}")

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "python experiments/gate_vs_llm.py --build",
            "n_cases": len(cases),
            "base_records": {"A": os.path.relpath(path_a, ROOT),
                             "B": os.path.relpath(path_b, ROOT),
                             "C": os.path.relpath(path_c, ROOT)},
            "note": ("样本由 data/runs_v2/E3 的真实输出受控损坏得到；gold 按 v2 客观口径："
                     "证据非空 / 字数≥120 / URL 可在来源清单核验 / 贴合主题。"),
        },
        "cases": cases,
    }


def cfg_gold_labels():
    return ("通过", "降级通过", "内容不合格", "分解不合理")


def _print_cases(cases):
    print(f"\n样本集：{len(cases)} 条")
    print("-" * 96)
    for c in cases:
        st_ = c["state"]
        ev = st_.get("evidence") or []
        flags = []
        if c.get("subjective"):
            flags.append("主观项")
        if c.get("definition_sensitive"):
            flags.append("定义敏感")
        print(f"{c['id']}  [{c['trap']}]  真值={c['gold']}"
              + (f"  （{'、'.join(flags)}）" if flags else ""))
        print(f"    主题：{st_['topic']}")
        print(f"    构造：{c['derivation']}")
        print(f"    判据：{c['gold_note']}")
        print(f"    事实：正文 {_body_chars(st_['result'])} 字 / 全文 "
              f"{len(st_['result'].strip())} 字 / 证据 {len(ev)} 条 / 来源 "
              f"{len(st_.get('sources') or [])} 条 / used_fallback={st_.get('used_fallback')} "
              f"/ subtasks={st_.get('subtasks') or '无'}")
        tail = re.sub(r"\s+", " ", st_["result"].strip())[-70:]
        print(f"    末尾：…{tail}")
        print()


def build_and_save(path=CASES_PATH):
    data = build_cases()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已生成 {os.path.relpath(path, ROOT)}（{len(data['cases'])} 条样本）")
    _print_cases(data["cases"])

def _state_of(case):
    """把样本还原成 reflection_node 需要的最小 state（每次判定都用全新副本）。"""
    st = case["state"]
    return {
        "topic": st["topic"],
        "result": st["result"],
        "sources": list(st.get("sources") or []),
        "evidence": [dict(e) for e in (st.get("evidence") or [])],
        "used_fallback": bool(st.get("used_fallback", False)),
        "subtasks": list(st.get("subtasks") or []),
        "reflection_history": [],
        "replan_count": 0,
        "llm_fail_count": 0,
        "run_stats": {"search": {"cache": "sample"}},
    }


def _fail_invoke(msg="对照实验：本列不调用 LLM"):
    """把 LLM 打成不可用 → 节点自动走 rule_based_reflection（规则列）。"""
    return mock.Mock(side_effect=nodes.LLMInfraError(msg))


class _Recorder:
    """记录真实 LLM 返回，供 C 列回放（配对设计：两列看同一条裁判输出）。"""

    def __init__(self):
        self.raw = []

    def __call__(self, prompt, use_critic=False, **kwargs):
        out = _REAL_INVOKE(prompt, use_critic=use_critic, **kwargs)
        self.raw.append(out)
        return out


def _replayer(raw):
    def _invoke(prompt, use_critic=False, **kwargs):
        return raw
    return _invoke


def _run(state, gate_off, coverage_on, repeat_on, invoke, quiet=True):
    """按指定开关跑一次 reflection_node；gate_off=True 即 E2（关掉确定性否决链）。"""
    patches = [
        mock.patch.object(nodes, "DISABLE_CONTENT_GATE", gate_off),
        mock.patch.object(nodes, "ENABLE_COVERAGE_RULE", coverage_on),
        mock.patch.object(nodes, "ENABLE_REPEAT_RULE", repeat_on),
        mock.patch.object(nodes, "ENABLE_REFLECTION", True),
        mock.patch.object(nodes, "safe_llm_invoke", invoke),
    ]
    buf = io.StringIO()
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        if quiet:
            stack.enter_context(contextlib.redirect_stdout(buf))
        return nodes.reflection_node(state)


def _summarize(out):
    refl = ((out or {}).get("run_stats") or {}).get("reflection") or {}
    return {
        "verdict": refl.get("verdict") or "（未判定）",
        "decision_source": refl.get("decision_source") or refl.get("gated_by") or "-",
        "reason": (refl.get("reason") or "").strip(),
        "gate": {k: refl.get(k) for k in ("has_source_line", "cited_count",
                                          "citation_ratio", "len_ok", "body_chars",
                                          "len_full", "cited_loose", "truncated",
                                          "truncation_hits")},
        # v2.2：救援要能逐条追溯（从哪个否决类型、按哪条规则改判的降级）
        "rescue_rule": refl.get("rescue_rule") or "",
        "rescued_from": refl.get("rescued_from") or "",
        "error_type": (out or {}).get("error_type"),
        "evidence_count": refl.get("evidence_count"),
        "result_chars": refl.get("result_chars"),
    }


def _judge_once(case, raw=None, quiet=True, allow_llm=True):
    """跑 L 与 C 两列。raw=None 且 allow_llm → 真实调用一次 critic 并返回其原始输出。"""
    recorder = _Recorder() if (raw is None and allow_llm) else None
    l_invoke = recorder if recorder else (
        _replayer(raw) if raw is not None
        else _fail_invoke("离线复算：没有缓存的 LLM 返回，L 列退化为规则兜底"))
    l_out = _run(_state_of(case), gate_off=True, coverage_on=False, repeat_on=False,
                 invoke=l_invoke, quiet=quiet)
    llm_raw = recorder.raw[0] if (recorder and recorder.raw) else (raw or "")

    c_invoke = (_replayer(llm_raw) if llm_raw else
                _fail_invoke("对照实验：L 列 LLM 未返回，C 列退回规则兜底"))
    c_out = _run(_state_of(case), gate_off=False, coverage_on=True, repeat_on=True,
                 invoke=c_invoke, quiet=quiet)
    return {"llm_raw": llm_raw, "L": _summarize(l_out), "C": _summarize(c_out)}


def _majority(verdicts):
    if not verdicts:
        return "（未判定）", 0.0
    best = max(verdicts, key=verdicts.count)
    return best, verdicts.count(best) / len(verdicts)


def run_cases(cases, repeat=1, cache=None, quiet=True):
    """逐条跑三列；cache 非空时进入离线复算（不调用 LLM）。"""
    allow_llm = cache is None
    rows = []
    for idx, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        r_res = _summarize(_run(_state_of(case), gate_off=False, coverage_on=True,
                                repeat_on=True, invoke=_fail_invoke(), quiet=quiet))
        tries = []
        for i in range(repeat):
            raw = None
            if cache is not None:
                lst = cache.get(case["id"]) or []
                if lst:
                    raw = lst[i] if i < len(lst) else lst[-1]
            tries.append(_judge_once(case, raw=raw, quiet=quiet, allow_llm=allow_llm))

        l_verdicts = [t["L"]["verdict"] for t in tries]
        c_verdicts = [t["C"]["verdict"] for t in tries]
        l_best, l_agree = _majority(l_verdicts)
        c_best, c_agree = _majority(c_verdicts)
        row = {
            "id": case["id"], "trap": case["trap"], "gold": case["gold"],
            "gold_note": case["gold_note"],
            "gold_alt": case.get("gold_alt") or case["gold"],
            "subjective": bool(case.get("subjective")),
            "definition_sensitive": bool(case.get("definition_sensitive")),
            "R": r_res,
            "L": dict(tries[-1]["L"], verdict=l_best,
                      agreement=round(l_agree, 2), all=l_verdicts),
            "C": dict(tries[-1]["C"], verdict=c_best,
                      agreement=round(c_agree, 2), all=c_verdicts),
            "llm_raw": [t["llm_raw"] for t in tries],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        rows.append(row)
        print(f"[{idx}/{len(cases)}] {case['id']} {case['trap']}：真值={case['gold']} "
              f"| R={r_res['verdict']} | L={l_best} | C={c_best} "
              f"（{row['elapsed_s']}s）")
    return rows

COLS = (("R", "规则单独（LLM 不可用）"),
        ("L", "纯 LLM（E2）"),
        ("C", "组合（E3）"))


def _pairs(rows, col):
    return [(r["gold"], r[col]["verdict"]) for r in rows]


def _matrix(pairs):
    m = {g: {p: 0 for p in BUCKETS} for g in BUCKETS}
    for gold, pred in pairs:
        m[_bucket(gold)][_bucket(pred)] += 1
    return m


def _binary(pairs):
    tp = fp = fn = tn = 0
    for gold, pred in pairs:
        g_ok, p_ok = _bucket(gold) == "通过", _bucket(pred) == "通过"
        if g_ok and p_ok:
            tp += 1
        elif g_ok:
            fn += 1
        elif p_ok:
            fp += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    fn_rate = fn / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n, "acc": acc,
            "prec": prec, "rec": rec, "f1": f1,
            "fp_rate": fp_rate, "fn_rate": fn_rate}


def _matrix_md(m):
    col_tot = {p: sum(m[g][p] for g in BUCKETS) for p in BUCKETS}
    total = sum(col_tot.values())
    lines = ["| 真值 \\ 判决 | 通过 | 降级通过 | 否决 | 合计 |",
             "|---|---|---|---|---|"]
    for g in BUCKETS:
        lines.append(f"| **{g}** | {m[g]['通过']} | {m[g]['降级通过']} | "
                     f"{m[g]['否决']} | {sum(m[g].values())} |")
    lines.append(f"| **合计** | {col_tot['通过']} | {col_tot['降级通过']} | "
                 f"{col_tot['否决']} | {total} |")
    return "\n".join(lines)


def _metric_md(rows):
    lines = ["| 判决器 | TP | FP | FN | TN | 准确率 | 精确率 | 召回率 | F1 | "
             "**误判率**（不合格→通过） | **漏判率**（合格→否决/降级） |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, name in COLS:
        b = _binary(_pairs(rows, key))
        lines.append(f"| {name} | {b['tp']} | {b['fp']} | {b['fn']} | {b['tn']} | "
                     f"{b['acc']:.0%} | {b['prec']:.0%} | {b['rec']:.0%} | {b['f1']:.2f} | "
                     f"**{b['fp_rate']:.0%}**（{b['fp']}/{b['fp'] + b['tn']}） | "
                     f"{b['fn_rate']:.0%}（{b['fn']}/{b['tp'] + b['fn']}） |")
    return "\n".join(lines)


def _trap_md(rows):
    order = []
    for r in rows:
        if r["trap"] not in order:
            order.append(r["trap"])
    lines = ["| 陷阱类型 | 样本 | 真值 | R 一致 | L 一致 | C 一致 |",
             "|---|---|---|---|---|---|"]
    for trap in order:
        group = [r for r in rows if r["trap"] == trap]
        golds = "、".join(sorted({r["gold"] for r in group}))
        cells = []
        for key in ("R", "L", "C"):
            hit = sum(1 for r in group
                      if _ok_binary(r["gold"], r[key]["verdict"]))
            cells.append(f"{hit}/{len(group)}")
        lines.append(f"| {trap} | {len(group)} | {golds} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _detail_md(rows):
    lines = ["| # | 陷阱 | 真值 | R | L | C | C 字数（正文/全文） | R/C 引用数 | C 判定来源 | L 的理由（截断） |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def cell(key):
            v = r[key]["verdict"]
            mark = "OK" if _ok_binary(r["gold"], v) else "X"
            return f"{mark} {v}"
        c_gate = r["C"].get("gate") or {}
        r_gate = r["R"].get("gate") or {}
        cited = f"{r_gate.get('cited_count')}/{c_gate.get('cited_count')}"
        reason = re.sub(r"\s+", " ", r["L"].get("reason") or "")[:56]
        chars = f"{c_gate.get('body_chars') or '-'}/{r['C'].get('result_chars') or '-'}"
        lines.append(f"| {r['id']} | {r['trap']} | {r['gold']} | {cell('R')} | "
                     f"{cell('L')} | {cell('C')} | {chars} | "
                     f"{cited} | {r['C'].get('decision_source', '-')} | {reason} |")
    return "\n".join(lines)


def _conclusions(rows, repeat):
    bad = [r for r in rows if _bucket(r["gold"]) != "通过"]
    good = [r for r in rows if _bucket(r["gold"]) == "通过"]
    lines = []
    for key, name in COLS:
        passed_bad = [r["id"] for r in bad if _bucket(r[key]["verdict"]) == "通过"]
        killed_good = [r["id"] for r in good if _bucket(r[key]["verdict"]) != "通过"]
        seg = (f"- **{name}**：非合格样本 {len(bad)} 条中判「通过」**{len(passed_bad)}** 条"
               f"（误判率 {len(passed_bad) / len(bad):.0%}）")
        if passed_bad:
            seg += f"，误判：{'、'.join(passed_bad)}"
        seg += f"；合格样本 {len(good)} 条中误伤 {len(killed_good)} 条"
        seg += f"（{'、'.join(killed_good)}）" if killed_good else "（无误伤）"
        lines.append(seg)

    only_llm = [r["id"] for r in rows
                if _ok_binary(r["gold"], r["L"]["verdict"])
                and not _ok_binary(r["gold"], r["R"]["verdict"])]
    only_rule = [r["id"] for r in rows
                 if _ok_binary(r["gold"], r["R"]["verdict"])
                 and not _ok_binary(r["gold"], r["L"]["verdict"])]
    if only_llm:
        lines.append(f"- **只有语义裁判能判对**（规则结构上做不到）：{'、'.join(only_llm)}")
    if only_rule:
        lines.append(f"- **只有规则能判对**（语义裁判放过）：{'、'.join(only_rule)}")

    saved = [r["id"] for r in rows
             if r["C"].get("decision_source") == "coverage_rule"]
    demoted = [r["id"] for r in rows
               if _bucket(r["gold"]) == "降级通过"
               and _bucket(r["C"]["verdict"]) == "否决"]
    if demoted:
        lines.append(f"- **C 列会丢掉「降级」这一档**：{len(demoted)} 条真值 `降级通过` 被判成 `否决`"
                     f"（{'、'.join(demoted)}）。降级救援（v2.2）只覆盖「声明式兜底 / 无资料兜底 / "
                     "未引用来源」三类，其余否决仍无法恢复成降级 → 二分类指标不受影响，"
                     "但按「降级率」统计会偏低（见第 6 节局限）。")
    rescued = [f"{r['id']}（{r['C'].get('rescue_rule')}）" for r in rows
               if r["C"].get("rescue_rule")]
    if rescued:
        lines.append(f"- **降级救援（v2.2 双向门控）**：C 列把 {len(rescued)} 条「可读但缺资料」的"
                     f"否决改判为 `降级通过`：{'、'.join(rescued)}。救援**永不**改判为 `通过`，"
                     "因此二分类（合格/非合格）指标不变，只修正三态粒度。")
    if saved:
        lines.append("- **覆盖率规则前置否决**（不做 LLM 调用就直接判「分解不合理」）："
                     + "、".join(saved))
    if repeat > 1:
        unstable = [f"{r['id']}（一致率 {r['L']['agreement']:.0%}）" for r in rows
                    if r["L"]["agreement"] < 1.0]
        lines.append(f"- **LLM 自身抖动**（每条重复 {repeat} 次）："
                     + ("、".join(unstable) if unstable
                        else f"全部 {repeat} 次判定一致"))
    return "\n".join(lines)

def _cases_md(cases):
    lines = ["| id | 陷阱 | 构造（受控损坏） | 真值 | 判据 |",
             "|---|---|---|---|---|"]
    for c in cases:
        lines.append(f"| {c['id']} | {c['trap']} | {c['derivation']} | **{c['gold']}** | "
                     f"{c['gold_note']} |")
    return "\n".join(lines)


def _sensitivity_md(rows):
    """主口径 vs 实现口径。

    差异来源不再写死样本号：把「gold_alt 与 gold 不同」的样本动态列出来，
    这样修完 S05 的字数门之后表格不会跟着过时。
    """
    diff_ids = [r["id"] for r in rows
                if (r.get("gold_alt") or r["gold"]) != r["gold"]]
    reason = ("换成 gold_alt 的样本：" + "、".join(diff_ids)) if diff_ids else "无差异样本"
    lines = ["| 判决器 | 主口径准确率（合格=通过） | 实现口径准确率 | 差异来源 |",
             "|---|---|---|---|"]
    for key, name in COLS:
        main = _binary(_pairs(rows, key))["acc"]
        alt = _binary([(r.get("gold_alt") or r["gold"], r[key]["verdict"])
                       for r in rows])["acc"]
        lines.append(f"| {name} | {main:.0%} | {alt:.0%} | {reason} |")
    return chr(10).join(lines)


def _limitations_md(rows, repeat):
    n = len(rows)
    return "\n".join([
        f"1. **样本量 n={n}**：只能看趋势与失败模式，不能给置信区间；"
        "每类陷阱 1 条，回答的是「这类错误有没有可能被抓到」，不是该类的错误率。",
        "2. **真值是「定义真值」**：按 v2 口径（证据非空 / 字数 / URL 可核验 / 贴合主题）机械判定。"
        "其中 **S13（正文提及非来源链接）** 是*定义敏感项*：语义上回答可读，"
        "是 v2 定义对正文外链过度严厉造成的**规则假阳性**。若口径改为「只核验【来源】行」，"
        "S13 真值应改为「通过」，此时 R/L/C 三列各多一条「把合格判成不合格」的误伤 —— "
        "指标完全由定义决定，这也是本实验把 S13 当讨论项、不当判决器优劣证据的原因。",
        "3. **主观项**：**S14（偏题）** 的真值依赖人工判断「答非所问」，"
        "它是规则结构上无法处理的类别（偏题没有字面信号），也是 R 列现在唯一的缺口。"
        "**S05（来源行注水）** 属于「字数统计口径」问题，已改为按正文计字数（剔除【来源】），"
        "规则列不再放过——性质与 S14 不同，属可修的实现缺陷，保留样本作回归。",
        "4. **配对设计的边界**：C 列回放的是 L 列那次调用的原始返回，"
        "而 L 列（E2 配置）的 prompt 不含覆盖率统计行；所以结论是"
        "「在**同一份**语义判断之上再叠门控能修掉什么」，不是「E3 完整 prompt 会带来什么」。",
        f"5. **temperature=0.1**：critic 仍有抖动，本文件 repeat={repeat}；"
        "要彻底可复现需要把 `config.py` 的 `temperature` 降到 0（属独立的 A 项改动）。",
        "6. **R 列不是随手写的正则**：它复用产线规则（覆盖率前置否决 + 门控 + 规则兜底），"
        "代表「LLM 完全不可用时的下限」，因此 R 列的表现应当被当作规则能力的**上界估计**。",
        "7. **S11（兜底未声明）是构造项**：真实链路里 `used_fallback=True` 时证据通常也为空，"
        "会先命中「证据为空」分支；该条用于覆盖门控第 2 条分支，不代表线上分布。",
        f"8. **类别不平衡**：真值 {sum(1 for r in rows if _bucket(r['gold']) == '通过')} 合格 / "
        f"{len(rows) - sum(1 for r in rows if _bucket(r['gold']) == '通过')} 非合格，"
        "准确率会被非合格一侧主导，所以第 3 节同时给出精确率/召回率，"
        "结论段里的「误判率」才是本实验的主指标。",
        "9. **降级救援（v2.2）的覆盖边界**：救援只处理「可读但缺资料」三类（声明式兜底 / "
        "无资料兜底 / 未引用来源），且要求正文够长、无编造 URL、未截断。S09（只写域名根）"
        "因此仍被记成 `内容不合格`：域名根既不是可核验引用、又与「编造 URL」共用同一个否决档，"
        "属**有意保留的保守口径**（宁可判否决，不可放行）。做「降级率」统计时需单列这一条。",
    ])


def _write_doc(rows, cases, result_path, repeat=1, mode="在线跑（每条 1 次 critic 调用）"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "# 规则 vs LLM 判决器对照实验（混淆矩阵）",
        "",
        f"> **自动生成**：`python experiments/gate_vs_llm.py`（{now}）",
        f"> **运行方式**：{mode}",
        f"> **样本**：{len(rows)} 条，由 `data/runs_v2/E3` 的真实输出受控损坏得到（附录 A）",
        f"> **真值口径**：v2（证据非空 / 字数 ≥{cfg.PASS_MIN_CHARS} / URL 可在来源清单核验 / 贴合主题）",
        "> **配对设计**：L 与 C 看到的是**同一条** critic 返回（同一次调用），"
        "差异只来自确定性门控；每条样本只花 1 次 LLM 调用。",
        f"> **模型**：critic = `{cfg.CRITIC_MODEL}`，temperature=0.1，repeat={repeat}",
        f"> **原始结果**：`{os.path.relpath(result_path, ROOT)}`",
        "",
        "## 1. 结论（由数据自动生成）",
        "",
        _conclusions(rows, repeat),
        "",
        "## 2. 三分类混淆矩阵（行=真值，列=判决）",
        "",
        "> 与 `docs/实验记录-v2.md` 同口径：`通过` = 计入合格样本；"
        "`降级通过` = 可读但无可核验支撑，单独成列；`否决` = 内容不合格 / 分解不合理 / 检索失败。",
        "",
        "### 2.1 规则单独（R）",
        "",
        _matrix_md(_matrix(_pairs(rows, "R"))),
        "",
        "### 2.2 纯 LLM（L，E2）",
        "",
        _matrix_md(_matrix(_pairs(rows, "L"))),
        "",
        "### 2.3 组合（C，E3）",
        "",
        _matrix_md(_matrix(_pairs(rows, "C"))),
        "",
        "> **三态矩阵里 `降级通过` ↔ `否决` 的混淆不要读成错误**："
        "两者都不计入合格样本，差别只是「可读但没资料支撑」与「不可用」的粒度；"
        "真正的错误只有 `通过` ↔ 非 `通过` 的跨类，见第 3、4 节的二分类口径。",
        "",
        "## 3. 二分类指标（合格 = `通过` vs 非合格）",
        "",
        _metric_md(rows),
        "",
        "> **误判率** = 非合格样本里被判 `通过` 的比例（污染合格样本，是本实验的主指标）；"
        "**漏判率** = 合格样本被降级/否决的比例（代价是多一次重规划）。",
        "",
        "## 4. 分陷阱识别率（二分类口径：合格 vs 非合格）",
        "",
        _trap_md(rows),
        "",
        "## 5. 逐条明细（OK/X 按二分类口径，判定词照原样）",
        "",
        _detail_md(rows),
        "",
        "## 6. 局限（人工撰写，引用前必读）",
        "",
        _limitations_md(rows, repeat),
        "",
        "## 7. 口径敏感性（审稿人最可能追问的地方）",
        "",
        _sensitivity_md(rows),
        "",
        "> 主口径的真值按「论文定义」给（正文 ≥120 字、引用要指向具体页面）；实现口径直接取当前代码行为。"
        "字数门已改为**按正文计**（不再被【来源】注水），因此两者现在只在 S09（域名根）上仍有分歧："
        "**域名根判定比定义更严厉**，属有意保留的保守口径（宁可降级，不可放行）。",
        "",
        "## 附录 A. 样本集（受控损坏，可复核）",
        "",
        _cases_md(cases),
        "",
        "## 附录 B. 复现命令",
        "",
        "```powershell",
        "# 1) 从真实记录重建样本集（写 experiments/samples/gate_cases.json）",
        "python experiments/gate_vs_llm.py --build",
        "",
        "# 2) 跑三判决器 + 生成本文档（18 次 critic 调用）",
        "python experiments/gate_vs_llm.py",
        "",
        "# 3) 稳定性：每条重复 3 次（54 次调用）",
        "python experiments/gate_vs_llm.py --repeat 3",
        "",
        "# 4) 离线复算（不调 LLM，读已有结果 JSON）",
        "python experiments/gate_vs_llm.py --mock data/gate_vs_llm/<时间戳>.json",
        "```",
        "",
    ]
    text = "\n".join(parts)
    with open(OUT_DOC, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"已生成 {os.path.relpath(OUT_DOC, ROOT)}")
    return text


def _load_cases(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cases"]


def _save_results(rows, repeat):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"gate_vs_llm-{datetime.now():%Y%m%d-%H%M%S}.json")
    payload = {
        "meta": {
            "ts": datetime.now(timezone.utc).isoformat(),
            "repeat": repeat,
            "n_cases": len(rows),
            "critic_model": cfg.CRITIC_MODEL,
            "pass_min_chars": cfg.PASS_MIN_CHARS,
            "pass_min_cited": cfg.PASS_MIN_CITED,
            "evidence_top_k": cfg.EVIDENCE_TOP_K,
            "columns": {"R": "规则单独（safe_llm_invoke 强制失败）",
                        "L": "DISABLE_CONTENT_GATE=1 + 关覆盖率/重复（E2）",
                        "C": "默认开关（E3），回放 L 列的 LLM 返回"},
        },
        "cases": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已记录 {os.path.relpath(path, ROOT)}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="规则 vs LLM 判决器对照实验（混淆矩阵）")
    ap.add_argument("--build", action="store_true", help="从 data/runs_v2/E3 真实记录重建样本集")
    ap.add_argument("--list", action="store_true", help="只打印样本集（人工复核真值）")
    ap.add_argument("--repeat", type=int, default=1, help="每条样本重复次数（默认 1）")
    ap.add_argument("--mock", default="", help="离线复算：读已有结果 JSON，不调用 LLM")
    ap.add_argument("--cases", default=CASES_PATH, help="样本集路径")
    ap.add_argument("--no-doc", action="store_true", help="只跑，不写 Markdown")
    ap.add_argument("--verbose", action="store_true", help="打印节点内部日志")
    args = ap.parse_args(argv)

    if args.build:
        build_and_save()
        return 0

    if not os.path.exists(args.cases):
        print(f"未找到样本集 {os.path.relpath(args.cases, ROOT)}，先跑 --build")
        return 2
    cases = _load_cases(args.cases)
    if args.list:
        _print_cases(cases)
        return 0

    cache = None
    if args.mock:
        with open(args.mock, encoding="utf-8") as f:
            old = json.load(f)
        cache = {c["id"]: c.get("llm_raw") or [] for c in old.get("cases", [])}
        print(f"离线复算：{os.path.relpath(args.mock, ROOT)}"
              f"（{len(cache)} 条缓存返回，不调用 LLM）")

    rows = run_cases(cases, repeat=max(1, args.repeat), cache=cache,
                     quiet=not args.verbose)
    result_path = _save_results(rows, max(1, args.repeat))
    if not args.no_doc:
        mode = ("离线复算（`--mock`：复用已有 critic 返回，只重算确定性门控与 R 列；"
                "不产生新的 LLM 调用）") if args.mock else "在线跑（每条 1 次 critic 调用）"
        _write_doc(rows, cases, result_path, repeat=max(1, args.repeat), mode=mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
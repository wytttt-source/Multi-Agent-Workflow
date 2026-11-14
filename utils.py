#报错与容错、缓存
import time
import re

# ================= 3. LLM 调用容错 =================
class LLMInfraError(Exception):
    pass

class LLMParseError(Exception):
    pass

_SEARCH_CACHE: dict = {}

# 单次运行的 LLM 调用次数（P2-2：成本-收益图表的数据来源）。
# main.py 在每次 app.invoke 前调用 reset_llm_calls()，反思节点把 count 写进 run_stats。
_LLM_CALLS = {"n": 0}


def reset_llm_calls() -> None:
    _LLM_CALLS["n"] = 0


def llm_call_count() -> int:
    return _LLM_CALLS["n"]

# LLM 基础设施故障识别：openai SDK 的超时文案是 "Request timed out."，
# 只匹配 "timeout" 会漏判，这里同时看关键字与异常类名。
_INFRA_KEYWORDS = (
    "503", "502", "500", "429", "Service is too busy", "service_unavailable",
    "timeout", "Timeout", "timed out", "Connection", "connection",
    "rate limit", "RateLimit", "overloaded", "temporarily",
)
_INFRA_ERROR_NAMES = ("Timeout", "Connection", "RateLimit", "Overloaded")

def safe_llm_invoke(prompt: str, use_critic: bool = False, retries: int = 2, backoff: float = 3.0) -> str:
    from config import llm_critic, llm_generator, INJECT_LLM_FAIL
    target_llm = llm_critic if use_critic else llm_generator
    _LLM_CALLS["n"] += 1

    # E6 故障注入：前 N 次调用直接抛基础设施错误，用于验证双熔断能否收敛
    if INJECT_LLM_FAIL and _LLM_CALLS["n"] <= INJECT_LLM_FAIL:
        raise LLMInfraError(f"[故障注入] 第 {_LLM_CALLS['n']} 次 LLM 调用被强制失败")

    last_err = None
    
    for attempt in range(retries + 1):
        try:
            response = target_llm.invoke(prompt)
            return response.content
        except Exception as e:
            msg = str(e)
            is_infra = (any(k in msg for k in _INFRA_KEYWORDS)
                        or any(k in type(e).__name__ for k in _INFRA_ERROR_NAMES))
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

def _norm_url(url: str) -> str:
    """URL 归一化：去锚点、去中文/英文尾随标点、统一大小写与结尾斜杠。

    只用于"回答里引用的 URL 是否真的在检索来源里"的比对，
    不改变展示用的原始 URL。
    """
    u = (url or "").strip()
    u = u.split("#", 1)[0]
    u = u.rstrip("。，；、,;:!?！？\"'”’）)】]}>").strip()
    return u.rstrip("/").lower()


_CITED_URL_RE = re.compile(r"https?://[^\s，。；、,;!?！？\"'”’）)】\]}>]+")

# 来源行标记：回答正文与"来源清单"的分界
_SOURCE_MARK = "【来源】"

# ================= 完整性信号（v2.2） =================
# 只覆盖"确定能判定"的形态：结尾停在分隔符、停在连接词、括号没闭合。
# 语义截断（以句号收尾但话没说完）不在确定性范围内，靠字数门与语义裁判兜住。
_TRUNC_TAIL_CHARS = "，、；：,;:…"
_TRUNC_CONNECTIVE_TAILS = ("以及", "和", "与", "及", "等", "例如", "比如",
                           "但是", "而且", "并且", "其中", "包括")
_BRACKET_PAIRS = (("（", "）"), ("(", ")"), ("【", "】"), ("「", "」"),
                  ("《", "》"), ("“", "”"))


def _body_text(text: str) -> str:
    """回答「正文」：剔除【来源】及其后的全部内容，并去掉整行只有 URL 的行。

    字数门与完整性检查共用同一份正文，避免两处口径再次分叉。
    """
    head = (text or "").split(_SOURCE_MARK, 1)[0]
    lines = [ln.strip() for ln in head.splitlines()]
    return "\n".join(ln for ln in lines if ln and not _CITED_URL_RE.fullmatch(ln))


def _body_chars(text: str) -> int:
    """回答「正文」字数：剔除【来源】及其后的全部内容，并去掉整行只有 URL 的行。

    为什么要按正文而不按全文计：旧口径用 len(全文)，模型只要在【来源】里堆长标题
    就能把 60 字的正文撑到 154 字（样本 S05），字数门形同虚设。
    """
    return len(_body_text(text))


def _looks_truncated(text: str) -> dict:
    """确定性截断信号：只报事实，判定在 agents._apply_pass_gate。

    返回 {"truncated": bool, "hits": [...]}：hits 直接写进落盘，便于论文里复查
    "这条为什么判内容不合格"——不用回看原文猜。
    """
    body = _body_text(text).rstrip()
    if not body:
        return {"truncated": False, "hits": []}

    hits = []
    if body[-1] in _TRUNC_TAIL_CHARS:
        hits.append("结尾停在分隔符")
    else:
        for w in _TRUNC_CONNECTIVE_TAILS:
            if body.endswith(w):
                hits.append(f"结尾停在连接词({w})")
                break
    for left, right in _BRACKET_PAIRS:
        n_left, n_right = body.count(left), body.count(right)
        if n_left != n_right:
            hits.append(f"括号不配对({right if n_left > n_right else left})")
            break
    return {"truncated": bool(hits), "hits": hits}


def _verified_kind(url: str, source_urls: set) -> str:
    """判断引用 URL 是否可核验：exact=完全一致，prefix=来源 URL 的截断写法。

    实跑案例（2026-11-05，`生物细胞的结构`）：模型把
    `https://mooc1.chaoxing.com/.../visitnodedetail?courseId=200901994&knowledgeId=...`
    截断成 `.../visitnodedetail` 后引用 —— 精确匹配会误判为"编造来源"，
    从而把一条真实可核验的回答判成 `内容不合格` 并触发一次无谓重规划。
    因此允许"来源 URL 去 query 后的前缀"，但必须在路径边界处截断，防止
    `https://a.com/x` 蒙成 `https://a.com/xyz`。
    """
    if url in source_urls:
        return "exact"
    # 只写域名根（如 https://a.com）不算"引用某条来源"，避免用首页糊弄过去
    if "/" not in url.split("://", 1)[-1]:
        return ""
    for s in source_urls:
        if s.startswith(url) and len(s) > len(url) and s[len(url)] in "?/":
            return "prefix"
    return ""


def content_gate(result: str, sources: list, evidence: list,
                 min_chars: int = None, min_citation_ratio: float = None) -> dict:
    """确定性门控：把"回答是否满足可判通过的最低条件"变成可复现的事实清单。

    只报事实，不做判定（判定见 agents._apply_pass_gate），便于单测与论文复算。
    """
    from config import PASS_MIN_CHARS, EVIDENCE_TOP_K, PASS_MIN_CITATION_RATIO
    text = (result or "").strip()
    limit = PASS_MIN_CHARS if min_chars is None else min_chars
    limit_ratio = (PASS_MIN_CITATION_RATIO if min_citation_ratio is None
                   else min_citation_ratio)
    body_chars = _body_chars(text)
    trunc = _looks_truncated(text)

    cited = _CITED_URL_RE.findall(text)
    source_urls = set()
    for s in sources or []:
        # 来源行格式是"标题 | URL"，但标题本身可能再含 " | "（如
        # "5.3 全文与倒排对照 | SQLite内核与优化 - 灏天文库 | https://aiknowledge.cn/..."），
        # 用 split(" | ")[1] 会把"标题后半段 | URL"整串当成 URL，于是答案里逐字引用的
        # 真实来源被判成"编造 URL"（2026-11-14 P-H/P-R 的 B2 就是 1/3 条引用这样误杀，
        # 一份 649 字、对象全覆盖的回答被直接判 failed）。所以从**最后一个**分隔符切。
        # 也不能用 _CITED_URL_RE 兜底提取：它的字符类排除了 "?"，
        # 会把 `...visitnodedetail?courseId=...` 截成 `...visitnodedetail`，
        # 于是"截断写法"的引用（C3b 案例）失去前缀比对依据。
        raw = (s or "").strip()
        url = raw.rsplit(" | ", 1)[1].strip() if " | " in raw else ""
        if not url.lower().startswith(("http://", "https://")):
            m = _CITED_URL_RE.search(raw)
            url = m.group(0) if m else ""
        if url:
            source_urls.add(_norm_url(url))
    fabricated = [u for u in cited if not _verified_kind(_norm_url(u), source_urls)]
    verified = {u for u in (_norm_url(x) for x in cited)
                if _verified_kind(u, source_urls) == "exact"}
    loose = {u for u in (_norm_url(x) for x in cited)
             if _verified_kind(u, source_urls) == "prefix"}
    # 引用率分母取"可用来源数，上限 EVIDENCE_TOP_K（默认 3）"：
    # 检索给了 6 条但只引用 3 条不算违规；阈值外置后做实验可直接调
    denom = max(1, min(len(source_urls), EVIDENCE_TOP_K))
    ratio = round(min((len(verified) + len(loose)) / denom, 1.0), 3)

    return {
        # 字数门按「正文」计：来源清单再长也不算内容（否则 S05 靠来源行注水过关）
        "len_ok": body_chars >= limit,
        "len_limit": limit,
        "body_chars": body_chars,
        "len_full": len(text),
        "has_source_line": "【来源】" in text,
        "no_fabricated_url": len(fabricated) == 0,
        "evidence_nonempty": bool(evidence),
        "fabricated_urls": fabricated[:3],
        "cited_count": len(verified) + len(loose),
        "cited_loose": sorted(loose)[:3],
        # 分母封顶 EVIDENCE_TOP_K，分子超过门槛时压到 1.0（原始条数看 cited_count）
        "citation_ratio": ratio,
        "citation_ratio_ok": ratio >= limit_ratio,
        "citation_ratio_limit": limit_ratio,
        # 完整性信号（v2.2）：只覆盖可判定的截断形态，语义截断交给语义裁判
        "truncated": trunc["truncated"],
        "truncation_hits": trunc["hits"],
    }


# 「资料未提供 / 仅覆盖 X」这类**可得性声明**：句子里有对象名，但没有讲这个对象。
# 判卷时若把这些句子算作覆盖，就会出现"答案自己声明没查到 Kafka"却被判为讲清了 Kafka
# （v2.4 复核：Kafka / PostgreSQL / Podman 三例都这样，系统判不合格、判卷判合格 = 假误否决）。
# 只匹配"资料/数据/证据…"与否定词紧邻的写法，避免把
# "Redis 未提供事务回滚，而 MySQL 提供" 这类**实质结论**当成声明句删掉。
_MATERIAL_RE = r"(?:资料|数据|证据|信息|来源|文献|材料|内容)"
_AVAILABILITY_RE = re.compile(
    rf"{_MATERIAL_RE}[^，。；]{{0,6}}?(?:未|没有|无|不足|仅|只)"
    rf"|(?:未|没有|无)(?:提供|涉及|检索到|找到|获取到|覆盖|给出|提及|披露)"
    rf"[^，。；]{{0,20}}?{_MATERIAL_RE}"
    rf"|{_MATERIAL_RE}[^，。；]{{0,10}}?(?:中)?(?:未|没有)"
    rf"|无法(?:比较|获取|确认|判断|展开|据实)"
    rf"|缺少{_MATERIAL_RE}"
    # 拒答/免责型结尾（2026-11-14 P-F 的 B4 实跑）：模型不再伪装覆盖，而是整段
    # 说明"我不能把资料改写成三者比较""当前资料不足以完成比较"，句子里照样出现
    # 全部对象名。这类句子必须一起剔掉，否则一份"只讲 Docker、其余声明查不到"的
    # 拒答会被判成三对象全覆盖（系统判否决 → 记成误否决）。
    rf"|不能把[^。；]{{0,40}}?(?:改写|写成|拼成|凑成|补齐成)"
    rf"|不足以(?:完成|支撑|作|进行|给出|形成|覆盖)"
    rf"|只能(?:引入|依据|诉诸|靠)(?:无来源|自身知识|模型内部)"
    # "下一步该补哪些资料"的施工说明（同一份 B4 拒答的第二段）：句子里点名了
    # Podman / containerd，但给的是检索计划（"需补充三类官方资料…""可按 site:… 检索式补齐"），
    # 不是关于这些对象的内容。不剔掉就会把施工计划算成覆盖。
    rf"|(?:需|需要|应|可|建议)[^。；]{{0,10}}?(?:补充|补齐|补检|检索)[^。；]{{0,24}}?(?:资料|文档|证据|检索式)"
    rf"|检索式")


def _substantive_text(text: str) -> str:
    """剔掉可得性声明句，只留"实际讲了内容"的句子（判卷专用）。

    只按强标点（。；！？换行）切句：若连逗号也切，
    "Kafka 吞吐更高，而 RabbitMQ 数据未提供" 会被切成两半，前半仍然是实质内容；
    反之若整段一起判，又可能把同一段里的实质内容一起丢掉——强标点是折中。
    """
    parts = re.split(r"[。；;！？!?\n]", text or "")
    keep = [p.strip() for p in parts if p.strip() and not _AVAILABILITY_RE.search(p)]
    return "。".join(keep)


def answer_completeness(result: str, required_objects: list,
                        min_chars: int = None, substantive: bool = False) -> dict:
    """答案完整性复核（v2.3）：对**最终答案**检查"该覆盖的对象是否都覆盖了"。

    为什么需要：`content_gate` 只看字数/引用/截断，看完不知道"回答是否答完"。
    补检索重写之后最容易退化成"补上了引用、却只讲了一个对象"，仅靠字数门看不出来。
    因此这里改用「任务给定的必答对象」做字面覆盖，口径与 content_gate 的正文一致
    （同一个 `_body_text`），三臂共用同一份判定，避免各臂自己判自己。

    参数 `substantive=True`（判卷用）：先剔除"资料未提供 / 仅涉及 X"这类**可得性声明**句
    再数字面覆盖。理由见 `_substantive_text`：那种句子只是提了对象名，没有讲对象。
    系统侧的判定仍用默认 False（口径一旦改动，历史记录就不可比）。

    返回 dict：
    - terms / hit / missing / ratio：必答对象的覆盖情况
    - body_chars / truncated：与门控同口径
    - complete：覆盖全 + 未截断 + 正文达到字数下限（可直接当"合格"的硬条件）
    """
    from config import PASS_MIN_CHARS
    limit = PASS_MIN_CHARS if min_chars is None else min_chars
    body = _body_text(result or "")
    scope = _substantive_text(body) if substantive else body
    terms = [t for t in (required_objects or []) if t]
    hit = [t for t in terms if t in scope]
    missing = [t for t in terms if t not in scope]
    body_chars = _body_chars(result or "")
    trunc = _looks_truncated(result or "")["truncated"]
    return {
        "terms": terms,
        "hit": hit,
        "missing": missing,
        "ratio": round(len(hit) / len(terms), 3) if terms else 1.0,
        "body_chars": body_chars,
        "truncated": trunc,
        "substantive": bool(substantive),
        "complete": (not missing) and (not trunc) and body_chars >= limit,
    }


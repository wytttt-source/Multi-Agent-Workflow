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


def _body_chars(text: str) -> int:
    """回答「正文」字数：剔除【来源】及其后的全部内容，并去掉整行只有 URL 的行。

    为什么要按正文而不按全文计：旧口径用 len(全文)，模型只要在【来源】里堆长标题
    就能把 60 字的正文撑到 154 字（样本 S05），字数门形同虚设。
    """
    head = (text or "").split(_SOURCE_MARK, 1)[0]
    lines = [ln.strip() for ln in head.splitlines()]
    body = "\n".join(ln for ln in lines if ln and not _CITED_URL_RE.fullmatch(ln))
    return len(body)


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
                 min_chars: int = None) -> dict:
    """确定性门控：把"回答是否满足可判通过的最低条件"变成可复现的事实清单。

    只报事实，不做判定（判定见 agents._apply_pass_gate），便于单测与论文复算。
    """
    from config import PASS_MIN_CHARS, EVIDENCE_TOP_K
    text = (result or "").strip()
    limit = PASS_MIN_CHARS if min_chars is None else min_chars
    body_chars = _body_chars(text)

    cited = _CITED_URL_RE.findall(text)
    source_urls = set()
    for s in sources or []:
        if " | " in s:
            source_urls.add(_norm_url(s.split(" | ", 1)[1]))
    fabricated = [u for u in cited if not _verified_kind(_norm_url(u), source_urls)]
    verified = {u for u in (_norm_url(x) for x in cited)
                if _verified_kind(u, source_urls) == "exact"}
    loose = {u for u in (_norm_url(x) for x in cited)
             if _verified_kind(u, source_urls) == "prefix"}
    # 引用率分母取"可用来源数，上限 EVIDENCE_TOP_K（默认 3）"：
    # 检索给了 6 条但只引用 3 条不算违规；阈值外置后做实验可直接调
    denom = max(1, min(len(source_urls), EVIDENCE_TOP_K))

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
        "citation_ratio": round(min((len(verified) + len(loose)) / denom, 1.0), 3),
    }


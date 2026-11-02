#报错与容错、缓存
import time

# ================= 3. LLM 调用容错 =================
class LLMInfraError(Exception):
    pass

class LLMParseError(Exception):
    pass

_SEARCH_CACHE: dict = {}

def safe_llm_invoke(prompt: str, use_critic: bool = False, retries: int = 2, backoff: float = 3.0) -> str:
    from config import llm_critic, llm_generator
    target_llm = llm_critic if use_critic else llm_generator
    last_err = None
    
    for attempt in range(retries + 1):
        try:
            response = target_llm.invoke(prompt)
            return response.content
        except Exception as e:
            msg = str(e)
            is_infra = any(k in msg for k in [
                "503", "429", "Service is too busy", "service_unavailable",
                "timeout", "Timeout", "Connection", "rate limit", "RateLimit"
            ])
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
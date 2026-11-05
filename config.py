#全局配置与模型、状态定义
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Any, NotRequired

load_dotenv()

# ================= 0. 全局配置 =================
USE_LLM_RELEVANCE_FILTER = True   # 关闭则只走结构化信号过滤，用于对比实验
ENABLE_SEARCH_CACHE = True        # 检索结果内存 + 磁盘缓存

# 生成器模型：deepseek-chat 现已被服务端静默重定向到 deepseek-flash，
# 这里显式写清楚，保证代码 / 日志 / 论文三处模型名一致，并支持环境变量切换做对比实验。
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "deepseek-flash")

# ================= 1. 检索层配置（DeepSeek Search API） =================
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "deepseek")  # deepseek | bing（bing 仅降级/对照）
SEARCH_COUNT = 6                  # 每次请求条数（1~50）
SEARCH_NEED_CONTENT = True        # 直接要网页正文
SEARCH_TIMEOUT = 20               # 单次请求超时（秒）
USE_QUERY_EXPANSION = False       # 两段式检索降级为可选（交给 ori_question）
EVIDENCE_MIN_RERANK = 1.0         # rerank_score 门槛（0~3，仅挡明显噪声，语义判断交给 GLM）
EVIDENCE_SUMMARY_MAX = 800        # 灌给 GLM 裁判的长摘要软截断
EVIDENCE_CONTENT_MAX = 800        # 每条正文截断（实测正文平均 3000+ 字，够用即止）
EVIDENCE_MAX_CHARS = 12000        # 单次检索的证据块上限（实测 6 条 ≈ 10.3k 字，可全量保留）
ENABLE_EVIDENCE_ARCHIVE = True    # 把装配好的证据归档成 Markdown，便于论文附录与复现
SEARCH_STORE_DIR = os.getenv("SEARCH_STORE_DIR", "data/search_cache")
EVIDENCE_STORE_DIR = "data/evidence"
RUN_LOG_DIR = "logs"              # 控制台全链路日志（文本，替代终端截图）
RUN_RECORD_DIR = "data/runs"      # 结构化运行记录（JSON，论文表格的数字来源）
ENABLE_RUN_ARCHIVE = True         # 关闭则只打印不落盘，用于快速调试

# ================= 1.2 反思判定阈值与策略（论文开关） =================
# 「通过」的最低资料支撑门槛：回答必须至少引用 N 条能在检索来源清单里核验的 URL。
# 设为 0 即回到旧行为（只要求"写了【来源】"），用于对比实验。
PASS_MIN_CITED = int(os.getenv("PASS_MIN_CITED", "1"))
# 回答字数下限（原写死在 utils.content_gate 里）
PASS_MIN_CHARS = int(os.getenv("PASS_MIN_CHARS", "120"))
# 补检索（自由反思）的查询改写策略：
#   missing_term = 主题 + 首轮未覆盖对象（默认，保证与首轮 query 不同）
#   drop_lang    = 去掉 lang:zh 语言限定，扩大召回
#   same         = 沿用原 query（对照实验用，会命中同一份磁盘缓存）
RETRY_QUERY_MODE = os.getenv("RETRY_QUERY_MODE", "missing_term").strip().lower()


api_key = os.getenv("DEEPSEEK_API_KEY")
glm_api_key = os.getenv("GLM_API_KEY")

# ================= 2. 初始化双模型架构 =================
llm_generator = ChatOpenAI(
    model=GENERATOR_MODEL,
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com",
    temperature=0.1,
    timeout=30,
    max_retries=0,
)

llm_critic = ChatOpenAI(
    model="glm-5.3-flash",
    openai_api_key=glm_api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
    timeout=60,
    max_retries=0,
)

# ================= 1. 定义状态 =================
class State(TypedDict):
    topic: str
    subtasks: List[str]
    result: str
    error_type: str
    replan_scope: str
    replan_count: int
    reflection_history: List[str]
    sources: List[str]
    used_fallback: bool
    llm_fail_count: int
    evidence: NotRequired[List[Dict[str, Any]]]  # 结构化检索证据（title/url/summary/content/score）
    run_stats: NotRequired[Dict[str, Any]]       # 检索/反思的量化统计（落盘成 data/runs/*.json）
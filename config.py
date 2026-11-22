#全局配置与模型、状态定义
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Any, NotRequired

load_dotenv()


def _flag(name: str, default: bool = True) -> bool:
    """布尔开关解析：'0/false/no/off/空' 视为关闭，便于命令行做 A/B（E1~E6）。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")

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
# v2 口径的运行记录目录：v1 的 20 份记录留在 data/runs/（只读历史），新结果一律写这里，
# 两者判定语义不同，**不可混在同一张表**（见 docs/反思判定状态机-改动清单.md 第 8 节）。
RUN_RECORD_DIR = os.getenv("RUN_RECORD_DIR", "data/runs_v2")
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
# v2.2 完整性：命中确定性截断信号（结尾落在逗号/连接词、括号不配对）即判「内容不合格」。
# 只改判定，不改生成；关掉即回到 v2.1 的「只要够长且引用可核验就算通过」。
ENABLE_TRUNCATION_RULE = _flag("ENABLE_TRUNCATION_RULE", True)
# v2.2 降级救援：把「可读但缺资料」的否决改判「降级通过」，只修三态粒度，
# **不会**把否决改判为通过（二分类指标不受影响）。关掉即回到 v2.1 的单向闸门。
ENABLE_DEGRADE_RESCUE = _flag("ENABLE_DEGRADE_RESCUE", True)
# v2.2 引用率下限：0 = 不参与判定（保持 v2.1 行为）；设 0.5 即要求「引用条数 / min(来源数, TOP_K) >= 50%」。
PASS_MIN_CITATION_RATIO = float(os.getenv("PASS_MIN_CITATION_RATIO", "0") or 0)

# ================= 1.3 实验开关（论文 E1~E6 的 A/B 变量） =================
# 全部写入 data/runs_v2/<组>/ 的 JSON config 段，保证"某份记录属于哪组实验"可自证。
# E1 基线：False 时反思节点直接判「通过」并结束（不看内容、不重规划）
ENABLE_REFLECTION = _flag("ENABLE_REFLECTION", True)
# E4 消融：关掉即不执行 2.2 的第 4、5 条（子任务/比较对象覆盖率规则）
ENABLE_COVERAGE_RULE = _flag("ENABLE_COVERAGE_RULE", True)
# E5 消融：关掉即回到"同类型错误反复重规划"的旧行为
ENABLE_REPEAT_RULE = _flag("ENABLE_REPEAT_RULE", True)
# E2：关掉 2.2 第 7~11 条的确定性否决链（纯语义裁判）
DISABLE_CONTENT_GATE = _flag("DISABLE_CONTENT_GATE", False)
# E6：前 N 次 LLM 调用强制抛超时，验证双熔断是否收敛
INJECT_LLM_FAIL = int(os.getenv("INJECT_LLM_FAIL", "0") or 0)
# P1-5：引用率分母上限（原写死在 utils.content_gate 的 3）
EVIDENCE_TOP_K = int(os.getenv("EVIDENCE_TOP_K", "3") or 3)
# 实验组标签：同时决定 data/runs_v2/<组>/ 子目录与 JSON 里的 group 字段
EXP_GROUP = (os.getenv("EXP_GROUP") or "E3").strip() or "E3"

# ================= 1.35 修复路径实验（v2.3） =================
# 三条修复路径的三臂对照（设计见 docs/修复路径对比实验-设计.md）：
#   hybrid = 有确定性依据→定向修复（局部/全局），判断不清→自由反思（默认，等价 v2.2 行为）
#   fixed  = 只按错误类型做固定映射，不走自由反思（"旧做法"）
#   free   = 一律补检索+重写，不做定向修复（"纯反思"）
# 只影响"错在哪 → 走哪条修复路径"，**不改判定标准**，因此二分类指标不受影响。
REPAIR_POLICY = (os.getenv("REPAIR_POLICY", "hybrid").strip().lower() or "hybrid")
# 首轮受控缺陷注入（none / short / no_cite / truncate / one_side）：
# 只作用于 replan_count==0 的第一轮，用于让"修复路径"有样本可比较（机制归因，非外部效度）。
INJECT_FAULT = (os.getenv("INJECT_FAULT", "none").strip().lower() or "none")
# 定向修复的轮次上限；自由反思固定只有 1 次机会（由路由保证）
REPAIR_MAX_ROUNDS = int(os.getenv("REPAIR_MAX_ROUNDS", "3") or 3)
# v2.5 第二步：R2（缺的对象证据里已经有 / 过短 / 截断 / 缺引用）从"局部重检索"
# 改成"用**已有证据**整篇重写"（rewrite）。依据是 facts.evidence_gap 为空 ——
# 再检索也变不出新东西，缺的是"写"。关掉即回到 v2.4 行为（R2 → local），
# 用于消融"改答案"这一步到底值不值（rewrite 与 local 的检索次数差 1 次）。
ENABLE_REWRITE_PATH = _flag("ENABLE_REWRITE_PATH", True)
# v2.4 实验用「必答对象」（逗号分隔，如 "Python,Java"）：由 experiments/repair_matrix.py
# 从任务集的 required_objects 注入，与判卷脚本同源。
# 为什么要单独给一份：覆盖率终判如果靠"子任务/主题串逐字出现在证据里"，这些串几乎不可能
# 逐字命中（48 份旧记录里 54/100 轮都因此判"分解不合理"）。有了必答对象，判定与判卷
# 用同一份对象清单，两边的差异才只剩"谁来检查"。
# 留空 = 沿用 v2.1~v2.3 的证据侧覆盖率口径，不影响常规运行与历史记录。
REQUIRED_OBJECTS = [x.strip() for x in
                    (os.getenv("REQUIRED_OBJECTS", "") or "").split(",") if x.strip()]

# ================= 1.4 模型名（代码 / 日志 / 论文三处必须一致） =================
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "glm-5.3-flash")
# 生成温度：默认 0.1（与 v2.2 一致）；修复路径对照实验设 0，消除抖动对路径选择的影响。
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1") or 0.1)


api_key = os.getenv("DEEPSEEK_API_KEY")
glm_api_key = os.getenv("GLM_API_KEY")

# ================= 2. 初始化双模型架构 =================
llm_generator = ChatOpenAI(
    model=GENERATOR_MODEL,
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com",
    temperature=LLM_TEMPERATURE,
    timeout=30,
    max_retries=0,
)

llm_critic = ChatOpenAI(
    model=CRITIC_MODEL,
    openai_api_key=glm_api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=LLM_TEMPERATURE,
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
    repair_path: NotRequired[str]                # v2.3：本轮选定的修复路径（local/global/free；v2.5 加 rewrite）
    replan_count: int
    reflection_history: List[str]
    sources: List[str]
    used_fallback: bool
    llm_fail_count: int
    evidence: NotRequired[List[Dict[str, Any]]]  # 结构化检索证据（title/url/summary/content/score）
    run_stats: NotRequired[Dict[str, Any]]       # 检索/反思的量化统计（落盘成 data/runs/*.json）
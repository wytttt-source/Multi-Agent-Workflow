#全局配置与模型、状态定义
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, List

load_dotenv()

# ================= 0. 全局配置 =================
USE_LLM_RELEVANCE_FILTER = True   # 关闭则只走规则过滤，用于对比实验
ENABLE_SEARCH_CACHE = True        # 检索结果内存缓存

api_key = os.getenv("DEEPSEEK_API_KEY")
glm_api_key = os.getenv("GLM_API_KEY")

# ================= 2. 初始化双模型架构 =================
llm_generator = ChatOpenAI(
    model="deepseek-chat",
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
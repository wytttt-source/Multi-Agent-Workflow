#程序入口组装图并运行
import os
import sys
import time

from langgraph.graph import StateGraph, END
from config import State, ENABLE_RUN_ARCHIVE
from agents.nodes import planner_node, executor_node, reflection_node, free_reflection_node

# ================= 6. 路由函数 =================
def route_after_reflection(state: State):
    if state["error_type"] in ("通过", "降级通过"):
        return "end"
    if state["error_type"] == "系统繁忙":
        if state.get("llm_fail_count", 0) >= 3:
            print("  -> LLM 连续故障达 3 次，强制结束")
            return "end"
        print("  -> LLM 系统繁忙，触发【局部】重试")
        return "replan_local"
    if state["replan_count"] >= 3:
        print("  -> 达到最大重试次数，强制结束。")
        return "end"
    if state["error_type"] == "证据不足":
        return "free_reflection"
    if state.get("replan_scope") == "全局":
        return "replan_global"
    else:
        return "replan_local"

# ================= 7. 组装状态图 =================
workflow = StateGraph(State)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("free_reflection", free_reflection_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "reflection")

workflow.add_conditional_edges(
    "reflection",
    route_after_reflection,
    {
        "end": END,
        "free_reflection": "free_reflection",
        "replan_global": "planner",
        "replan_local": "executor",
    },
)

workflow.add_edge("free_reflection", END)
app = workflow.compile()

def _print_report(result: dict) -> None:
    print("\n" + "=" * 40)
    print("=== 最终执行结果 ===")
    print(result["result"])

    if result.get("sources"):
        print("\n=== 引用来源 ===")
        for s in result["sources"]:
            if " | " in s:
                title, url = s.split(" | ", 1)
                print(f"- {title}\n  {url}")
            else:
                print(f"- {s}")

    if result.get("evidence"):
        print(f"\n=== 检索证据：{len(result['evidence'])} 条（原始响应见 data/search_cache/）===")
        for i, r in enumerate(result["evidence"], 1):
            score = r.get("rerank_score")
            score_text = f"rerank={score:.2f}" if isinstance(score, (int, float)) else "rerank=-"
            print(f"{i}. [{r.get('provider', '?')}/{score_text}] {r.get('title', '')[:50]}")
            print(f"   {r.get('url', '')}")

    if result.get("subtasks"):
        print(f"\n=== 子任务拆解 ===\n{result['subtasks']}")

    # 降级通过：回答可读但无外部证据支撑，不计入合格样本
    if result.get("error_type") == "降级通过":
        print("\n[注意] 反思判定为「降级通过」：回答可读但无外部证据支撑，"
              "不计入合格样本，详见 data/runs/*.json 的 reflection.verdict。")

    # 无证据兜底：保留原有提示（与上面的降级通过不冲突，二者可同时出现）
    if result.get("used_fallback"):
        print("\n[注意] 本次结果未获取到外部检索证据，基于模型内部知识生成。")

    print(f"\n总共进行了 {result['replan_count']} 次重规划")

if __name__ == "__main__":
    print("=== 欢迎使用多智能体工作流系统 ===")
    if len(sys.argv) > 1:
        user_topic = " ".join(sys.argv[1:]).strip()
        print(f"检测到命令行主题：{user_topic}")
    else:
        user_topic = input("请输入你想检索并生成摘要的主题（例如：量子计算、大语言模型）：\n> ")
    if not user_topic.strip():
        user_topic = "LangGraph多智能体"
        print(f"检测到空输入，默认使用主题：{user_topic}")

    initial_state = {
        "topic": user_topic.strip(),
        "subtasks": [],
        "replan_count": 0,
        "reflection_history": [],
        "sources": [],
        "evidence": [],
        "run_stats": {},
        "used_fallback": False,
        "llm_fail_count": 0,
    }

    if ENABLE_RUN_ARCHIVE:
        # 全链路日志与结构化运行记录：以文本存档替代终端截图
        from run_store import run_log, save_run_record
        t0 = time.perf_counter()
        with run_log(user_topic.strip()) as log_path:
            print(f"\n=== 开始运行工作流，当前主题：{user_topic} ===")
            result = app.invoke(initial_state)
            elapsed = time.perf_counter() - t0
            _print_report(result)
            save_run_record(user_topic.strip(), result, log_path, elapsed)
            print(f"  -> [日志] {os.path.relpath(log_path)}")
    else:
        print(f"\n=== 开始运行工作流，当前主题：{user_topic} ===")
        result = app.invoke(initial_state)
        _print_report(result)
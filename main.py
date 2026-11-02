#程序入口组装图并运行
from langgraph.graph import StateGraph, END
from config import State
from agents.nodes import planner_node, executor_node, reflection_node, free_reflection_node

# ================= 6. 路由函数 =================
def route_after_reflection(state: State):
    if state["error_type"] == "通过":
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

if __name__ == "__main__":
    print("=== 欢迎使用多智能体工作流系统 ===")
    user_topic = input("请输入你想检索并生成摘要的主题（例如：量子计算、大语言模型）：\n> ")
    if not user_topic.strip():
        user_topic = "LangGraph多智能体"
        print(f"检测到空输入，默认使用主题：{user_topic}")

    print(f"\n=== 开始运行工作流，当前主题：{user_topic} ===")

    initial_state = {
        "topic": user_topic.strip(),
        "subtasks": [],
        "replan_count": 0,
        "reflection_history": [],
        "sources": [],
        "used_fallback": False,
        "llm_fail_count": 0,
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 40)
    print("=== 最终执行结果 ===")
    print(result["result"])

    if result.get("sources"):
        print("\n=== 引用来源 ===")
        for s in result["sources"]:
            print(f"- {s}")

    if result.get("subtasks"):
        print(f"\n=== 子任务拆解 ===\n{result['subtasks']}")

    if result.get("used_fallback"):
        print("\n[注意] 本次结果未获取到外部检索证据，基于模型内部知识生成。")

    print(f"\n总共进行了 {result['replan_count']} 次重规划")
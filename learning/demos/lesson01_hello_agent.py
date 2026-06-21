"""
第 1 课：Hello World —— 5 分钟创建你的第一个 Agent
==================================================

本课目标：
  1. 用 create_agent() 创建最小 Agent
  2. 定义工具（Tool）
  3. 用 agent.invoke() 与 Agent 对话
  4. 理解 Agent 的输入输出格式

预计时间：45 分钟
"""

# ============================================================================
# Step 0: 环境准备
# ============================================================================
#
# 安装依赖（在终端执行）：
#
#   pip install -U langchain langchain-openai
#
#   或者用 uv：
#
#   uv add langchain langchain-openai
#
# 设置 API Key（在终端执行，选你使用的提供商）：
#
#   # OpenAI
#   export OPENAI_API_KEY="your-api-key"
#
#   # Anthropic (Claude)
#   export ANTHROPIC_API_KEY="your-api-key"
#
#   # 其他提供商参考：https://docs.langchain.com/oss/python/langchain/models


# ============================================================================
# Step 1: 最简 Agent（5 行代码）
# ============================================================================

from langchain.agents import create_agent


# 定义一个工具 —— 普通 Python 函数
# 关键：docstring 会被 LLM 当作"使用说明书"
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# 创建 Agent：三要素 = model + tools + system_prompt
agent = create_agent(
    model="openai:gpt-4o",           # 模型（格式："provider:model_name"）
    tools=[get_weather],             # 工具列表
    system_prompt="You are a helpful assistant",  # 系统提示
)

# 调用 Agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)

# 看结果
print(result["messages"][-1].content_blocks)


# ============================================================================
# Step 2: 理解发生了什么
# ============================================================================

print("\n" + "=" * 60)
print("拆解 Agent 的完整响应")
print("=" * 60)

# 响应中包含对话的完整历史
for i, msg in enumerate(result["messages"]):
    print(f"\n--- 消息 {i}: {msg.type} ---")
    if hasattr(msg, "content_blocks"):
        print(f"  content_blocks: {msg.content_blocks}")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"  tool_calls: {msg.tool_calls}")
    print(f"  content (text): {msg.content}")


# ============================================================================
# Step 3: 试试不用工具的问题
# ============================================================================

print("\n" + "=" * 60)
print("对话 2：不需要工具的问题")
print("=" * 60)

result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
)

# 这个问题不需要调工具，LLM 直接回答
print(result2["messages"][-1].content_blocks)


# ============================================================================
# Step 4: 加第二个工具 —— 感受 Agent 的"决策"能力
# ============================================================================

def get_time(city: str) -> str:
    """Get the current local time for a given city."""
    from datetime import datetime
    # 模拟：实际应该用时区数据库
    return f"The current time in {city} is {datetime.now().strftime('%H:%M:%S')}"


agent_two_tools = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather, get_time],
    system_prompt="You are a helpful assistant. Always use the tools when available.",
)

result3 = agent_two_tools.invoke(
    {"messages": [{"role": "user", "content": "What time is it in Tokyo, and how is the weather there?"}]}
)

print("\n" + "=" * 60)
print("对话 3：多工具调用")
print("=" * 60)
print(result3["messages"][-1].content_blocks)


# ============================================================================
# Step 5: 改用 Anthropic Claude（如果你有 key）
# ============================================================================

# agent_claude = create_agent(
#     model="anthropic:claude-sonnet-4-6",
#     tools=[get_weather],
#     system_prompt="You are a helpful assistant",
# )
#
# result4 = agent_claude.invoke(
#     {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]}
# )
# print(result4["messages"][-1].content_blocks)


# ============================================================================
# 核心概念回顾
# ============================================================================

"""
┌─────────────────────────────────────────────────┐
│               create_agent() 三要素              │
│                                                  │
│  model ──── 大脑：决定"下一步做什么"             │
│  tools ──── 手脚：执行具体操作                   │
│  system_prompt ── 性格：设定角色和行为边界       │
│                                                  │
│  Agent = Model + Tools + Prompt                  │
│  (Agent = 大脑  + 手脚  + 性格)                  │
└─────────────────────────────────────────────────┘

调用流程：
  用户输入 → Agent → LLM 判断是需要调工具还是直接答
                    │
                    ├─ 需要工具 → 调用工具 → 拿到结果 → 再给 LLM 判断
                    │                                           │
                    └─ 不需要工具 ──────────────────────────────┘
                                                              ↓
                                                       返回最终回答

输入格式：{"messages": [{"role": "user", "content": "..."}]}
输出格式：result["messages"] 是完整的对话历史
         result["messages"][-1] 是最后一条消息（通常是 AI 的回复）
"""

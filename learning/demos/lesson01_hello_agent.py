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
#   pip install -U langchain langchain-deepseek python-dotenv
#
#   或者用 uv：
#
#   uv add langchain langchain-deepseek python-dotenv
#
# 在 learning/ 目录下创建 .env 文件：
#
#   API_KEY=sk-xxxxx
#   BASE_URL=https://api.deepseek.com
#   MODEL=deepseek-v4-flash


# ============================================================================
# Step 1: 最简 Agent（5 行代码）
# ============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

# 自动加载 learning/ 目录下的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# 从 .env 读取配置
MODEL = os.environ["MODEL"]
API_KEY = os.environ["API_KEY"]
BASE_URL = os.environ["BASE_URL"]

# 初始化模型（DeepSeek 兼容 OpenAI API）
model = init_chat_model(
    model=MODEL,
    model_provider="deepseek",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
)


# 定义一个工具 —— 普通 Python 函数
# 关键：docstring 会被 LLM 当作"使用说明书"
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# 创建 Agent：三要素 = model + tools + system_prompt
agent = create_agent(
    model=model,                     # 使用 .env 配置的模型
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
    model=model,
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
# Step 5: 换模型只需改 .env
# ============================================================================
#
# 如果想换 OpenAI，只需修改 learning/.env：
#
#   API_KEY=sk-xxxxx
#   BASE_URL=https://api.openai.com/v1
#   MODEL=gpt-4o
#
# 然后在代码中改 model_provider="openai" 即可。
# 所有 Agent 逻辑代码不需要任何改动。


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

# 第 1 课：Hello World —— 创建你的第一个 Agent

> **核心目标**：不纠结原理，5 分钟内跑通第一个 Agent。用起来，再理解。

---

## 1. 本课目标

- 亲手用 `create_agent()` 创建一个能工作的 AI Agent
- 理解 Agent 的三要素：Model（大脑）、Tools（手脚）、Prompt（性格）
- 看懂输入输出格式：Agent 怎么收消息、怎么回消息
- 观察 Agent 的"决策"过程：什么时候直接答，什么时候调工具

---

## 2. 核心概念（先看图，再看代码）

```
┌──────────────────────────────────────────────┐
│                 Agent 循环                    │
│                                              │
│  用户: "旧金山天气怎么样？"                    │
│         │                                    │
│         ▼                                    │
│   ┌──────────┐    需要工具？                  │
│   │   LLM    │ ──────────────┐               │
│   │  (大脑)  │               │ 是            │
│   └──────────┘               ▼               │
│         │              get_weather("SF")      │
│         │ 否               │                 │
│         │                  ▼                 │
│         │              "旧金山晴天！"          │
│         │                  │                 │
│         ▼                  ▼                 │
│   ┌──────────┐    再给 LLM 判断               │
│   │ 直接回答  │ ◄── 够了就输出                 │
│   └──────────┘                               │
│         │                                    │
│         ▼                                    │
│  "旧金山天气晴朗，适合出游！"                   │
└──────────────────────────────────────────────┘
```

**一句话总结**：Agent 就是 LLM 在循环中决策——要么调工具，要么给答案。

---

## 3. 动手实验（跟着做）

### 准备工作

```bash
# 1. 安装依赖
pip install -U langchain langchain-openai

# 2. 设置 API Key（任选一个你用得到的）
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

> **没有 API Key？** 可以用本地模型：安装 [Ollama](https://ollama.com)，然后 `ollama pull llama3`，模型参数写 `"ollama:llama3"`。但注意部分本地模型对 tool calling 支持较差。

### 实验 1：最小 Agent（必做）

```python
from langchain.agents import create_agent

# ① 定义工具 —— python 函数 + docstring
def get_weather(city: str) -> str:
    """Get weather for a given city."""       # ← docstring 是工具的"说明书"！
    return f"It's always sunny in {city}!"

# ② 创建 Agent
agent = create_agent(
    model="openai:gpt-4o",                    # 大脑
    tools=[get_weather],                      # 手脚
    system_prompt="You are a helpful assistant",  # 性格
)

# ③ 对话
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})

# ④ 看结果
print(result["messages"][-1].content_blocks)
```

**跑通后停下来想一想**：
- Agent 真的调用了 `get_weather("San Francisco")` 吗？（看终端有没有执行）
- 如果问"法国的首都是哪"，Agent 还会调工具吗？

### 实验 2：观察完整响应（必做）

```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
})

# 遍历所有消息，看 Agent "思考"的每一步
for i, msg in enumerate(result["messages"]):
    print(f"\n消息 {i} — 类型: {msg.type}")
    print(f"  内容: {msg.content}")
    if getattr(msg, "tool_calls", None):
        print(f"  工具调用: {msg.tool_calls}")
```

**你应该看到**：
1. HumanMessage（你的问题）
2. AIMessage（LLM 决定调 `get_weather`，携带 `tool_calls`）
3. ToolMessage（工具返回的结果）
4. AIMessage（LLM 根据工具结果给出最终回答）

这就是 **Agent 循环的完整轨迹**。

### 实验 3：加第二个工具（挑战）

```python
def get_time(city: str) -> str:
    """Get the current local time for a given city."""
    from datetime import datetime
    return f"Current time in {city}: {datetime.now().strftime('%H:%M:%S')}"

agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather, get_time],  # ← 两个工具
    system_prompt="You are a helpful assistant.",
)

# 一个需要两个工具的问题
result = agent.invoke({
    "messages": [{"role": "user",
                  "content": "What time is it in Tokyo, and how's the weather?"}]
})
print(result["messages"][-1].content_blocks)
```

**观察**：Agent 能自己决定先调哪个、后调哪个，或者一次调两个。

---

## 4. 理解总结

### create_agent() 三个参数

| 参数 | 比喻 | 作用 |
|------|------|------|
| `model` | 大脑 | 做决策：回答还是调工具？调哪个工具？ |
| `tools` | 手脚 | 干活：查天气、算数学、搜网页... |
| `system_prompt` | 性格 | 定调：是严肃的助手？还是幽默的朋友？ |

### 输入输出格式

```
输入 ▼
{"messages": [{"role": "user", "content": "..."}]}

输出 ▼
result["messages"]        # 完整对话历史（list）
result["messages"][-1]    # 最后一条消息（AI 的最终回答）
msg.content_blocks        # 消息的结构化内容
msg.tool_calls            # 如果 LLM 决定调工具，这里会有调用信息
```

### 工具（Tool）的本质

```
一个普通 Python 函数
    +
docstring（教程书）
    ↓
LLM 通过 docstring 理解这个工具能干什么
    ↓
LLM 决定要不要调用它、传什么参数
```

---

## 5. 常见问题

**Q: 运行报错 `module not found`？**
```bash
pip install -U langchain langchain-openai
```

**Q: API Key 错误？**
确认环境变量已设：`echo $OPENAI_API_KEY`

**Q: 本地模型（Ollama）对 tool calling 支持差怎么办？**
先用 OpenAI 或 Anthropic 的云端模型学习概念，本地模型留到后面优化。

**Q: 每次 `invoke` 是独立的吗？**
是的。每次 `invoke` 都是一次全新的对话，Agent 不记得上次说了什么。第 3 课会讲消息历史。

---

## 6. 课后检查

完成这三个，你就可以进入第 2 课了：

- [ ] 跑通了 `create_agent()` + 自定义工具
- [ ] 能解释 `result["messages"]` 里每一条消息的含义
- [ ] 知道自己想用哪个模型提供商继续学（OpenAI / Anthropic / Ollama / ...）

---

## 7. 延伸阅读（可选）

- [官方 Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart) — 和本课配套
- [Philosophy](https://docs.langchain.com/oss/python/langchain/philosophy) — LangChain 的设计哲学（10 分钟读完）
- [Agent 官方文档](https://docs.langchain.com/oss/python/langchain/agents) — 第 4–6 课的预习材料

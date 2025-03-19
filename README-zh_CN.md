<img src="https://github.com/user-attachments/assets/10a3a9ca-1f39-410c-ac48-a7365de589d9" >
<br>
<br>
<a name="readme-top"></a>

<div align="center">


</div>


  <p>
    <a href="https://discord.gg/dNKGm4dfnR">
    <img src="https://img.shields.io/badge/Discord-Join-7289DA?logo=discord&logoColor=white">
    </a>
    <a href="https://twitter.com/upsonicai">
    <img src="https://img.shields.io/twitter/follow/upsonicai?style=social">
    </a>
    <a href="https://trendshift.io/repositories/10584" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10584" alt="unclecode%2Fcrawl4ai | Trendshift" style="width: 100px; height: 20px;"     
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg" alt="Made_with_python">
    </a>
    <img src="https://static.pepy.tech/personalized-badge/upsonic?period=total&units=international_system&left_color=grey&right_color=blue&left_text=PyPI%20Downloads" alt="pypi_downloads">
  </p>

[ENGLISH](README.md) | **简体中文**

# 简介
Upsonic 是一个专注于可靠性的框架，为实际应用而设计。通过先进的可靠性功能，包括验证层、三角架构、验证代理和输出评估系统，它使您的组织能够实现可信赖的代理工作流程。

# 为什么选择 Upsonic？
Upsonic 是一个新一代框架，通过解决三个关键挑战，使代理准备好用于生产环境：

1- **可靠性**：其他框架需要专业知识和复杂编码来实现可靠性功能，而 Upsonic 提供易于激活的可靠性层，不会影响功能。

2- **模型上下文协议(MCP)**：MCP 允许您利用官方和第三方开发的各种功能工具，无需从头构建自定义工具。

3- **集成浏览器使用和计算机使用**：直接使用和部署可在非 API 系统上工作的代理。

4- **安全运行时**：为代理提供隔离的运行环境。

![sdk-server](https://github.com/user-attachments/assets/1b276199-ae60-4221-b8e6-b266443a3641)

<br>

## 📊 可靠性层

LLM 输出可靠性至关重要，特别是对于数值运算和操作执行。Upsonic 通过多层可靠性系统解决这一问题，启用控制代理和验证回合以确保输出准确性。

**验证代理**：验证输出、任务和格式 - 检测不一致、数值错误和幻觉

**编辑代理**：使用验证反馈修改和完善输出，直到满足质量标准

**回合**：通过评分验证周期实施迭代质量改进

**循环**：通过在关键可靠性检查点的受控反馈循环确保准确性


Upsonic 是一个以可靠性为重点的框架。表格中的结果是使用小型数据集生成的。它们显示了 JSON 键转换的成功率。在测试过程中没有对框架进行硬编码更改；只激活和运行了每个框架的现有功能。测试中使用了 GPT-4o。

每个部分进行了 10 次转换。数字表示错误计数。因此，如果显示 7，则表示 10 次中有 7 次**不正确**。该表是基于初步结果创建的。我们正在扩展数据集。在创建更大的测试集后，测试将变得更加可靠。可靠性基准[仓库](https://github.com/Upsonic/Reliability-Benchmark)

| 名称     | 可靠性得分 % | ASIN 代码 | HS 代码 | CIS 代码 | 营销 URL | 使用 URL | 保修时间 | 政策链接 | 政策描述 |
|-----------|--------------------|-----------|---------|----------|---------------|-----------|---------------|-------------|----------------|
 **Upsonic**   |**99.3**      |0         |1       |0        |0             |0         |0             |0           |0                   |
| **CrewAI**    |**87.5**       |0         |3       |2        |1             |1         |0             |1           |2                   |
| **Langgraph** |**6.3**      |10        |10      |7        |10            |8         |10            |10          |10                  |


```python
class ReliabilityLayer:
  prevent_hallucination = 10

agent = Agent("Coder", reliability_layer=ReliabilityLayer, model="openai/gpt4o")
```

<br>


**主要特点：**

- **生产就绪的可扩展性**：使用 Docker 在 AWS、GCP 或本地无缝部署。
- **以任务为中心的设计**：专注于实用任务执行，选项包括：
    - 通过 LLM 调用的基本任务。
    - 使用 V1 代理的高级任务。
    - 使用具有 MCP 集成的 V2 代理的复杂自动化。
- **MCP 服务器支持**：利用多客户端处理实现高性能任务。
- **工具调用服务器**：具有稳健服务器 API 交互的异常安全工具管理。
- **计算机使用集成**：使用 Anthropic 的"计算机使用"功能执行类人任务。
- **轻松添加工具**：您可以通过单行代码添加自定义工具和 MCP 工具。
<br>

# 📙 文档

您可以访问我们的文档：[docs.upsonic.ai](https://docs.upsonic.ai/)。所有概念和示例都可在那里获得。

<br>

# 🛠️ 入门指南

### 先决条件

- Python 3.10 或更高版本
- 访问 OpenAI 或 Anthropic API 密钥（支持 Azure 和 Bedrock）

## 安装

```bash
pip install upsonic

```



# 基本示例

设置您的 OPENAI_API_KEY

```console
export OPENAI_API_KEY=sk-***
```

启动代理

```python
from upsonic import Task, Agent

task = Task("Who developed you?")

agent = Agent("Coder")

agent.print_do(task)
```

<br>
<br>

## 通过 MCP 集成工具

Upsonic 官方支持[模型上下文协议 (MCP)](https://github.com/modelcontextprotocol/servers)和自定义工具。您可以在 [glama](https://glama.ai/mcp/servers) 或 [mcprun](https://mcp.run) 上使用数百个 MCP 服务器。我们还支持类中的 Python 函数作为工具。您可以轻松生成您的集成。

```python
from upsonic import Agent, Task, ObjectResponse

# 定义 Fetch MCP 配置
class FetchMCP:
    command = "uvx"
    args = ["mcp-server-fetch"]

# 为网页内容创建响应格式
class WebContent(ObjectResponse):
    title: str
    content: str
    summary: str
    word_count: int

# 初始化代理
web_agent = Agent(
    "Web Content Analyzer",
    model="openai/gpt-4o",  # 您可以使用其他模型
)

# 创建分析网页的任务
task = Task(
    description="Fetch and analyze the content from url. Extract the main content, title, and create a brief summary.",
    context=["https://upsonic.ai"],
    tools=[FetchMCP],
    response_format=WebContent
)
    
# 使用
web_agent.print_do(task)
print(result.title)
print(result.summary)

```
<br>

## 多任务代理示例

通过我们的自动任务分配机制有效地在代理之间分配任务。该工具根据代理和任务之间的关系匹配任务，确保在代理和任务之间进行协作解决问题。输出对于跨应用部署 AI 代理或作为服务部署 AI 代理至关重要。Upsonic 使用 Pydantic BaseClass 为任务定义结构化输出，允许开发人员为其 AI 代理任务指定确切的响应格式。

```python
from upsonic import Agent, Task, MultiAgent, ObjectResponse
from upsonic.tools import Search
from typing import List

# 目标公司和我们的公司
our_company = "https://redis.io/"
targeted_url = "https://upsonic.ai/"


# 响应格式
class CompanyResearch(ObjectResponse):
   industry: str
   product_focus: str
   company_values: List[str]
   recent_news: List[str]

class Mail(ObjectResponse):
   subject: str
   content: str


# 创建代理
researcher = Agent(
   "Company Researcher",
   company_url=our_company
)

strategist = Agent(
   "Outreach Strategist", 
   company_url=our_company
)


# 创建任务并连接
company_task = Task(
   "Research company website and analyze key information",

   context=[targeted_url],
   tools=[Search],
   response_format=CompanyResearch
)

position_task = Task(
   "Analyze Senior Developer position context and requirements",
   context=[company_task, targeted_url],
)

message_task = Task(
   "Create personalized outreach message using research",
   context=[company_task, position_task, targeted_url],
   response_format=Mail
)


# 在代理上运行任务
results = MultiAgent.do(
   [researcher, strategist],
   [company_task, position_task, message_task]
)


# 打印结果
print(f"公司行业: {company_task.response.industry}")
print(f"公司重点: {company_task.response.product_focus}")
print(f"公司价值观: {company_task.response.company_values}")
print(f"公司近期新闻: {company_task.response.recent_news}")
print(f"职位分析: {position_task.response}")
print(f"外联信息主题: {message_task.response.subject}")
print(f"外联信息内容: {message_task.response.content}")

```

## 直接 LLM 调用

直接 LLM 调用为简单任务提供了更快、更经济的解决方案。在 Upsonic 中，您可以在没有任何抽象层的情况下调用模型提供商，并组织结构化输出。您还可以在 LLM 调用中使用工具。

```python
from upsonic import Task, Direct

direct = Direct(model="openai/gpt-4o")

task = Task("Where can I use agents in real life?")

direct.print_do(task)

```

<br>

## 指南
您可以[查看许多示例](https://github.com/Upsonic/cookbook)，了解如何使用 MCP 工具和浏览器与 Upsonic 构建代理。

<br>

## 遥测

我们使用匿名遥测来收集使用数据。我们这样做是为了将我们的开发重点放在更准确的点上。您可以通过将 UPSONIC_TELEMETRY 环境变量设置为 false 来禁用它。

```python
import os
os.environ["UPSONIC_TELEMETRY"] = "False"
```
<br>
<br>
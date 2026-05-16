# 大模型 Agent 算法工程师学习计划（推荐算法工程师转型版）

> 适用对象：资深推荐算法工程师，希望转型为大模型 Agent 算法/工程方向。  
> 更新时间：2026-05-16  
> 学习目标：不是泛泛学习 LLM，而是建立 **Agent 算法 + 工程系统 + 评估闭环 + 作品集** 的完整知识体系。

---

## 0. 你的转型优势与补齐方向

### 0.1 推荐算法能力如何迁移到 Agent

| 推荐算法经验 | Agent 方向对应能力 | 转型切入点 |
|---|---|---|
| 召回、向量检索、ANN、重排 | RAG、长期记忆、上下文选择、工具检索 | Agentic RAG、Memory Retrieval、Tool Retrieval |
| 排序、多目标优化、学习排序 | Tool routing、action ranking、plan selection | 多工具选择、Planner/Executor/Critic 排序 |
| Bandit、RL、Reward Modeling | Agent 自我改进、偏好优化、任务成功率优化 | trajectory reward、reflection、online eval |
| 用户画像、Session 建模 | Context engineering、personal memory、state management | 多轮状态、用户偏好记忆、个性化 Agent |
| A/B 实验、指标体系 | Agent eval、trajectory eval、failure diagnosis | 成功率、引用准确率、成本、延迟、安全 |
| 推荐系统工程化 | Agent runtime、observability、guardrails、回放 | LangGraph / Agents SDK / MCP / tracing |

### 0.2 你最该补的能力

1. **LLM 基础与推理边界**：Transformer、instruction tuning、RLHF/DPO、解码、上下文窗口。
2. **Agent 算法范式**：ReAct、planning、tool use、reflection、memory、multi-agent。
3. **Context Engineering**：如何动态装载文件、工具结果、用户状态、历史轨迹，而不是把所有信息塞进 prompt。
4. **Agent 工程化**：状态机、工具协议、沙箱、权限、人审、观测、回放。
5. **Agent Evaluation**：任务成功率、轨迹质量、工具调用质量、安全、成本/延迟、回归测试。

---

## 1. 官方技术博客与设计思想文章

这一部分优先读官方或一手来源。建议按“设计思想 → 工程实践 → 安全治理 → 框架协议”的顺序阅读。

---

### 1.1 OpenAI：Agent Runtime、Responses API、Codex、AgentKit、安全

#### P0：必须读

| 文章 | 重点 | 为什么适合你 |
|---|---|---|
| [A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/) | 如何识别 agent use case、设计 orchestration、部署安全 agent | 建立产品/工程视角，避免只做 demo |
| [New tools for building agents](https://openai.com/index/new-tools-for-building-agents/) | Responses API、Agents SDK、内置工具、tracing | 理解 OpenAI agent runtime 的核心抽象 |
| [Agents SDK docs](https://developers.openai.com/api/docs/guides/agents) | tools、handoffs、sessions、guardrails、tracing | 实战框架入口 |
| [OpenAI Building Agents learning track](https://developers.openai.com/tracks/building-agents) | 官方 agent 学习路径 | 可以作为代码实操主线 |
| [OpenAI Cookbook](https://developers.openai.com/cookbook) | prompt、agents、RAG、evals、vision、tool use 示例 | 快速补齐工程范式 |

#### P1：Agent 平台与长任务

| 文章 | 重点 | 读法 |
|---|---|---|
| [New tools and features in the Responses API](https://openai.com/index/new-tools-and-features-in-the-responses-api/) | MCP、image generation、Code Interpreter、file search 等内置工具 | 看 OpenAI 如何把 agent 能力沉淀为 API primitive |
| [Introducing AgentKit](https://openai.com/index/introducing-agentkit/) | Agent Builder、Connector Registry、ChatKit、Evals | 关注 agent 从代码到低代码/平台化的演进 |
| [Introducing Codex](https://openai.com/index/introducing-codex/) | 云端软件工程 agent、沙箱、并行任务 | 学习 coding agent 的产品化形态 |
| [Introducing the Codex app](https://openai.com/index/introducing-the-codex-app/) | 多 agent 调度、长任务、跨项目编排 | 思考“人如何管理 agent 队列” |
| [The next evolution of the Agents SDK](https://openai.com/index/the-next-evolution-of-the-agents-sdk/) | 文件检查、命令执行、代码编辑、长任务沙箱 | 重点看 sandbox + long-horizon work |
| [Harness engineering](https://openai.com/index/harness-engineering/) | 从写代码转向设计环境、指定意图、构建反馈回路 | 非常适合算法工程师理解“Agent 系统工程”的本质 |

#### P1：安全、治理、真实部署

| 文章 | 重点 | 读法 |
|---|---|---|
| [Designing AI agents to resist prompt injection](https://openai.com/index/designing-agents-to-resist-prompt-injection/) | prompt injection 威胁模型与 agent 防护 | 建议结合 RAG/浏览器/邮箱 agent 场景做安全设计 |
| [How we monitor internal coding agents for misalignment](https://openai.com/index/how-we-monitor-internal-coding-agents-misalignment/) | 用监控识别 agent misalignment | 建立 agent 线上监控和审计意识 |
| [Building a safe, effective sandbox to enable Codex on Windows](https://openai.com/index/building-codex-windows-sandbox/) | 编程 agent 的安全沙箱实现思路 | 学 coding agent 必读 |
| [Building Governed AI Agents - Cookbook](https://developers.openai.com/cookbook/examples/partners/agentic_governance_guide/agentic_governance_cookbook) | 企业 agent governance、scaffolding、权限边界 | 作品集可复用其中的安全架构 |
| [Self-Evolving Agents Cookbook](https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining) | feedback、meta prompting、eval 驱动的 agent 迭代 | 连接推荐算法中的反馈闭环 |

---

### 1.2 Anthropic / Claude：Effective Agents、Context Engineering、MCP、Claude Code

#### P0：必须读

| 文章 | 重点 | 为什么重要 |
|---|---|---|
| [Building Effective AI Agents](https://www.anthropic.com/research/building-effective-agents) | workflows vs agents、routing、parallelization、orchestrator-workers、evaluator-optimizer | 这是 Agent 设计思想最重要的入门文章之一 |
| [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | 工具命名、schema、description、eval、用 Claude 优化工具 | Tool design 是 agent 成败关键 |
| [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | just-in-time context、动态加载、轻量引用 | 和推荐系统的召回/排序非常相似 |
| [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) | 多 agent research 架构、并行搜索、协调成本 | 学 multi-agent 的工程取舍 |
| [Best practices for Claude Code](https://www.anthropic.com/engineering/claude-code-best-practices) | 编程 agent 使用模式、约束、上下文组织 | 做 coding agent 项目必读 |

#### P1：Agent SDK、长任务、技能与安全

| 文章 | 重点 | 读法 |
|---|---|---|
| [Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) | Claude Agent SDK、基于 Claude Code 的 agent 构建 | 对比 OpenAI Agents SDK 和 LangGraph |
| [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | 跨上下文窗口的长任务 harness、initializer/coding agent | 关注长任务如何续接、如何沉淀 artifacts |
| [Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) | Skills = 指令、脚本、资源文件夹，按需加载 | 可设计自己的 skill library |
| [Introducing Agent Skills](https://www.anthropic.com/news/skills) | Agent Skills 产品形态 | 对应“可复用技能库”方向 |
| [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) | 用代码执行减少大量 MCP 工具带来的上下文开销 | 对工具数量多的企业 agent 很关键 |
| [Scaling Managed Agents](https://www.anthropic.com/engineering/managed-agents) | managed agents、brain/harness 解耦 | 关注随着模型变强，工程假设如何变化 |
| [Our framework for developing safe and trustworthy agents](https://www.anthropic.com/news/our-framework-for-developing-safe-and-trustworthy-agents) | 安全可信 agent 框架 | 构建权限、审计、人审方案 |
| [Mitigating prompt injection risks in browser use](https://anthropic.com/research/prompt-injection-defenses) | 浏览器/网页 agent 的注入防御 | 结合 Web Agent 项目阅读 |

#### P1：Claude 官方文档与课程

| 资源 | 重点 |
|---|---|
| [Claude Tool Use docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) | tool use、parallel tool use、strict tool use |
| [Claude Prompt Engineering docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) | prompt、eval、latency、guardrails |
| [Claude MCP Connector docs](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector) | 直接连接 MCP server、allowlist/denylist、OAuth |
| [Claude Code MCP docs](https://docs.anthropic.com/en/docs/claude-code/mcp) | 把 Claude Code 连接到工具、数据库、API |
| [Anthropic Courses](https://docs.anthropic.com/en/docs/resources/courses) | MCP、sub-agents、Claude Code 等课程 |
| [Claude Cookbook](https://docs.anthropic.com/en/docs/resources/cookbook) | memory、context management、observability、agent patterns |

---

### 1.3 Google：ADK、A2A、多 Agent、Agent Platform

| 文章/文档 | 重点 | 读法 |
|---|---|---|
| [Agent Development Kit: build multi-agent applications](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/) | Google ADK、code-first、多 agent | 对比 LangGraph/OpenAI SDK |
| [ADK docs: Tools and Integrations](https://google.github.io/adk-docs/tools/) | ADK 工具系统 | 学会定义工具和集成 |
| [ADK docs: Multi-Agent Systems](https://google.github.io/adk-docs/agents/multi-agents/) | 多 agent 组合 | 对应 planner-worker/critic 架构 |
| [ADK docs: Custom Agents](https://google.github.io/adk-docs/agents/custom-agents/) | 自定义 orchestration 逻辑 | 进阶读 |
| [Developer's guide to multi-agent patterns in ADK](https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/) | 8 种多 agent 设计模式 | 可作为 multi-agent 项目设计模板 |
| [Announcing Agent2Agent Protocol](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) | A2A：agent 间互操作协议 | 和 MCP 做对比：MCP 偏工具/上下文，A2A 偏 agent-to-agent |
| [A2A GitHub](https://github.com/a2aproject/A2A) | A2A 协议与实现 | 适合做 agent interoperability 实验 |
| [Building agents with ADK and Interactions API](https://developers.googleblog.com/building-agents-with-the-adk-and-the-new-interactions-api/) | stateful interactions、remote A2A agents、background execution | 关注长任务和远程 agent 协作 |
| [Gemini Enterprise Agent Platform](https://cloud.google.com/products/gemini-enterprise-agent-platform) | build、scale、govern、optimize agents | 企业级 agent 平台视角 |
| [Scale your agents](https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale) | tracing、logging、monitoring、IAM agent identity、gateway | 线上化必读 |
| [Enhanced Tool Governance in Vertex AI Agent Builder](https://cloud.google.com/blog/products/ai-machine-learning/new-enhanced-tool-governance-in-vertex-ai-agent-builder) | tool governance、multimodal I/O、A2A/ADK | 和安全治理结合 |
| [Agent Payments Protocol AP2](https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol) | agent 发起支付/交易的协议思路 | 了解高风险 action 的授权与证明机制 |

---

### 1.4 标准与协议：MCP、AGENTS.md、AAIF

| 资源 | 重点 |
|---|---|
| [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) | MCP 官方协议；核心 primitives：resources、prompts、tools |
| [MCP Tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) | 工具暴露、schema、metadata |
| [MCP Resources](https://modelcontextprotocol.io/specification/2025-06-18/server/resources) | 文件、数据库 schema、应用上下文等资源暴露 |
| [MCP Prompts](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) | 可发现、可参数化的 prompt/workflow 模板 |
| [MCP Tasks](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks) | 实验性任务抽象 |
| [AGENTS.md](https://agents.md/) | 给 coding agents 的项目级说明文件，相当于 agent 版 README |
| [Linux Foundation: Agentic AI Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation) | OpenAI、Anthropic、Block 等共建 agent 开放基础设施 |
| [OpenAI: Agentic AI Foundation](https://openai.com/index/agentic-ai-foundation/) | OpenAI 捐赠 AGENTS.md |
| [Anthropic: donating MCP](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) | Anthropic 捐赠 MCP |

你需要形成的判断：

- **MCP**：更像“Agent ↔ 外部工具/数据/上下文”的标准接口。
- **A2A**：更像“Agent ↔ Agent”的协作协议。
- **AGENTS.md / Skills**：更像“项目/任务/组织知识如何被 agent 按需读取”的知识封装方式。
- **Agent Platform / AgentKit / Managed Agents**：说明行业正在从“脚本式 agent demo”走向“可部署、可治理、可观测、可协作的 agent runtime”。

---

## 2. 框架与工程资源

### 2.1 推荐优先级

| 框架 | 建议优先级 | 使用场景 |
|---|---:|---|
| LangGraph | P0 | 状态机、多步流程、可恢复执行、人审、复杂业务 agent |
| OpenAI Agents SDK | P0 | OpenAI 生态、tools/handoffs/guardrails/tracing、快速构建生产级 agent |
| Claude Agent SDK / Claude Code | P1 | coding agent、long-running engineering tasks |
| LlamaIndex Workflows | P1 | RAG、文档处理、agentic document workflows |
| Google ADK | P1 | 多 agent、A2A、Google Cloud 生态 |
| AutoGen AgentChat | P2 | 多 agent 研究和原型；注意关注维护状态和生态变化 |
| Hugging Face smolagents | P2 | 轻量 agent、开源教学、快速理解 agent loop |

### 2.2 具体资源

| 资源 | 重点 |
|---|---|
| [LangGraph Durable Execution](https://docs.langchain.com/oss/python/langgraph/durable-execution) | 持久化执行、失败恢复、长任务 |
| [LangGraph Human-in-the-Loop](https://docs.langchain.com/oss/python/langchain/frontend/human-in-the-loop) | interrupt、审批、修改 agent state |
| [LangGraph GitHub](https://github.com/langchain-ai/langgraph) | resilient agents、durable execution、human-in-the-loop |
| [LlamaIndex Agent Workflows](https://developers.llamaindex.ai/typescript/framework/modules/agents/agent_workflow/) | orchestrate one or multiple agents with tools |
| [LlamaIndex Multi-Agent Report Generation](https://developers.llamaindex.ai/python/examples/agent/agent_workflow_multi/) | 多 agent report generation notebook |
| [Agentic Document Workflows](https://www.llamaindex.ai/blog/introducing-agentic-document-workflows) | 文档处理、RAG、structured outputs、agentic orchestration |
| [AutoGen AgentChat](https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/index.html) | high-level API for multi-agent applications |
| [AutoGen v0.4](https://www.microsoft.com/en-us/research/blog/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) | modularity、scalability、robustness |
| [Hugging Face Agents Course](https://huggingface.co/agents-course) | 工具、思考、行动、观察、框架、Agentic RAG、eval |

---

## 3. 论文阅读路线

建议每篇论文都做三件事：

1. 用 1 页纸总结：问题、核心方法、关键实验、局限。
2. 写一个 100～300 行 demo 复现核心思想。
3. 把它放进你的 Agent 项目中，形成可观测指标。

---

### 3.1 LLM 基础与对齐

| 论文/资料 | 重点 | 链接 |
|---|---|---|
| Attention Is All You Need | Transformer 基础 | https://arxiv.org/abs/1706.03762 |
| BERT | 预训练/微调范式 | https://arxiv.org/abs/1810.04805 |
| InstructGPT / RLHF | 指令对齐与人类反馈 | https://arxiv.org/abs/2203.02155 |
| DPO | 直接偏好优化 | https://arxiv.org/abs/2305.18290 |
| LoRA | 参数高效微调 | https://arxiv.org/abs/2106.09685 |
| QLoRA | 4-bit 量化 + LoRA | https://arxiv.org/abs/2305.14314 |

---

### 3.2 RAG、检索、上下文选择

| 论文/资料 | 重点 | 链接 |
|---|---|---|
| Retrieval-Augmented Generation | 参数化记忆 + 非参数化检索 | https://arxiv.org/abs/2005.11401 |
| ColBERT | late interaction 检索与重排 | https://arxiv.org/abs/2004.12832 |
| Self-RAG | 按需检索、自我反思 | https://arxiv.org/abs/2310.11511 |
| Corrective RAG | 检索质量判断与纠偏 | https://arxiv.org/abs/2401.15884 |
| RAPTOR | 树状摘要索引，支持层次检索 | https://arxiv.org/abs/2401.18059 |
| Ragas | RAG 评估框架 | https://docs.ragas.io/en/stable/ |

---

### 3.3 Agent 核心算法

| 论文 | 重点 | 链接 |
|---|---|---|
| ReAct | reasoning + acting 交替 | https://arxiv.org/abs/2210.03629 |
| Toolformer | 模型学习何时调用工具 | https://arxiv.org/abs/2302.04761 |
| Tree of Thoughts | 多路径搜索、评估、回溯 | https://arxiv.org/abs/2305.10601 |
| Reflexion | 语言反馈 + episodic memory | https://arxiv.org/abs/2303.11366 |
| Generative Agents | memory、reflection、planning | https://arxiv.org/abs/2304.03442 |
| Voyager | 自动课程、技能库、持续探索 | https://arxiv.org/abs/2305.16291 |
| AutoGen | 多 agent 对话框架 | https://arxiv.org/abs/2308.08155 |
| SWE-agent | agent-computer interface、coding agent | https://arxiv.org/abs/2405.15793 |

---

### 3.4 Agent 评估与 Benchmark

| 论文/Benchmark | 重点 | 链接 |
|---|---|---|
| AgentBench | 多环境评估 LLM-as-Agent | https://arxiv.org/abs/2308.03688 |
| WebArena | 真实 Web 环境中的 Agent 任务 | https://arxiv.org/abs/2307.13854 |
| GAIA | 通用 AI assistant benchmark | https://arxiv.org/abs/2311.12983 |
| τ-bench | 工具-用户-Agent 交互评估 | https://arxiv.org/abs/2406.12045 |
| SWE-bench | 真实 GitHub issue 修复 | https://www.swebench.com/SWE-bench/ |
| OSWorld | 电脑操作系统环境 agent benchmark | https://arxiv.org/abs/2404.07972 |
| Terminal-Bench | 终端任务评估 | https://www.tbench.ai/ |
| Survey on Evaluation of LLM-based Agents | 评估能力、应用、benchmark、框架综述 | https://arxiv.org/abs/2503.16416 |
| Evaluation and Benchmarking of LLM Agents: A Survey | agent 能力、任务和评估方法综述 | https://arxiv.org/abs/2507.21504 |

---

### 3.5 Memory、Context、Tool Learning、Agent 安全

| 论文/资料 | 重点 | 链接 |
|---|---|---|
| A-Mem: Agentic Memory for LLM Agents | 自组织 agent memory | https://openreview.net/forum?id=FiM0M8gcct |
| Agent Workflow Memory | 从历史工作流中复用经验 | https://arxiv.org/abs/2409.07429 |
| Memory for Autonomous LLM Agents | memory 机制综述 | https://arxiv.org/html/2603.07670v1 |
| A Survey on the Security of Long-Term Memory in LLM Agents | 长期记忆安全 | https://arxiv.org/abs/2604.16548 |
| Agent Skills for LLMs | Skills 架构与治理 | https://arxiv.org/html/2602.12430v3 |
| MCP-Zero | 动态工具发现与工具链构建 | https://arxiv.org/html/2506.01056 |
| Enhancing MCP with Context-Aware Server Collaboration | MCP 中的上下文感知协作 | https://arxiv.org/html/2601.11595v2 |
| Holistic Evaluation and Failure Diagnosis of AI Agents | tracing + failure diagnosis | https://arxiv.org/pdf/2605.14865 |
| ClawArena | 冲突状态下的 agent benchmark | https://arxiv.org/html/2604.04202v1 |

---

## 4. 课程资源

### 4.1 推荐课程路线

| 优先级 | 课程 | 重点 |
|---|---|---|
| P0 | [Hugging Face Agents Course](https://huggingface.co/agents-course) | Agent fundamentals、tools、thoughts/actions/observations、frameworks、Agentic RAG、eval |
| P0 | [OpenAI Building Agents Track](https://developers.openai.com/tracks/building-agents) | OpenAI 官方 agent 构建流程 |
| P0 | [DeepLearning.AI Agentic AI](https://www.deeplearning.ai/courses/agentic-ai) | 规划、工具使用、反思、多步工作流 |
| P0 | [DeepLearning.AI Evaluating AI Agents](https://www.deeplearning.ai/courses/evaluating-ai-agents) | 系统化评估 agent 表现 |
| P1 | [DeepLearning.AI Retrieval Augmented Generation](https://www.deeplearning.ai/courses/retrieval-augmented-generation) | production-ready RAG |
| P1 | [DeepLearning.AI Building and Evaluating Advanced RAG](https://www.deeplearning.ai/courses/building-evaluating-advanced-rag) | sentence-window、auto-merging、RAG triad |
| P1 | [DeepLearning.AI Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/courses/building-agentic-rag-with-llamaindex) | autonomous research agent、agentic retrieval |
| P1 | [OpenAI Academy](https://academy.openai.com/) | OpenAI 专家与社区课程 |
| P1 | [OpenAI Academy: How to Build AI Agents](https://academy.openai.com/public/clubs/india-gkubq/videos/how-to-build-ai-agents-2025-06-04) | Responses API、built-in tools、Agents SDK |
| P1 | [Anthropic Courses](https://docs.anthropic.com/en/docs/resources/courses) | MCP、sub-agents、Claude Code |
| P1 | [Berkeley CS294/194-196 LLM Agents](https://rdi.berkeley.edu/llm-agents/f24) | LLM agents 学术/工程体系 |
| P1 | [Berkeley CS294/194-196 Agentic AI](https://rdi.berkeley.edu/agentic-ai/f25) | Agentic AI、推理、规划、基础设施、应用 |
| P2 | [Coursera: Building AI Agents with OpenAI Specialization](https://www.coursera.org/specializations/building-ai-agents-openai) | OpenAI agent specialization |

---

## 5. 20 周学习计划

建议每周投入 10～15 小时。每周固定节奏：

- 3 小时：论文/官方文章阅读。
- 3 小时：代码复现。
- 4～6 小时：项目开发。
- 1 小时：失败样本分析。
- 1 小时：写技术笔记/博客。

---

### 阶段 1：LLM 与 Agent 基础（第 1～3 周）

| 周 | 主题 | 必读 | 实践 | 产出 |
|---|---|---|---|---|
| 第 1 周 | Transformer 与 LLM 基础 | Attention、BERT、InstructGPT | 用 API 调用不同模型，比较 decoding、temperature、system prompt | 《LLM 基础速查笔记》 |
| 第 2 周 | Prompting、tool calling、structured output | OpenAI Cookbook、Claude Prompt Engineering | 实现 function calling：天气、SQL、搜索、计算器 | `tool_calling_baseline.py` |
| 第 3 周 | ReAct 与基本 Agent Loop | ReAct、Toolformer、HF Agents Course Unit 1 | 手写 ReAct loop：thought/action/observation | 一个最小可运行 agent |

---

### 阶段 2：RAG、检索与 Context Engineering（第 4～6 周）

| 周 | 主题 | 必读 | 实践 | 产出 |
|---|---|---|---|---|
| 第 4 周 | RAG 基础 | RAG、ColBERT、DL.AI RAG | BM25 + dense retrieval + reranker | 基础知识库 QA |
| 第 5 周 | Agentic RAG | Self-RAG、Corrective RAG、LlamaIndex ADW | query rewrite、query decomposition、citation grounding | Agentic RAG v1 |
| 第 6 周 | Context Engineering | Anthropic Context Engineering、MCP Resources | just-in-time context loading、file path/resource id 引用 | Context 管理模块 |

关键指标：

- retrieval precision@k
- answer groundedness
- answer relevance
- citation accuracy
- average latency
- average token cost
- refusal accuracy

---

### 阶段 3：Planning、Reflection、Memory、Skills（第 7～10 周）

| 周 | 主题 | 必读 | 实践 | 产出 |
|---|---|---|---|---|
| 第 7 周 | Planning | Tree of Thoughts、Anthropic Building Effective Agents | plan-and-execute、planner-worker、evaluator-optimizer | Planner/Executor 原型 |
| 第 8 周 | Reflection | Reflexion、Voyager | 失败后写 reflection memory，下一轮复用 | Reflection memory demo |
| 第 9 周 | Memory | A-Mem、Agent Workflow Memory、Memory Survey | episodic memory、semantic memory、workflow memory | Agent memory store |
| 第 10 周 | Skills | Claude Agent Skills、Agent Skills paper、AGENTS.md | 把常用任务封装为 skill folder | `skills/` 目录规范 |

你需要特别关注：

- Memory 什么时候写入？
- Memory 什么时候检索？
- Memory 是否需要过期、合并、去重？
- Reflection 是提升任务成功率，还是引入错误偏见？
- Skill 与工具的区别是什么？

---

### 阶段 4：Agent 工程框架与协议（第 11～13 周）

| 周 | 主题 | 必读 | 实践 | 产出 |
|---|---|---|---|---|
| 第 11 周 | LangGraph | Durable Execution、Human-in-the-Loop | 用 LangGraph 重构前面的 Agentic RAG | 可恢复执行的 agent graph |
| 第 12 周 | OpenAI / Claude SDK | OpenAI Agents SDK、Claude Agent SDK | tools、handoffs、guardrails、sessions | 两套 SDK 对比文档 |
| 第 13 周 | MCP / A2A / ADK | MCP Spec、Google ADK、A2A | 写一个 MCP server + 一个 ADK multi-agent demo | MCP 工具服务 + 多 agent demo |

工程能力检查：

- 是否有状态持久化？
- 是否支持失败恢复？
- 是否能回放每一步？
- 工具调用是否有 schema 校验？
- 高风险动作是否有人审？
- 是否能限制工具权限？
- 是否有 tracing 和成本统计？

---

### 阶段 5：Agent Evaluation 与 Observability（第 14～16 周）

| 周 | 主题 | 必读 | 实践 | 产出 |
|---|---|---|---|---|
| 第 14 周 | Agent Benchmark | AgentBench、WebArena、GAIA、τ-bench、SWE-bench | 设计自己的 eval set | `eval_cases.jsonl` |
| 第 15 周 | Trajectory Eval | Survey on Evaluation、Holistic Failure Diagnosis | 给每条轨迹打标签：无效工具、错工具、早停、幻觉 | failure taxonomy |
| 第 16 周 | Observability | LangSmith/Phoenix/OpenTelemetry 思路 | trace 每个 LLM call、tool call、latency、cost | eval dashboard |

建议你的 eval schema：

```json
{
  "task_id": "exp_analysis_001",
  "user_goal": "分析 A/B 实验中 CTR 提升是否可信",
  "expected_tools": ["sql_query", "metric_calculator", "segment_analyzer"],
  "success_criteria": [
    "能计算核心指标",
    "能识别样本不均衡",
    "能给出风险提示",
    "不会过度断言因果"
  ],
  "risk_tags": ["statistics", "business_decision"],
  "gold_answer": "..."
}
```

---

### 阶段 6：作品集项目（第 17～20 周）

#### 项目 A：推荐/广告实验分析 Agent

**目标**：构建一个能分析推荐系统 A/B 实验的 Agent。

功能要求：

- 接入 DuckDB / SQLite / mock data warehouse。
- 自动生成 SQL。
- 自动分析 CTR、CVR、留存、GMV、多目标 trade-off。
- 识别 sample ratio mismatch、Simpson's paradox、异常分桶。
- 输出实验结论、风险、下一步建议。
- 所有结论必须引用 SQL 结果或统计计算结果。
- 高风险结论必须触发 verifier。

推荐架构：

```text
User Query
  -> Intent Parser
  -> Metric Planner
  -> SQL Agent
  -> Statistics Agent
  -> Segment Analyzer
  -> Business Critic
  -> Report Generator
  -> Verifier
```

核心指标：

- SQL correctness
- metric correctness
- hallucination rate
- unsupported conclusion rate
- analysis coverage
- average cost
- average latency

---

#### 项目 B：Agentic RAG + 企业知识库助手

**目标**：构建一个能处理复杂知识查询的 research agent。

功能要求：

- hybrid retrieval：BM25 + embedding。
- reranker。
- query decomposition。
- citation grounding。
- self-check。
- “不知道就拒答”。
- 支持 MCP resource/tool 接入。
- 支持 eval regression。

推荐架构：

```text
Question
  -> Query Classifier
  -> Query Decomposer
  -> Retriever Pool
  -> Reranker
  -> Evidence Selector
  -> Answer Generator
  -> Grounding Verifier
  -> Citation Formatter
```

核心指标：

- context precision
- answer faithfulness
- citation recall
- citation precision
- refusal correctness
- cost/latency

---

#### 项目 C：Customer Support Tool-Using Agent

**目标**：模拟真实业务 agent，重点训练工具调用、策略遵守、人审与安全。

功能要求：

- 查询订单、退款、改地址。
- 接入 order/refund/policy API。
- 金额阈值触发人审。
- 风险动作需要 confirmation。
- 记录工具调用轨迹。
- 用 user simulator 做自动评估。

推荐参考：

- τ-bench
- OpenAI agent governance cookbook
- Anthropic safe/trustworthy agents
- Google tool governance

---

#### 项目 D：Mini Coding Agent

**目标**：实现简化版 SWE-agent。

功能要求：

- 输入 issue。
- agent 搜索文件、定位 bug、修改 patch。
- 运行测试。
- 失败后反思并重试。
- 使用 AGENTS.md 提供 repo 规则。
- 使用 sandbox 限制执行权限。

推荐参考：

- SWE-agent
- SWE-bench
- Claude Code best practices
- OpenAI Codex / sandbox / harness engineering

---

## 6. 面试知识体系

### 6.1 必须能讲清楚的问题

#### Agent 基础

- 什么是 workflow，什么是 agent？
- ReAct、plan-and-execute、evaluator-optimizer 有什么区别？
- Tool calling 和 function calling 有什么区别？
- Agent 为什么需要状态？
- Long-running agent 的难点是什么？

#### RAG / Context

- 普通 RAG 和 Agentic RAG 的区别？
- 怎么做 query decomposition？
- 怎么判断检索失败还是生成失败？
- Context engineering 和 prompt engineering 的区别？
- Memory 应该如何写入、检索、压缩、过期？

#### 工程架构

- LangGraph 为什么适合复杂 agent？
- MCP、A2A、AGENTS.md 分别解决什么问题？
- 如何设计工具 schema？
- 如何做高风险工具调用的人审？
- 如何做 sandbox？
- 如何防 prompt injection？

#### 评估

- Agent 成功率如何定义？
- 为什么只看 final answer 不够？
- 如何评估 trajectory？
- 如何做 regression eval？
- 如何定位 agent 失败原因？
- 如何把成本和延迟纳入优化目标？

---

## 7. 作品集 README 模板

每个项目建议按下面结构写 README：

```markdown
# Project Name

## Problem
这个 Agent 解决什么业务问题？

## Why Agent?
为什么普通 LLM/RAG/规则系统不够？

## Architecture
放一张模块图，说明 planner、tools、memory、verifier、eval。

## Tools
列出工具 schema、权限、失败处理。

## Evaluation
- Task success rate
- Tool accuracy
- Groundedness
- Cost
- Latency
- Failure taxonomy

## Safety
- Prompt injection handling
- Permission boundary
- Human-in-the-loop
- Audit log

## Key Results
用表格展示迭代前后指标变化。

## Lessons Learned
总结失败案例和改进方向。
```

---

## 8. 你可以持续跟踪的方向

### 8.1 技术趋势

1. **Agent Runtime 平台化**：OpenAI AgentKit/Agents SDK、Google Agent Platform、Anthropic Managed Agents。
2. **协议标准化**：MCP、A2A、AGENTS.md、AAIF。
3. **Context Engineering 成为核心能力**：按需加载、动态检索、context compaction、long-horizon memory。
4. **Coding Agent 成为主战场**：Codex、Claude Code、SWE-agent、SWE-bench。
5. **Evaluation 从 final answer 走向 trajectory diagnosis**。
6. **Security/Governance 成为上线门槛**：prompt injection、权限、人审、sandbox、审计。
7. **Agent Skills / Skill Library**：将组织知识、流程、脚本封装成可复用能力。
8. **多 Agent 协作协议**：A2A、remote agents、agent teams。
9. **工具数量爆炸后的工具检索与选择**：tool retrieval、dynamic tool loading、code execution with MCP。
10. **推荐/搜索/广告场景的 Agentic Decision System**：实验分析、策略生成、运营自动化、个性化助手。

### 8.2 与推荐算法结合的研究/工程问题

- Agent 的 tool/action selection 能不能建模为 ranking problem？
- Memory retrieval 能不能复用召回 + 重排 + diversity？
- Agent trajectory 能不能做 off-policy evaluation？
- 多 agent 协作中的任务分发能不能用 learning-to-route？
- Agent 的失败样本能不能像推荐系统一样做 hard negative mining？
- Prompt/tool/schema 的优化能不能进入自动实验平台？
- 用户个性化 Agent 如何处理隐私、安全和偏好漂移？
- 推荐系统解释分析能否由 Agent 自动生成并验证？

---

## 9. 最短 8 周冲刺版

如果你要快速转型找机会，可以压缩为：

| 周 | 重点 | 产出 |
|---|---|---|
| 1 | ReAct + Tool Calling + OpenAI/Claude docs | 最小 agent loop |
| 2 | RAG + rerank + citation | 基础 Agentic RAG |
| 3 | Context Engineering + MCP | MCP tool/resource demo |
| 4 | LangGraph + Durable Execution | 可恢复执行的 agent |
| 5 | Reflection + Memory + Skills | skill/memory 原型 |
| 6 | Agent Evaluation | eval set + dashboard |
| 7 | 推荐实验分析 Agent | 作品集项目 1 |
| 8 | Agentic RAG 完整化 + README | 作品集项目 2 |

---

## 10. 建议最终作品集组合

对你最有竞争力的组合：

1. **推荐/广告实验分析 Agent**  
   体现你的推荐算法背景、统计能力、业务理解和 Agent 工程能力。

2. **Agentic RAG + 企业知识库助手**  
   体现检索、上下文工程、RAG eval、grounding。

3. **Tool-Using Customer Support Agent**  
   体现真实业务工具调用、安全、人审、治理。

4. **Mini Coding Agent**  
   体现长任务、sandbox、代码搜索、测试、reflection。

最终定位可以写成：

> 推荐系统背景的大模型 Agent 算法工程师，擅长将 Agent 问题抽象为 retrieval、ranking、tool routing、memory management、trajectory evaluation 与 feedback optimization 闭环。

---

## 11. 每月复盘清单

每个月问自己：

- 我是否新增了一个可运行 demo？
- 我是否为项目新增了 eval cases？
- 我是否降低了失败率、成本或延迟？
- 我是否记录了 failure taxonomy？
- 我是否把一个论文思想转成了代码？
- 我是否能解释一个官方 agent 架构的 trade-off？
- 我的作品集是否能被别人 clone 后跑起来？
- 我的 README 是否有指标，而不是只有功能描述？

---

## 12. 推荐阅读顺序总表

### 第一轮：建立 Agent 基础

1. Anthropic: Building Effective AI Agents
2. OpenAI: A practical guide to building agents
3. Hugging Face Agents Course
4. ReAct
5. Toolformer
6. OpenAI Agents SDK docs
7. Claude Tool Use docs

### 第二轮：建立 RAG/Context 能力

1. RAG
2. ColBERT
3. Self-RAG
4. Anthropic: Effective context engineering
5. LlamaIndex Agentic Document Workflows
6. Ragas
7. DeepLearning.AI RAG / Advanced RAG

### 第三轮：建立 Planning/Memory/Skills

1. Tree of Thoughts
2. Reflexion
3. Voyager
4. Generative Agents
5. A-Mem
6. Agent Workflow Memory
7. Claude Agent Skills
8. AGENTS.md

### 第四轮：建立工程化能力

1. LangGraph Durable Execution
2. LangGraph Human-in-the-Loop
3. OpenAI Responses API / Agents SDK / AgentKit
4. Claude Agent SDK / Claude Code
5. MCP Specification
6. Google ADK
7. A2A Protocol

### 第五轮：建立评估与上线能力

1. AgentBench
2. WebArena
3. GAIA
4. τ-bench
5. SWE-bench
6. Survey on Evaluation of LLM-based Agents
7. OpenAI/Anthropic prompt injection 文章
8. Governance / sandbox / monitoring 文章

---

## 13. 最后建议

你不需要把自己定位成“会调 LangChain 的人”。你的差异化应该是：

- 能把 Agent 的 **工具选择** 建模为排序/决策问题；
- 能把 Agent 的 **记忆检索** 建模为召回/重排问题；
- 能把 Agent 的 **自我改进** 建模为反馈闭环问题；
- 能把 Agent 的 **上线质量** 建模为 eval、trace、guardrail、A/B test 问题；
- 能把推荐系统经验迁移到真实企业 Agent 场景。

这会比普通“LLM 应用工程师”更有竞争力。

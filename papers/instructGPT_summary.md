# 论文精读: Training language models to follow instructions with human feedback

## Reading Verdict

- Depth reached: 深度重构（按三步阅读法完成快速筛选、结构化通读、方法/实验重构）
- Decision: Keep as reference / Build on it / Implement a small RLHF prototype
- Relevance: 这篇论文是现代指令微调和 RLHF 路线的代表作。它把“预训练语言模型会续写文本”推进到“模型能按用户意图完成任务”，也是理解 ChatGPT 类助手、偏好数据、奖励模型、PPO 对齐训练、alignment tax 和安全评测的核心论文。

## Metadata

- Title: Training language models to follow instructions with human feedback
- Authors: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe
- Organization: OpenAI
- arXiv: 2203.02155v1, 2022-03-04
- PDF: `/Users/daixiujia/Documents/projects/agent_learning/papers/instructGPT.pdf`
- Released samples: `https://github.com/openai/following-instructions-human-feedback`

## One-Sentence Contribution

论文提出用人类示范数据做监督微调、用人类偏好排序训练奖励模型、再用 PPO 优化语言模型的 RLHF 流程，使 GPT-3 更能遵循用户指令，并在真实 API prompt 分布上显著优于原始 GPT-3。

## Why This Paper Matters

这篇论文要解决的问题不是“如何让语言模型更大”，而是“如何让已经很大的语言模型更像用户真正想要的助手”。GPT-3 的预训练目标是预测网页文本中的下一个 token，这个目标和“有帮助、诚实、无害地完成用户任务”并不一致，所以大模型会出现不听指令、胡编、输出有害内容、答非所问等行为。

InstructGPT 的关键判断是：行为质量不只来自参数规模，也来自训练目标是否贴近人类偏好。论文最强的 headline result 是，1.3B 参数的 InstructGPT 输出比 175B 参数的原始 GPT-3 更受标注员偏好。这说明对齐训练可以在真实用户任务上带来比单纯扩大模型更直接的收益。

它也把很多后来大模型产品中的核心工程问题提前摊开了：偏好数据怎么采、奖励模型怎么训、RL 训练如何防止 reward hacking、对齐训练会不会伤害原有 NLP 能力、模型到底对齐到谁的偏好、以及安全评测该怎么定义。

## Core Idea

论文的核心不是发明一个新模型结构，而是把 GPT-3 的行为目标从“互联网文本续写”改成“输出人类更喜欢的回答”。完整流程分三步：

1. **SFT: Supervised Fine-Tuning**  
   收集标注员针对 prompt 写出的理想回答，用这些 demonstrations 微调 GPT-3。

2. **RM: Reward Modeling**  
   对同一个 prompt 生成多个候选回答，让标注员排序；训练奖励模型预测哪一个回答更符合人类偏好。

3. **PPO: Reinforcement Learning from Human Feedback**  
   把奖励模型当作打分器，用 PPO 继续优化 SFT 模型。最终主要报告的 InstructGPT 是 PPO-ptx，即在 PPO 训练中额外混入预训练分布的语言建模梯度，降低原有能力退化。

这个流程的实质是：先用示范数据把模型拉到“能像助手一样回答”的区域，再用偏好数据学习更细粒度的回答质量，最后用强化学习直接优化人类偏好分数。

## Method Reconstruction

### 1. Problem Formulation

论文把 alignment 定义得很实用：让语言模型按用户意图行动。这个意图既包括显式意图，例如“写一个故事”“总结这段文本”，也包括隐式意图，例如不要编造事实、不要输出有害内容、不要误导用户。

论文借用了 helpful、honest、harmless 这组三元目标：

- Helpful: 帮用户完成任务，理解明确指令、few-shot pattern 或隐含续写意图。
- Honest: 不编造、不误导。论文实际评测中主要用 truthfulness 和 hallucination 作为代理指标。
- Harmless: 避免可能造成物理、心理、社会伤害的输出。论文实际评测中使用毒性、偏见、有害建议、暴力/性内容等代理指标。

这里的关键限制是：论文并没有解决“全人类价值对齐”，而是把模型对齐到研究者设计的标注规范、约 40 名标注员的判断、以及 OpenAI API Playground 用户的 prompt 分布上。

### 2. Prompt and Data Pipeline

数据来自两类 prompt：

- 标注员自己写的 prompt，用于启动早期 InstructGPT 训练。
- OpenAI API Playground 上早期 InstructGPT 用户提交的 prompt。论文不使用生产 API 数据，并过滤 PII。

标注员写 prompt 时有三种形式：

- Plain: 任意自然语言任务。
- Few-shot: 写出任务指令和多个 query/response 示例。
- User-based: 根据 API waitlist 中的 use case 写 prompt。

数据被拆成三类：

| Dataset | Purpose | Training prompts | Source |
| --- | --- | ---: | --- |
| SFT data | 训练监督微调模型 | 11,295 labeler + 1,430 customer | 标注员示范回答 |
| RM data | 训练奖励模型 | 6,623 labeler + 26,584 customer | 标注员排序 |
| PPO data | RLHF prompt 输入 | 31,144 customer | 无人工标签 |

API prompt 的任务分布很偏开放式生成，而不是传统 NLP benchmark 那种分类/问答为主：

| Use case | Share |
| --- | ---: |
| Generation | 45.6% |
| Open QA | 12.4% |
| Brainstorming | 11.2% |
| Chat | 8.4% |
| Rewrite | 6.6% |
| Summarization | 4.2% |
| Classification | 3.5% |
| Other | 3.5% |
| Closed QA | 2.6% |
| Extract | 1.9% |

这个分布很重要，因为它解释了为什么 FLAN/T0 这类公开 NLP 任务微调模型在真实 API prompt 上不如 InstructGPT。真实用户大量要的是生成、头脑风暴、改写和对话，而不是可用标准答案自动评测的任务。

### 3. Human Data Collection

论文雇佣约 40 名 contractor，来源包括 Upwork 和 ScaleAI。标注员筛选关注四点：

- 对 sensitive speech 标注与研究者的一致性。
- 对多个模型回答排序与研究者的一致性。
- 对敏感 prompt 写出高质量 demonstration 的能力。
- 自评能识别哪些主题或群体相关的敏感内容。

标注员在训练数据阶段更优先 helpfulness；最终评估时更强调 truthfulness 和 harmlessness。这个差异很关键：模型训练目标和最终评测目标并不完全一致，也解释了某些局限，例如模型可能过度遵循有害用户指令。

一致性方面，训练标注员彼此 agreement 为 72.6±1.5%，held-out 标注员为 77.3±1.3%。这说明任务主观性强，但一致性并不低。

### 4. Step 1: Supervised Fine-Tuning

SFT 从 GPT-3 预训练模型出发，用人工 demonstrations 做监督学习。

主要设置：

| Item | Setting |
| --- | --- |
| Architecture | GPT-3 architecture |
| Model sizes | 1.3B, 6B, 175B |
| Epochs | 16 |
| LR schedule | cosine decay to 10% |
| Dropout | residual dropout 0.2 |
| 1.3B / 6B LR | 9.65e-6 |
| 1.3B / 6B batch size | 32 |
| 175B LR | 5.03e-6 |
| 175B batch size | 8 |
| Model selection | validation RM score |

一个反直觉点是：SFT validation loss 在 1 epoch 后过拟合，但继续训练到更多 epoch 反而提高 RM score 和人类偏好评分。说明普通 next-token validation loss 不一定对应“作为助手的回答质量”。

### 5. Step 2: Reward Model

奖励模型输入 prompt 和 completion，输出一个标量 reward。论文用 SFT 模型去掉最终 unembedding layer 后接标量投影。

标注时，每个 prompt 会展示 K 个模型输出，K 在 4 到 9 之间。标注员排序后，可以产生 K choose 2 个 pairwise comparison。奖励模型学习让人类偏好的回答分数更高。

损失函数可以理解为 Bradley-Terry / pairwise logistic loss：

```text
loss(theta) = - E[ log sigmoid(r_theta(x, y_w) - r_theta(x, y_l)) ]
```

其中 `y_w` 是更受偏好的 completion，`y_l` 是较差 completion。

论文的一个工程细节很重要：同一个 prompt 下的 pairwise comparisons 高度相关。如果把所有 pair 当成独立样本随机打散，奖励模型会很快过拟合。因此作者把同一个 prompt 的所有 comparisons 作为一个 batch element 来训练，这样每个 completion 只需要一次 forward，也降低重复更新造成的过拟合。

奖励模型设置：

| Item | Setting |
| --- | --- |
| RM size | 6B |
| RM used for | 所有 PPO policy sizes |
| Epochs | 1 |
| LR | 9e-6 |
| Batch size | 64 prompts |
| Schedule | cosine decay to 10% |
| Scalar normalization | demonstrations mean reward normalized to 0 |

论文尝试过 175B RM，但训练不稳定，也会显著增加 PPO 计算成本，所以最终使用 6B RM。

### 6. Step 3: PPO / PPO-ptx

RL 阶段是一个 bandit environment：

1. 环境给出一个 customer prompt。
2. policy 生成回答。
3. reward model 给 prompt-response 打分。
4. episode 结束。

为了防止模型过度优化奖励模型，论文加入相对 SFT policy 的 per-token KL penalty。直觉上，模型可以提高奖励，但不能离 SFT 行为分布太远。

PPO 的目标可理解为：

```text
maximize reward_model_score - beta * KL(policy || SFT_policy)
```

PPO-ptx 进一步加入预训练数据的语言建模目标：

```text
maximize reward_model_score
       - beta * KL(policy || SFT_policy)
       + gamma * logprob_on_pretraining_data
```

其中 `gamma` 控制预训练混合项强度。论文默认把 InstructGPT 指 PPO-ptx 模型。

RLHF 设置：

| Item | Setting |
| --- | --- |
| Init policy | SFT with 2 epochs and 10% pretraining mix |
| Episodes | 256k |
| Unique PPO prompts | about 31k |
| Batch size | 512 |
| Minibatch size | 64 |
| PPO clip ratio | 0.2 |
| Rollout temperature | 1 |
| KL coefficient beta | 0.02 |
| PPO-ptx pretraining examples | 8x RL episodes |
| Context length | 2k tokens |
| Max prompt length | 1k tokens |
| Max response length | 1k tokens |

PPO-ptx 是整篇论文里非常实际的修补：纯 PPO 会提高人类偏好，但会伤害一些公开 NLP benchmark；混入预训练梯度能显著缓解这种 alignment tax。

## Evaluation Reconstruction

### 1. API Prompt Distribution

主评测是在 held-out customer prompt 上做人类偏好排序。训练/验证/测试按 user ID 拆分，避免同一个用户的 prompt 同时出现在训练和测试中。

评测指标包括：

- 与 baseline 的 win rate。baseline 主要是 175B SFT。
- 1-7 Likert overall quality。
- 是否尝试正确任务。
- 是否适合 customer assistant。
- 是否满足明确约束。
- closed-domain 任务中是否 hallucinate。
- 是否包含性、暴力、有害建议、protected class 贬损等内容。

论文还在 GPT-3 API prompt distribution 上评测，避免只在“专为 InstructGPT 写的 prompt”上赢 GPT-3。

### 2. Safety and Truthfulness Benchmarks

论文用以下数据集评估 truthfulness、toxicity 和 bias：

| Dataset | Measures | Notes |
| --- | --- | --- |
| TruthfulQA | 真实性、真实性+信息性 | 817 short-answer questions |
| RealToxicityPrompts | 毒性 | Perspective API + human evaluation |
| Winogender | 偏见代理指标 | 用二选一 entropy 衡量偏向 |
| CrowS-Pairs | 偏见代理指标 | 用二选一 entropy 衡量偏向 |

TruthfulQA 的评测有两类 prompt：普通 QA prompt，以及带有“如果不确定就回答 I have no comment”的 instruction prompt。InstructGPT 在 instruction prompt 下更能遵循“不确定就别编”的约束。

### 3. Public NLP Benchmarks

论文还评估传统 NLP 能力，包括 DROP、QuAC、SQuADv2、HellaSwag、SST、RTE、WSC、WMT 2015 Fr-En、CNN/DailyMail、Reddit TLDR 等。这一部分不是为了证明 InstructGPT 最会刷榜，而是为了检查对齐训练是否损伤预训练模型已有能力。

## Experiments and Evidence

### 1. InstructGPT 明显优于 GPT-3

最核心结果来自 Figure 1 和 Section 4.1：

- 1.3B InstructGPT 输出比 175B GPT-3 更受偏好。
- 175B InstructGPT 相比 175B GPT-3，直接对比中被偏好 85±3%。
- 175B InstructGPT 相比 few-shot prompted 175B GPT-3，被偏好 71±4%。
- 从 GPT-3 到 prompted GPT-3，到 SFT，再到 PPO/PPO-ptx，偏好分数逐步提升。

这个结果支撑了论文最核心的判断：指令遵循和用户满意度不是参数规模自动带来的，对齐数据和训练目标非常关键。

### 2. InstructGPT 更会遵循约束，也更少 hallucination

在 API distribution 的人工 metadata 上，InstructGPT 相比 GPT-3：

- 更常尝试完成正确任务。
- 更适合作为 customer assistant。
- 更常满足用户显式约束，例如段落数、格式、长度限制。
- 在 closed-domain 任务中更少编造输入中没有的信息。

论文摘要级结果给出 closed-domain hallucination rate：InstructGPT 约 21%，GPT-3 约 41%。这不是完全解决幻觉，但差距很实际。

### 3. Truthfulness 提升明显

在 TruthfulQA 上，PPO 模型比 GPT-3 更常生成 truthful and informative answers。Table 14 中，175B GPT 在 QA prompt 下 true+info 为 0.251，而 175B PPO 为 0.752，175B PPO-ptx 为 0.689。

需要注意，PPO-ptx 在某些 TruthfulQA prompt 组合上不总是最好。比如 Table 14 中 QA + instruction 的 true+info，PPO-ptx 175B 为 0.315，低于 PPO 175B 的 0.588。论文主体强调的是整体趋势：RLHF 显著提高 truthfulness，但不同 prompt 和预训练混合会带来复杂 trade-off。

### 4. Toxicity 小幅改善，Bias 没有明显改善

RealToxicityPrompts 上，当模型被要求以 respectful 方式续写时，InstructGPT 生成毒性更低的文本。论文摘要称约减少 25% toxic outputs。

但有两个重要细节：

- 如果没有 respectful prompt，InstructGPT 和 GPT-3 的毒性差异不明显。
- 如果明确要求模型输出有偏见/冒犯语言，InstructGPT 可能比 GPT-3 更毒，因为它更会遵循用户指令。

偏见方面，Winogender 和 CrowS-Pairs 没有显示 InstructGPT 显著优于 GPT-3。有时加上 respectful instruction 后 entropy 反而更低，意味着模型更确定地偏向某些选项。论文没有宣称解决社会偏见问题。

### 5. PPO-ptx 缓解 Alignment Tax

纯 PPO 会在部分公开 NLP 任务上退化，尤其是 SQuADv2、DROP、HellaSwag、WMT Fr-En。论文称这是 alignment tax：为了对齐用户偏好而付出的能力代价。

PPO-ptx 通过混入预训练梯度缓解退化。Figure 33 显示，调大 pretraining loss coefficient 可以恢复 DROP 和 SQuADv2 表现，同时 validation reward 下降不大。Figure 34 显示，单纯调大 KL penalty 无法完全解决问题，还会明显伤害 validation reward。

这说明保持预训练分布上的能力，不只是“别离 SFT 太远”，还需要直接继续优化预训练数据的 likelihood。

### 6. Held-out Labelers 支持泛化，但范围有限

held-out 标注员没有参与训练数据生产，但他们对模型的偏好趋势与训练标注员类似。奖励模型用 5-fold labeler split 训练时：

- 预测训练组 labeler 偏好的准确率：72.4±0.4%
- 预测 held-out 组 labeler 偏好的准确率：69.6±0.9%

这说明模型不只是机械过拟合某几个标注员，但泛化范围仍然只是“同来源标注员”，不是广泛社会群体。

### 7. FLAN/T0 不如真实偏好数据

论文把 175B GPT-3 分别在 FLAN 和 T0 上 fine-tune，再和 InstructGPT 对比。结果是 FLAN/T0 比原始 GPT-3 好，但不如 SFT，更不如 InstructGPT。

直接对比中：

- 175B InstructGPT over FLAN: 78±4%
- 175B InstructGPT over T0: 79±4%

论文解释很到位：公开 NLP 数据集偏向可自动评测的任务，而真实 API prompt 更偏开放生成和头脑风暴。对 Agent 工程来说，这是非常重要的提醒：训练/评测数据分布比 benchmark 名气更关键。

## Ablations and What Actually Matters

### SFT 本身已经很强，但不够

SFT 把 GPT-3 推向指令跟随模式，是后续 RLHF 的起点。仅靠 prompt engineering 也能改善 GPT-3，但 SFT 明显更稳。PPO/RLHF 进一步提升细粒度偏好。

实际理解可以是：

- Prompting: 临时告诉模型“你现在要像助手”。
- SFT: 把“像助手”写进模型参数。
- RLHF: 在多个“都像助手”的回答里学习哪个更让人满意。

### Reward model 的数据组织很关键

同一个 prompt 下 K 个 completion 产生的 pairwise comparison 高度相关。如果把相关 pair 当独立样本训练，会导致过拟合。论文的 batch 设计是一个容易被忽略但很重要的工程点。

### PPO 不能只追求 reward

如果只优化奖励模型，policy 可能偏离语言模型原本能力，甚至利用奖励模型漏洞。KL penalty 是第一层约束，pretraining mix 是第二层能力保持机制。

### Pretraining mix 比单纯 KL 更有效

Figure 33/34 的对照说明：能力保持不是简单地让 policy 靠近旧模型，而是需要显式维护预训练分布上的建模能力。这对后续做 RLHF 或 DPO/偏好优化都有启发：偏好优化最好搭配 reference model 或 supervised / LM regularization。

## Key Figures and Tables

| Item | What it shows | Why it matters |
| --- | --- | --- |
| Figure 1 | 不同模型相对 175B SFT 的人类偏好 win rate | 证明 InstructGPT 大幅优于 GPT-3，且 1.3B InstructGPT 可超过 175B GPT-3 |
| Figure 2 | SFT -> RM -> PPO 三步流程图 | 是 RLHF 训练范式的核心图 |
| Table 1 | API prompt use-case 分布 | 说明真实用户分布与公开 NLP benchmark 差异很大 |
| Table 3 | 标注员收集的 metadata 指标 | 展示论文如何把 helpful/honest/harmless 转成可标注代理指标 |
| Figure 4 | 约束遵循、正确任务、assistant 适配、hallucination | 证明 InstructGPT 的改善不只是主观偏好，也反映在可解释行为指标上 |
| Figure 5 | InstructGPT vs FLAN/T0 Likert score | 说明公开 instruction dataset 不等于真实用户偏好数据 |
| Figure 6 | TruthfulQA 结果 | 显示 RLHF 对 truthfulness 有明显帮助 |
| Figure 7 | RealToxicityPrompts 毒性评测 | 说明安全收益依赖 prompt setting，改善有限 |
| Figure 8 | 非英语和代码任务泛化样例 | 说明 instruction following 能部分泛化到低监督领域 |
| Figure 9 | 错误前提和过度 hedging 示例 | 展示 InstructGPT 仍然会犯简单错误 |
| Table 6 | SFT/RM/PPO 数据规模 | 复现或估算数据成本的关键表 |
| Figure 28/29 | public NLP zero-shot/few-shot 性能 | 展示 PPO 退化和 PPO-ptx 缓解退化 |
| Figure 33/34 | pretraining coefficient vs KL coefficient | 证明 pretraining mix 对缓解 alignment tax 更有效 |
| Table 14 | 自动评测完整结果 | 提供 truthfulness、toxicity、bias、NLP task 的细粒度数值 |

## Strengths

- 研究问题非常真实：不是在干净 benchmark 上优化单一指标，而是在真实 API prompt 分布上优化用户体验。
- 方法形成了可复用工程范式：SFT、reward model、PPO、KL penalty、pretraining mix。
- 实验覆盖面广：主观偏好、metadata、truthfulness、toxicity、bias、传统 NLP 能力、held-out labelers、FLAN/T0 对比。
- 对限制讨论诚实：论文明确说模型不是对齐到“全人类价值”，也没有彻底解决幻觉、偏见、有害输出。
- 成本观点重要：论文指出 175B SFT 约 4.9 petaflops/s-days，175B PPO-ptx 约 60 petaflops/s-days，而 GPT-3 预训练约 3640 petaflops/s-days；对齐训练相对预训练成本较低但收益很大。

## Limitations and Risks

- **对齐对象有限**：主要对齐到 OpenAI 研究者、约 40 名 contractor、API Playground 用户构成的偏好分布，不代表所有用户或受影响群体。
- **标注规范会塑造模型价值观**：labeling instructions、界面、研究者答疑都会影响最终模型行为。
- **训练目标和最终安全目标有冲突**：训练中更重 helpfulness，最终评估更重 truthfulness/harmlessness；模型可能学会更好地执行有害指令。
- **仍然会 hallucinate**：closed-domain hallucination 降低但未消失。
- **偏见没有显著改善**：Winogender 和 CrowS-Pairs 上没有明显优于 GPT-3。
- **安全性依赖提示**：toxicity 改善在 respectful prompt 下更明显；没有安全提示时收益有限。
- **可能增强滥用能力**：更会遵循指令的模型，也可能更容易被用于生成误导、仇恨或操纵性内容。
- **public NLP 能力会退化**：纯 PPO 会带来 alignment tax，需要 PPO-ptx 等机制缓解。
- **评测仍是代理指标**：truthfulness、honesty、harmlessness 都不能完全被现有 benchmark 捕捉。

## Reproducibility Notes

- Code/data availability: 论文只释放了部分模型 samples，未释放完整训练数据、标注数据、奖励模型或训练代码。
- Required implementation details:
  - GPT-style causal LM backbone。
  - SFT demonstration dataset。
  - Pairwise/ranking preference dataset。
  - Reward model with scalar head。
  - PPO with KL penalty to SFT reference model。
  - Optional pretraining mix to reduce alignment tax。
  - Human preference evaluation pipeline。
- Missing details:
  - 完整 API prompt 数据不可公开。
  - 标注员实时沟通和规范迭代细节难以完全复现。
  - 训练基础设施、分布式 PPO、大模型稳定性细节有限。
  - 175B 规模模型和 GPT-3 预训练权重不可直接获得。
- Estimated reproduction difficulty: High for full paper; Medium for small-scale prototype. 小模型上复现 SFT -> RM -> PPO 思路可行，但复现论文级结果需要真实用户 prompt、大量人工偏好数据和大规模训练资源。

## Minimal Prototype Plan

如果要把这篇论文思想转成可做的工程项目，可以先做一个小型 RLHF pipeline：

1. 选一个小型开源 causal LM，例如 1B 以下模型。
2. 收集 1k-5k 条 instruction demonstration，先做 SFT。
3. 对每个 prompt 采样 4 个回答，人工或用强模型辅助排序，得到 pairwise preference。
4. 训练 reward model：prompt + response -> scalar reward。
5. 用 PPO 或更简单的 DPO/IPO 类方法做偏好优化。
6. 固定一套 held-out prompts，比较 base / SFT / preference-optimized 三个模型。
7. 额外测 hallucination、格式约束遵循、拒答行为和退化任务。

对学习来说，不必一开始复现 PPO-ptx 的全部复杂度。先把数据流、偏好建模和 reference regularization 跑通，收益更大。

## Questions After Reading

- 人类偏好数据中，多少提升来自“更会遵循格式/语气”，多少来自真实能力提升？
- 如果训练时 helpfulness 优先，最终模型如何系统性学会拒绝有害请求？
- Reward model 的泛化边界在哪里？它能否识别训练分布外的高风险 prompt？
- PPO-ptx 的 pretraining mix 会不会重新引入预训练数据中的有害模式？
- 对齐到不同群体偏好时，应该训练多个可控模型，还是一个能根据价值配置条件化的模型？
- 真实用户偏好、标注员偏好、研究者偏好、社会利益冲突时，训练目标该如何定义？

## Follow-Up Ideas

- 复现一个小规模 SFT + reward model + preference optimization 实验，重点观察 SFT 到 RLHF 的边际收益。
- 对比 PPO、DPO、rejection sampling fine-tuning 在同一偏好数据上的效果和稳定性。
- 做一个“alignment tax”小实验：偏好优化后测原始任务能力是否下降，再加入 LM regularization。
- 专门构造 false premise prompts，测试模型是否会反驳错误前提。
- 把推荐系统中的用户反馈思想迁移进来：显式偏好、隐式点击、长期满意度和短期 reward 的冲突，和 RLHF 很像。
- 建一个 prompt taxonomy，把真实用户任务分成 generation、rewrite、QA、chat、summarization 等，看不同任务上偏好优化收益是否一致。

## Important Prior Work to Chase

| Citation | Why it matters |
| --- | --- |
| Brown et al., 2020, GPT-3 | InstructGPT 的 base model 和主要对比对象 |
| Christiano et al., 2017 | RLHF 的早期通用框架 |
| Ziegler et al., 2019 | 用人类偏好微调文本生成的前序工作 |
| Stiennon et al., 2020 | 将 RLHF 用于摘要任务，是本文方法的直接前身 |
| Askell et al., 2021 | helpful、honest、harmless 框架的重要背景 |
| Wei et al., 2021, FLAN | instruction tuning 的公开任务路线，对比真实用户偏好数据 |
| Sanh et al., 2021, T0 | 多任务 prompt/instruction tuning 代表工作 |
| Lin et al., 2021, TruthfulQA | 真实性评测核心 benchmark |
| Gehman et al., 2020, RealToxicityPrompts | 毒性评测核心 benchmark |
| Nangia et al., 2020, CrowS-Pairs | 社会偏见评测 benchmark |
| Rudinger et al., 2018, Winogender | 性别偏见评测 benchmark |

## Personal Takeaways

这篇论文最值得带走的不是“PPO 是答案”，而是一个更一般的产品化训练思想：当预训练目标和用户目标不一致时，要显式收集目标行为和偏好信号，把模型往真实使用场景里拉。

对 Agent 学习尤其重要的是：Agent 的好坏也很难只靠标准 benchmark 判断。真实用户关心的是是否理解任务、是否遵守约束、是否可靠、是否少胡编、是否知道什么时候拒绝。InstructGPT 给出的范式是，把这些软目标拆成可标注数据、可训练 reward、可评估指标，再持续迭代。

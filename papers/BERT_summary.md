# 论文精读: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Reading Verdict

- Depth reached: 深度重构（对应原 three-pass skill 的最高阅读深度，但正文不再按 Pass 1/2/3 分块）
- Decision: Keep as reference / Implement fine-tuning baseline / Build on it
- Relevance: 这篇论文是现代 NLP 预训练-微调范式的标志性工作。它把 Transformer encoder、masked language modeling、sentence-pair representation 和端到端 fine-tuning 组合成一个统一框架，使同一个预训练模型可以覆盖分类、匹配、抽取式问答、序列标注等任务。对 Agent 和 RAG 学习来说，BERT 的价值不只是“一个 encoder 模型”，更是理解 embedding、reranker、reader、领域微调和预训练目标设计的基础。

## Metadata

- Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- Organization: Google AI Language
- arXiv: 1810.04805v2, 2019-05-24
- PDF: `https://arxiv.org/pdf/1810.04805`
- Code / pretrained models: `https://github.com/google-research/bert`

## One-Sentence Contribution

BERT 提出用 masked language modeling 和 next sentence prediction 预训练一个深层双向 Transformer encoder，再通过极少任务特定输出层进行全参数 fine-tuning，从而在 11 个自然语言理解任务上刷新当时的 state of the art。

## Why This Paper Matters

BERT 要解决的核心问题是：已有预训练语言表示没有同时满足“深层双向上下文建模”和“下游任务端到端微调”。

在 BERT 之前，主流路线大致分成两类：

- Feature-based：ELMo 这类方法把预训练表示当作下游模型的额外特征，优点是灵活，缺点是下游模型仍然需要较多任务结构设计。
- Fine-tuning：OpenAI GPT 这类方法把预训练模型整体迁移到下游任务，优点是任务特定参数少，但预训练采用 left-to-right LM，表示在每一层都只能看左侧上下文。

作者认为 left-to-right 限制会伤害语言理解，尤其是 token-level 任务。例如抽取式问答中，判断某个 token 是否是答案边界，天然需要同时看问题、左侧上下文和右侧上下文。ELMo 虽然拼接了左到右和右到左表示，但两个方向是独立训练的，不是在每一层都联合条件化左右上下文。

BERT 的关键推进是：让预训练阶段本身就产生 deep bidirectional representations，并让这些表示直接作为下游任务模型的初始化。

## Core Idea

BERT 的核心不是发明新的 Transformer block，而是把三个设计组合在一起：

1. 用 Transformer encoder 作为统一主干，让所有 token 在每一层都可以双向 self-attention。
2. 用 masked language model 解决双向预训练中的“看到答案”问题：随机遮住一部分 token，只预测被选中的原始 token。
3. 用统一输入格式处理单句和句对任务：`[CLS] sentence A [SEP] sentence B [SEP]`，再叠加 token embedding、segment embedding、position embedding。

这样做之后，下游任务只需要替换很薄的一层输出头：

- 句子分类 / 句对分类：取 `[CLS]` 最终向量做分类。
- 抽取式问答：对每个 token 的最终向量预测 answer start / end。
- 多选任务：给每个候选构造一个输入序列，用 `[CLS]` 打分后 softmax。
- 序列标注：对每个 token 的最终向量做标签分类。

一句话说，BERT 把“每个任务都要设计一套神经网络结构”改成了“预训练一个强 encoder，再用任务输出层适配”。

## Method Reconstruction

### 1. Model Architecture

BERT 使用多层双向 Transformer encoder。论文主要报告两个规模：

| Model | Layers L | Hidden H | Attention heads A | Parameters |
| --- | ---: | ---: | ---: | ---: |
| BERTBASE | 12 | 768 | 12 | 110M |
| BERTLARGE | 24 | 1024 | 16 | 340M |

Feed-forward/filter size 设为 `4H`，即 BERTBASE 为 3072，BERTLARGE 为 4096。BERTBASE 的规模被故意设成和 OpenAI GPT 接近，方便比较：两者架构大小相近，但 GPT 使用 left-to-right constrained self-attention，BERT 使用双向 self-attention。

### 2. Input Representation

BERT 的输入序列可以表示单句，也可以表示句对。

```text
[CLS] token_1 ... token_n [SEP] token_1 ... token_m [SEP]
```

每个位置的输入向量由三部分相加：

```text
input_embedding = token_embedding + segment_embedding + position_embedding
```

其中：

- Token embedding 使用 WordPiece，词表大小为 30,000。
- Segment embedding 标记 token 属于 sentence A 还是 sentence B。
- Position embedding 表示绝对位置。
- `[CLS]` 放在每个输入序列最前面，其最终 hidden state `C` 用作分类任务的聚合表示。
- `[SEP]` 用来分隔两个片段，也标记序列结束。

这个输入格式很重要，因为它让句对任务不再需要先独立编码再做 cross-attention。把两个文本拼成一个序列后，Transformer self-attention 在编码过程中自然包含了两段文本之间的双向交互。

### 3. Pre-training Task 1: Masked Language Model

如果直接训练双向 language model，每个 token 在多层 self-attention 中会间接看到自己，目标会退化。BERT 的解决办法是 MLM：

1. 随机选择 15% 的 WordPiece token 作为预测目标。
2. 对被选中的位置：
   - 80% 替换成 `[MASK]`
   - 10% 替换成随机 token
   - 10% 保持原 token 不变
3. 用该位置的最终 hidden state 预测原始 token id。

混合替换策略是为了减少 pre-training 和 fine-tuning 的 mismatch，因为 `[MASK]` 不会出现在下游任务输入中。保留 10% 原 token 也迫使模型为所有位置保持分布式上下文表示，而不是只识别 `[MASK]` 位置。

MLM 的代价是每个 batch 只在 15% token 上产生预测损失，训练信号比 left-to-right LM 稀疏。附录 C.1 显示 MLM 收敛略慢，但在 MNLI 上几乎从一开始就超过 left-to-right 预训练模型。

### 4. Pre-training Task 2: Next Sentence Prediction

NSP 用来让模型学习句间关系，服务于 QA、NLI 等句对任务。

构造方式：

- 50% 样本中，sentence B 是 sentence A 在原文中的真实下一句，标签为 `IsNext`。
- 50% 样本中，sentence B 是从语料中随机采样的句子，标签为 `NotNext`。
- 使用 `[CLS]` 的最终向量 `C` 做二分类。

论文报告最终模型在 NSP 上达到 97%-98% accuracy。更关键的是，消融实验显示去掉 NSP 会明显伤害 QNLI、MNLI 和 SQuAD。

需要注意：论文脚注明确说，未经 fine-tuning 的 `C` 并不是一个通用句向量。它在预训练中服务于 NSP，但并不等价于可直接拿来做语义相似度的 sentence embedding。

### 5. Pre-training Corpus and Procedure

预训练语料：

| Corpus | Size |
| --- | ---: |
| BooksCorpus | 800M words |
| English Wikipedia | 2,500M words |

作者强调要使用 document-level corpus，而不是打乱的 sentence-level corpus，因为 NSP 和长连续上下文需要文档内相邻片段。

预训练设置：

| Setting | Value |
| --- | --- |
| Batch size | 256 sequences |
| Max sequence length | 512 tokens |
| Tokens per batch | 128,000 |
| Training steps | 1,000,000 |
| Approximate epochs | 40 epochs over 3.3B words |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Adam beta1 / beta2 | 0.9 / 0.999 |
| Weight decay | 0.01 |
| Warmup | first 10,000 steps |
| LR schedule | linear decay |
| Dropout | 0.1 |
| Activation | GELU |
| Loss | mean MLM loss + mean NSP loss |

为了降低长序列 attention 的成本，训练前 90% steps 使用 sequence length 128，最后 10% steps 使用 sequence length 512，以学习长位置 embedding。BERTBASE 在 4 个 Cloud TPU Pod 配置上训练，BERTLARGE 在 16 个 Cloud TPU 上训练，每次预训练约 4 天。

### 6. Fine-tuning Procedure

Fine-tuning 时，大部分超参数沿用预训练；主要搜索 batch size、learning rate 和 epochs：

- Batch size: 16 或 32
- Learning rate: 5e-5、3e-5 或 2e-5
- Epochs: 2、3 或 4
- Dropout: 0.1

论文强调 fine-tuning 很快：从同一个预训练 checkpoint 出发，所有结果最多约 1 小时可在单个 Cloud TPU 上复现，GPU 上则是几小时级别。SQuAD 模型例子中，约 30 分钟即可在单个 Cloud TPU 上达到 Dev F1 91.0。

## Experiments and Evidence

### 1. GLUE

GLUE 是论文最重要的句级和句对理解评测。Table 1 中，BERT 在所有任务上都超过当时系统。

| System | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| BERTBASE | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.6 |
| BERTLARGE | 86.7/85.9 | 72.1 | 92.7 | 94.9 | 60.5 | 86.5 | 89.3 | 70.1 | 82.1 |

论文摘要还给出官方 GLUE score：BERT 将 GLUE 推到 80.5%，相比之前提升 7.7 个百分点。对比 OpenAI GPT 时，最值得关注的是 BERTBASE 和 GPT 模型规模相近，BERT 的主要差异来自双向预训练和输入/任务设计，而不是简单扩大模型。

### 2. SQuAD v1.1

SQuAD v1.1 是抽取式问答任务，需要在给定 passage 中预测答案 span。

BERT 的 QA head 非常薄：

- 输入为 `[CLS] question [SEP] paragraph [SEP]`
- question 使用 segment A，paragraph 使用 segment B
- 新增 start vector `S` 和 end vector `E`
- 每个 token 的 start / end 概率由最终 token representation 与 `S`/`E` 点积后 softmax 得到

主要结果：

| System | Dev EM | Dev F1 | Test EM | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| Human | - | - | 82.3 | 91.2 |
| #1 Ensemble-nlnet | - | - | 86.0 | 91.7 |
| BiDAF+ELMo Single | - | 85.6 | - | 85.8 |
| BERTBASE Single | 80.8 | 88.5 | - | - |
| BERTLARGE Single | 84.1 | 90.9 | - | - |
| BERTLARGE Ensemble | 85.8 | 91.8 | - | - |
| BERTLARGE Single + TriviaQA | 84.2 | 91.1 | 85.1 | 91.8 |
| BERTLARGE Ensemble + TriviaQA | 86.2 | 92.2 | 87.4 | 93.2 |

摘要中的 headline number 是 SQuAD v1.1 Test F1 93.2，相比之前提升 1.5 F1。更有说服力的是 single BERT 已经超过当时 top ensemble 的 F1。

### 3. SQuAD v2.0

SQuAD v2.0 增加了不可回答问题。BERT 的处理方式很直接：把 no-answer 看作 start 和 end 都在 `[CLS]` 的特殊 span。

预测时比较：

- `s_null = S · C + E · C`
- 最佳非空 span 分数 `s_hat`
- 当 `s_hat > s_null + threshold` 时输出非空答案，否则输出 no-answer

结果：

| System | Dev EM | Dev F1 | Test EM | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| Human | 86.3 | 89.0 | 86.9 | 89.5 |
| #1 Single MIR-MRC | - | - | 74.8 | 78.0 |
| #2 Single nlnet | - | - | 74.2 | 77.1 |
| unet Ensemble | - | - | 71.4 | 74.9 |
| SLQA+ Single | - | - | 71.4 | 74.4 |
| BERTLARGE Single | 78.7 | 81.9 | 80.0 | 83.1 |

摘要中的 headline number 是 SQuAD v2.0 Test F1 83.1，相比之前提升 5.1 F1。

### 4. SWAG

SWAG 测试 grounded commonsense inference。给定一个句子，从四个候选 continuation 中选择最合理的一个。

Fine-tuning 时，BERT 为四个候选分别构造 `[CLS] sentence [SEP] continuation [SEP]`，用 `[CLS]` 表示打分，再对四个分数做 softmax。

| System | Dev | Test |
| --- | ---: | ---: |
| ESIM + GloVe | 51.9 | 52.7 |
| ESIM + ELMo | 59.1 | 59.2 |
| OpenAI GPT | - | 78.0 |
| BERTBASE | 81.6 | - |
| BERTLARGE | 86.6 | 86.3 |
| Human expert | - | 85.0 |
| Human 5 annotations | - | 88.0 |

BERTLARGE 相比 ESIM+ELMo 提升 27.1%，相比 OpenAI GPT 提升 8.3%。

## Ablations: What Actually Matters

### 1. Bidirectionality and NSP

Table 5 是理解 BERT 的核心消融。

| Model | MNLI-m Acc | QNLI Acc | MRPC Acc | SST-2 Acc | SQuAD F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERTBASE | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| No NSP | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| LTR & No NSP | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |
| LTR & No NSP + BiLSTM | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

读法：

- 去掉 NSP 对 QNLI、MNLI、SQuAD 有明显影响，说明句间预训练目标对句对任务有价值。
- 从 bidirectional MLM 改成 left-to-right LM 后，所有任务都变差，MRPC 和 SQuAD 掉得尤其明显。
- 在 left-to-right 模型上加随机初始化 BiLSTM 可以补救 SQuAD 一部分，但仍远不如预训练时就双向的模型，且 GLUE 任务上表现变差。

这组实验支撑了论文主张：BERT 的收益不是单纯来自 Transformer 或数据量，而是来自“深层双向预训练 + fine-tuning”的组合。

### 2. Model Size

Table 6 显示，在相同训练流程下，增大模型几乎稳定提升小数据下游任务。

| L | H | A | MLM ppl | MNLI-m | MRPC | SST-2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 768 | 12 | 5.84 | 77.9 | 79.8 | 88.4 |
| 6 | 768 | 3 | 5.24 | 80.6 | 82.2 | 90.7 |
| 6 | 768 | 12 | 4.68 | 81.9 | 84.8 | 91.3 |
| 12 | 768 | 12 | 3.99 | 84.4 | 86.7 | 92.9 |
| 12 | 1024 | 16 | 3.54 | 85.7 | 86.9 | 93.3 |
| 24 | 1024 | 16 | 3.23 | 86.6 | 87.8 | 93.7 |

论文的一个重要观察是：即使 MRPC 只有约 3.6k 训练样本，大模型也继续受益。作者解释为，当下游任务只新增极少随机初始化参数、并全模型 fine-tune 时，小数据任务可以利用更大、更表达力强的预训练表示。

### 3. Feature-based BERT Still Works

论文虽然主推 fine-tuning，但也验证了 feature-based 用法。CoNLL-2003 NER 上，使用 BERTBASE 固定特征加两层 BiLSTM，也能接近全量 fine-tuning。

| Method | Dev F1 | Test F1 |
| --- | ---: | ---: |
| ELMo | 95.7 | 92.2 |
| CVT | - | 92.6 |
| CSE | - | 93.1 |
| Fine-tuning BERTLARGE | 96.6 | 92.8 |
| Fine-tuning BERTBASE | 96.4 | 92.4 |
| Feature: embeddings | 91.0 | - |
| Feature: second-to-last hidden | 95.6 | - |
| Feature: last hidden | 94.9 | - |
| Feature: weighted sum last four | 95.9 | - |
| Feature: concat last four | 96.1 | - |
| Feature: weighted sum all 12 | 95.5 | - |

最好的 feature-based 设置是拼接最后四层，Dev F1 96.1，只比 fine-tuning BERTBASE 低 0.3。这解释了为什么后续很多检索、序列标注、分类系统会把 BERT 当 encoder feature extractor 使用。

### 4. Masking Strategy

附录 C.2 对 MLM 替换策略做了消融。

| MASK | SAME | RND | MNLI fine-tune | NER fine-tune | NER feature-based |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 80% | 10% | 10% | 84.2 | 95.4 | 94.9 |
| 100% | 0% | 0% | 84.3 | 94.9 | 94.0 |
| 80% | 0% | 20% | 84.1 | 95.2 | 94.6 |
| 80% | 20% | 0% | 84.4 | 95.2 | 94.7 |
| 0% | 20% | 80% | 83.7 | 94.8 | 94.6 |
| 0% | 0% | 100% | 83.6 | 94.9 | 94.6 |

Fine-tuning 对 masking 策略比较鲁棒，但 feature-based NER 对 100% `[MASK]` 更敏感。这符合作者解释：feature-based 模型不能通过 fine-tuning 调整 BERT 表示，因此 pretrain/fine-tune mismatch 会更明显。

## Key Figures and Tables

| Item | What it shows | Why it matters |
| --- | --- | --- |
| Figure 1 | BERT 的 pre-training 和 fine-tuning 总流程 | 说明同一个预训练模型可初始化多种下游任务，任务差异主要在输出层 |
| Figure 2 | token、segment、position 三类 embedding 相加 | 解释 BERT 如何统一表示单句和句对输入 |
| Figure 3 | BERT、OpenAI GPT、ELMo 架构差异 | 强调只有 BERT 在所有层都联合条件化左右上下文 |
| Figure 4 | 分类、句对、QA、NER 等 fine-tuning 图示 | 展示统一 encoder + 轻量 task head 的迁移范式 |
| Figure 5 | MLM 与 left-to-right 预训练步数消融 | MLM 收敛略慢，但下游准确率很早就超过 LTR |
| Table 1 | GLUE 测试集结果 | BERT 在多类语言理解任务上整体超过此前方法 |
| Table 2 | SQuAD v1.1 结果 | BERT 的薄 QA head 达到或超过当时强系统 |
| Table 3 | SQuAD v2.0 结果 | BERT 对不可回答问题也显著提升 |
| Table 4 | SWAG 结果 | BERT 在常识推理多选任务上大幅超过 GPT 与 ELMo baselines |
| Table 5 | 预训练任务消融 | 支撑 bidirectionality 和 NSP 的作用 |
| Table 6 | 模型规模消融 | 说明充分预训练后，大模型也能提升小数据任务 |
| Table 7 | NER feature-based vs fine-tuning | 说明 BERT 既能端到端微调，也能作为固定特征 |
| Table 8 | MLM masking 策略消融 | 说明 80/10/10 策略主要缓解 `[MASK]` mismatch |

## Strengths

- 问题定义准确：作者抓住了 left-to-right fine-tuning 模型和浅层双向 feature 模型之间的空缺。
- 方法非常简洁：没有复杂任务结构，主要靠统一输入表示、MLM、NSP 和全模型 fine-tuning。
- 实验证据覆盖面广：GLUE、SQuAD v1.1/v2.0、SWAG、NER、预训练任务消融、模型规模消融、masking 策略消融都比较完整。
- 工程配方清楚：模型规模、语料、batch、学习率、warmup、训练步数、fine-tuning 搜索范围都有明确说明。
- 对后续工作可复用性强：encoder-only 预训练、CLS pooling、句对拼接、span head、reranker/reader 架构都成为后续系统常用模块。

## Limitations and Risks

- NSP 后续争议较大：BERT 论文中 NSP 有收益，但 RoBERTa 等后续工作显示更强训练配方下 NSP 未必必要，甚至可能不是核心贡献。
- MLM 训练效率低：每个 batch 只预测 15% token，训练信号稀疏；这也是后续 ELECTRA 等方法改进的动机。
- `[MASK]` mismatch 没有完全消除：80/10/10 策略只是缓解，预训练输入分布与真实下游输入仍不同。
- 最大长度和 attention 成本受限：标准 self-attention 对长度是二次复杂度，论文主要覆盖 512 token 内的理解任务。
- `[CLS]` 不是天然通用句向量：论文脚注提醒没有 fine-tuning 的 `C` 并不是 meaningful sentence representation，这一点在很多直接拿 BERT 做 embedding 的实践中容易被忽略。
- 英文语料和任务为主：论文实验集中在英文理解 benchmark，对跨语言、领域迁移、长文本、生成式任务没有直接验证。

## Reproducibility Notes

- Code/data availability: 代码和预训练模型在 `https://github.com/google-research/bert`。训练语料使用 BooksCorpus 和 English Wikipedia。
- Required implementation details:
  - WordPiece vocab 30,000。
  - 输入为 token + segment + position embeddings。
  - MLM 选择 15% token，替换策略为 80% `[MASK]`、10% random、10% unchanged。
  - NSP 按 50% IsNext / 50% NotNext 构造。
  - 预训练 loss 为 MLM loss + NSP loss。
  - Fine-tuning 通常搜索 batch size 16/32、learning rate 5e-5/3e-5/2e-5、epochs 2/3/4。
- Missing or easy-to-miss details:
  - 前 90% steps 用 sequence length 128，后 10% 用 512。
  - BERTLARGE 在小数据集上 fine-tuning 有不稳定性，论文使用多个 random restarts 并按 dev set 选择。
  - SQuAD v2.0 的 no-answer 通过 `[CLS]` span 建模，并需要在 dev set 上选择 threshold。
  - NER 使用 WordPiece 时，以第一个 sub-token 的表示作为 token-level classifier 输入。
- Estimated reproduction difficulty:
  - Fine-tuning 现成 checkpoint：低到中等。
  - 从零预训练 BERTBASE/LARGE：高，主要成本在大语料、TPU/GPU 资源和稳定训练。

## Important Prior Work to Chase

| Citation | Why it matters |
| --- | --- |
| Vaswani et al., 2017, Attention Is All You Need | BERT 的 Transformer encoder 架构来源 |
| Peters et al., 2018a, ELMo | feature-based contextual representation 的代表，BERT 的直接对照对象 |
| Radford et al., 2018, OpenAI GPT | fine-tuning 预训练 Transformer 的代表，BERTBASE 主要与其比较 |
| Taylor, 1953, Cloze task | MLM 目标的早期思想来源 |
| Wang et al., 2018a, GLUE | 语言理解综合 benchmark，BERT 的核心评测平台 |
| Rajpurkar et al., 2016, SQuAD | 抽取式问答任务，展示 token-level 双向表示的重要性 |
| Zellers et al., 2018, SWAG | 常识推理多选任务，展示句对/候选 continuation 建模能力 |
| Wu et al., 2016, WordPiece / GNMT | BERT 使用 WordPiece tokenization 的背景 |

## How to Reuse This Paper in Current Learning

### For embedding and retrieval

BERT 本身不是最理想的直接句向量模型，但它解释了为什么 encoder-only 模型适合做文本表征。后续 bi-encoder、cross-encoder reranker、ColBERT 等方法都能从这里接上：

- bi-encoder 关心如何把文本压缩成可检索向量；
- cross-encoder 继承 BERT 的句对拼接和双向 cross-attention，适合 reranking；
- ColBERT 则保留 token-level 表示，用 late interaction 平衡效果和检索效率。

### For RAG systems

BERT 的 QA fine-tuning 方式可以看作早期 reader 模块雏形：给定 question 和 passage，输出 answer span。现代 RAG 多数改成生成式回答，但 reranker、extractive evidence selection、answerability 判断仍然大量借鉴 BERT-style encoder。

### For model adaptation

BERT 证明了一个重要工程原则：预训练模型越强，下游任务越应该先尝试极简 task head 和全参数/参数高效微调，而不是马上为任务设计复杂网络。这个原则在后续 LLM instruction tuning、adapter、LoRA、domain fine-tuning 中仍然成立。

## Questions After Reading

- NSP 的收益到底来自句间关系建模，还是来自某种数据构造正则化？后续 RoBERTa 的结论需要一起读。
- `[CLS]` 表示为什么在 fine-tuning 后适合分类，但未经 fine-tuning 不适合直接做句向量？这和 pooling、contrastive learning 的关系是什么？
- MLM 只预测 15% token 是否浪费计算？ELECTRA 的 replaced token detection 如何更高效利用每个 token？
- 对长文档任务，BERT 的 512 token 限制该如何处理？Longformer、BigBird、hierarchical encoder 是后续方向。
- 对中文或领域文本，哪些收益来自架构，哪些来自 tokenizer、语料和训练任务？

## Follow-Up Ideas

- 用 HuggingFace 跑一个最小实验：`bert-base-uncased` 在 MRPC 或 SST-2 上 fine-tune，观察 learning rate 和 epoch 对小数据稳定性的影响。
- 对同一批 query-document pairs 比较 bi-encoder 与 BERT cross-encoder reranker，理解“编码效率 vs 交互深度”的 trade-off。
- 阅读 RoBERTa，专门对照 BERT 的 NSP、batch size、训练步数、动态 masking、语料规模。
- 阅读 ELECTRA，理解如何把 MLM 的稀疏监督改成 replaced token detection。
- 阅读 Sentence-BERT，理解为什么原始 BERT 的 `[CLS]` 不适合直接做 semantic textual similarity embedding，以及 contrastive / siamese 改造如何解决。

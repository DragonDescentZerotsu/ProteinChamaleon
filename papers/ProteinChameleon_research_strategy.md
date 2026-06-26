# ProteinChameleon 论文化改造建议

## 结论先行

当前 ProteinChameleon 的核心想法是有价值的：把蛋白质结构离散 token、文本 token 和结构边界 token 放进同一个 causal LM 词表，用早期融合的方式建模 `text <PROT_START> structure tokens <PROT_END> text`。这个方向和 Chameleon、AnyGPT、ESM3、GeoBPE、Yeti、STELLA 等新工作是同一条技术线上。

但如果按当前代码和数据直接写 paper，贡献还不够强。它容易被审稿人归类为“把结构 token 拼到 LLM 里做 function captioning”，并且现有 BLEU/ROUGE/BERTScore 评测不能证明模型真的理解或使用了结构。要冲 ICLR/NeurIPS/ICML 这类 AI 会，建议把项目重构成一个更明确的问题：

> 蛋白结构-文本多模态模型是否真的使用了结构证据，而不是依赖序列、功能文本先验或合成叙述中的语言捷径？

结合后续补充调研，最推荐的投稿主线应收窄为：

> **Counterfactual Structure Reliance for ProteinChameleon：用蛋白特异的反事实结构干预证明早期融合蛋白 LLM 真正利用结构 token。**

MIRAGE 和 latent multimodal interleaving 仍然是很好的动机，但 Cuttlefish 已经覆盖了较泛的 structure-grounded LLM 和 structural hallucination 叙事。因此 ProteinChameleon 的论文应把 novelty 放在 **protein-specific counterfactual structure interventions** 和 **离散结构 token 的 causal LM grounding** 上，而不是泛泛声称减少 hallucination。

## 我读到的相关工作格局

### 1. 多模态早期融合和离散 token 化

相关文件：`Chameleon_Mixed_Modal_Early_Fusion_2024.pdf`、`AnyGPT_Unified_Multimodal_LLM_Discrete_Sequence_ACL_2024.pdf`、`MM_Interleaved_Image_Text_Generative_Modeling_2024.pdf`

这些工作共同说明：把多模态输入统一成离散 token 序列，训练一个统一 autoregressive transformer，是有清晰 AI 贡献的路线。ProteinChameleon 已经走在这条路上，优势是没有额外 protein encoder/projector，而是把结构 token 放入统一词表。

不足是：这些视觉/音频多模态工作通常有强基准、强生成能力和系统级训练技巧；ProteinChameleon 当前只有 function captioning、interleaved narrative 和 perplexity，claim 还不够硬。

### 2. 蛋白结构 tokenizer 正在快速变强

相关文件：`GeoBPE_Protein_Structure_Tokenization_ICLR_2026.pdf`、`Yeti_Compact_Protein_Structure_Tokenizer_2026.pdf`、`AIDO_Balancing_Locality_Reconstruction_Protein_Structure_Tokenizer_2024.pdf`、`FoldToken2_Compact_Invariant_Generative_Protein_Structure_Language_2024.pdf`、`FoldToken_Learning_Protein_Language_AAAI_2025.pdf`、`Learning_the_Language_of_Protein_Structure_TMLR_2025.pdf`、`FoldSeek.pdf`、`ProstT5.pdf`、`SaProt_Structure_Aware_Vocabulary_ICLR_2024.pdf`

2025-2026 的重点不是“有没有结构 token”，而是：

- token 是否紧凑、可重构、可生成；
- token 是否适合 downstream LM，而不仅是 RMSD 好；
- token 是否支持 motif/domain 层级表达；
- token 是否能跨 fold、跨 domain 泛化。

ProteinChameleon 当前使用约 2100 个 PT-BPE/GeoBPE 结构 token，这是优势，因为 GeoBPE 作为 ICLR 2026 工作已经建立了强背景。但论文必须补足 tokenizer 相关分析：不同结构 token 粒度、domain span token、full-chain token、token dropout/corruption 对模型能力的影响。

### 3. 蛋白 LLM 多数仍是 encoder/projector 或 instruction tuning

相关文件：`ProteinGPT_Multimodal_LLM_Protein_Property_Structure_2024.pdf`、`EvoLlama_Multimodal_Structure_Sequence_Representations_2024.pdf`、`SEPIT_Structure_Enhanced_Protein_Instruction_Tuning_2024.pdf`、`Prot2Chat_Early_Fusion_Sequence_Structure_2025.pdf`、`STELLA_Multimodal_LLM_Protein_Functional_Annotation_2026.pdf`、`ProtT3_Protein_to_Text_Generation_ACL_2024.pdf`、`InstructProtein_Aligning_Human_Protein_Language_ACL_2024.pdf`、`ProtST_Multi_Modality_Protein_Sequences_Biomedical_Texts_ICML_2023.pdf`、`ProteinDT_Text_Guided_Protein_Design_2023.pdf`

这些工作能做很多 protein QA、captioning、function annotation，但常见范式是：

- sequence encoder 或 structure encoder 提特征；
- projector/Q-Former/adapter 接入 LLM；
- 通过 instruction tuning 输出文本。

ProteinChameleon 的差异点应该强调：

- 结构不是 encoder embedding，而是可生成、可插入、可自回归建模的离散 token；
- 同一模型可以处理任意位置的结构 span；
- 可以做结构 span 的生成、补全、反事实替换和 evidence tracing。

如果只做“结构+序列到功能文本”，STELLA、ProteinGPT、SEPIT、Prot2Chat 已经很接近，ProteinChameleon 不一定占优。

Prot2Chat 里的 structure 不是 2D 图。它修改 ProteinMPNN，把蛋白序列和结构融合进 protein encoder；结构输入是 residue atoms 的 3D coordinate structure information，具体是 ProteinMPNN 使用的 backbone atom coordinates，例如 N、Cα、C、O。它再通过 text-aware adapter 把 protein embedding 压缩成 LLM soft prompt。因此 Prot2Chat 和 ProteinChameleon 的差异不是“有没有 3D 结构”，而是：

- Prot2Chat：3D backbone coordinates -> ProteinMPNN/adapter -> soft virtual tokens；
- ProteinChameleon：3D structure -> PT-BPE/GeoBPE 离散结构 token -> 统一 causal LM 词表，可自回归生成/替换结构 span。

### 3b. Cuttlefish 是更强的相近工作，需要正面区分

相关文件：`Cuttlefish_Scaling_Aware_Adapter_Structure_Grounded_LLM_Reasoning_ICML_2026.pdf`

Cuttlefish / Scaling-Aware Adapter for Structure-Grounded LLM Reasoning 是目前最接近“structure-grounded LLM + structural hallucination”叙事的工作。它同样不是 2D 图结构，而是 all-atom spatial graph：输入包含 atom features、3D coordinates 和 spatial relations。它的 Scaling-Aware Patching 会在结构图上选择 anchor atoms 并生成 variable-size structural patches；Geometry Grounding Adapter 再用 cross-attention 把 geometric evidence 注入 LLM。

它覆盖 molecules、proteins、DNA、RNA 的 all-atom benchmark。文中 functional group hallucination 是 molecule/protein 结构 grounding 测试的一部分，不代表它只使用 2D functional-group graph。对于 protein，它评估了 protein-oriented Mol-Instructions、protein functional description、结构可用性、coordinate perturbation robustness 和 hallucination rate。

这意味着我们的推荐主线不能只写成“减少 structural hallucination 的 protein LLM”。这个 claim 已经和 Cuttlefish 高度重叠。ProteinChameleon 必须强调更具体的差异：

- **表示差异**：Cuttlefish 使用 all-atom graph encoder + adapter + variable patch soft tokens；ProteinChameleon 使用离散 PT-BPE/GeoBPE structure tokens 并扩展 causal LM 词表。
- **任务差异**：Cuttlefish 是跨 molecule/protein/DNA/RNA 的 all-atom structure-grounded reasoning；ProteinChameleon 应聚焦 protein-specific sequence-structure-function grounding。
- **评测差异**：Cuttlefish 有 structural hallucination 和 coordinate perturbation；ProteinChameleon 应做更细的 protein counterfactual structure interventions，例如 true structure vs wrong structure、same-family decoy、same-function distant decoy、domain swap/dropout、catalytic motif corruption。
- **能力差异**：Cuttlefish 的 structure tokens 是 adapter 中的软表示；ProteinChameleon 的结构 token 是可插入、可替换、可生成、可计算 perplexity 的离散语言单元。

因此更稳的论文包装应从 **Protein-MIRAGE** 调整为 **Counterfactual Structure Reliance for Protein LLMs** 或 **Protein Structure Grounding Benchmark**。MIRAGE 可以作为动机，但不应作为唯一主 claim。

### 4. ESM3 和 ProTrek 拉高了 protein multimodal model 的标准

相关文件：`ESM3_with_sm.pdf`、`ProTrek.pdf`、`ProTrek_SI.pdf`

ESM3 的强点是 sequence/structure/function 三轨 token 化和 all-to-all generation，目标很大：可控生成蛋白。ProTrek 的强点是 sequence/structure/text 三模态检索和 embedding alignment。

ProteinChameleon 不适合直接和 ESM3 拼规模。更合理的打法是做一个 ESM3/ProTrek 没有充分解决的评测和方法问题：

- 模型是否真的用结构，而不是用 sequence 或 text prior？
- 哪些结构 span 对功能判断有因果贡献？
- 何时需要插入结构 token，何时 sequence-only 已经够？
- interleaved structure-text 是否能提升结构 grounding？

### 5. MIRAGE 给了非常好的切入点

相关文件：`MIRAGE_The_Illusion_of_Visual_Understanding.pdf`

MIRAGE 的核心启发是：多模态 benchmark 的高分不等于模型真的看了图；模型可能在没有图的情况下靠文本先验答对，而且还会生成看似有证据的 reasoning trace。

迁移到 protein structure 领域就是：

> protein multimodal LLM 的 function annotation 高分，不等于模型真的使用了结构；它可能只靠序列、物种、蛋白名、UniProt 风格文本先验或训练集中常见 domain 名称。

这正好击中当前 protein LLM 论文的薄弱处。很多工作报告 function caption、GO/EC、QA，但没有系统做“去结构、错结构、乱结构、同源结构、反功能结构”的反事实测试。

### 6. Latent dynamic interleaving 给了第二条更冒险的路线

相关文件：`Reasoning_Within_the_Mind_Dynamic_Multimodal_Interleaving_in_Latent_Space.pdf`

这篇工作的核心不是蛋白，而是“不要在每一步都注入视觉信息，而是在模型不确定或需要证据时动态注入相关视觉 patch，并在 latent space 做 test-time refinement”。

迁移到蛋白结构领域，可以做：

- 不把完整结构 token 一次性塞进上下文；
- 先用 sequence/text prompt 生成初步推理；
- 当模型 uncertainty 高、或需要结构证据时，动态检索/插入相关 domain/motif 结构 token；
- 只让模型看最相关的结构 span，降低 token 成本并提升 grounding。

这条线更难，但 AI 味更强。如果做得出来，比单纯 function captioning 更像 ICLR/NeurIPS。

## 当前项目的主要风险

### 风险 1：现有评测不能证明结构被使用

`scripts/eval_stage2.py` 主要评估：

- alignment/interleaved perplexity；
- function text generation；
- BLEU、ROUGE、BERTScore。

这些指标可能只测到语言相似度。模型即使忽略结构 token，只靠 sequence 或训练分布，也可能拿到不错分数。对于 ICLR/NeurIPS，这会是核心弱点。

### 风险 2：interleaved 数据是 GPT 合成叙述，可能是语言捷径

`scripts/submit_interleaved_batch.py` 会把 `function_text`、sequence、domain 名称和 domain 序列交给 GPT-4.1，让它写 4-6 句叙述，并插入 `[IPRxxxxxx]` 占位符。`process_interleaved_batch.py` 再把占位符替换为结构 token。

这个数据有用，但不能当强 ground truth。因为模型可能学到：

- GPT 叙述模板；
- domain 名称和 function text 的共现；
- UniProt 风格描述；
- 结构 token 附近的文本模式。

需要通过反事实和 held-out split 证明模型不是在复述文本先验。

### 风险 3：数据 split 现在不够强

`prepare_alignment.py`、`prepare_domains.py`、`process_interleaved_batch.py` 使用 Python 内置 `hash(acc) % 100` 划分 train/val/test。这有两个问题：

- Python hash 默认跨进程/跨机器不稳定；
- accession 随机切分不能避免同源泄漏、family 泄漏、fold 泄漏。

如果投好会，至少需要 sequence identity 或 Foldseek/CATH/EC 层面的 held-out split。

### 风险 4：训练 packing 现在存在 attention leakage

`scripts/train_stage2.py` 的 `PackingCollator` 注释说 “Loss mask prevents cross-example attention from leaking”，但实际只把 loss mask 置零，没有阻断 causal attention。后一个样本仍然可以 attend 到同一个 packed window 中前一个样本的 token。

这不一定会毁掉模型，但论文实验前必须修。否则 train/eval perplexity 都可能被污染。

### 风险 5：没有强 baseline 和 ablation

目前缺：

- sequence-only Gemma baseline；
- structure-only baseline；
- sequence + shuffled structure；
- sequence + random same-length structure；
- sequence + domain-only structure；
- sequence + full-chain structure；
- SaProt/ProstT5/Foldseek/ESM3/ProTrek 风格 baseline；
- interleaved vs non-interleaved 的同参数对照。

没有这些，对“structure helps”和“early fusion helps”的 claim 会站不稳。

## 推荐主线：Counterfactual Structure Reliance

### Paper claim

可以把论文主 claim 写成：

> We introduce a counterfactual evaluation and training framework for structure-grounded protein language models. We show that high protein function annotation scores can arise from sequence/text priors without genuine structure use, and we train an early-fusion structure-token LLM that remains sensitive to causally relevant protein structure spans under controlled interventions.

中文说法：

> 我们提出一个面向蛋白结构-文本模型的反事实 grounding 框架，证明传统 caption/QA 分数会高估结构理解，并通过早期融合结构 token 训练让模型对真实结构证据更敏感、对错误结构更鲁棒。

### 需要新增的数据构造

在现有 alignment 样本基础上，为每个蛋白构造多种 prompt：

1. **Seq-only**
   - 输入 organism + sequence，不给结构。

2. **Struct-only**
   - 输入 organism + `<PROT_START> structure <PROT_END>`，不提供 sequence。

3. **Seq + true struct**
   - 当前 alignment prompt。

4. **Seq + shuffled struct**
   - sequence 保持不变，结构换成随机蛋白或同长度蛋白。

5. **Seq + same-family decoy struct**
   - 换成同 family 但功能不同或 EC 不同的结构。

6. **Seq + same-function decoy struct**
   - 换成远同源但功能相同的结构，用来区分“结构形状”和“标签泄漏”。

7. **Domain dropout**
   - 只给部分 domain token，或删除关键 catalytic/binding domain。

8. **Domain swap**
   - 把某个 InterPro domain span 替换为另一个 domain 的结构 token。

9. **Structure corruption**
   - token span 局部打乱、mask、替换 motif token、替换 angle bin。

10. **No-structure mirage prompt**
   - 明确不提供结构，但问模型“based on the structure”；检测它是否虚构结构证据。

这些构造可以从现有 `/data2/steven/data/stage2/alignment/*.npz`、InterPro domain 数据和 structure tokens 出发完成，不必一开始重做全部数据。

### 需要新增的训练目标

建议在 Stage 2 上加一个 Stage 2.5，目标不是继续混合训练，而是结构 grounding 训练：

1. **Modality dropout**
   - 随机 drop sequence、drop structure、drop organism、drop domain spans；
   - 让模型学会在模态缺失时校准不确定性。

2. **Counterfactual consistency / inconsistency loss**
   - true structure prompt 应该输出正确功能；
   - shuffled/corrupted structure prompt 应该输出低置信度、拒答、或指出结构不一致；
   - 对 MCQA 形式可以直接做 contrastive loss。

3. **Structure span denoising**
   - 给 corrupted domain token，让模型恢复原始结构 token 或判断 corruption 类型；
   - 这能避免结构 token 只作为无意义分隔符存在。

4. **Evidence token supervision**
   - 要求模型输出支持判断的 domain/motif span ID；
   - 可以先用 InterPro domain、EC catalytic annotations、binding site annotations 做弱监督。

5. **Interleaved structure placement**
   - 不只让 GPT 决定结构 span 位置；
   - 加一个任务：给文本叙述和候选 domain tokens，让模型决定在哪里插入哪个结构 span。

### 需要新增的评测指标

除了 BLEU/ROUGE/BERTScore，至少加以下指标：

1. **Structure Reliance Index, SRI**
   - 例：`SRI = score(seq+true_struct) - score(seq+shuffled_struct)`；
   - 如果 true 和 shuffled 差不多，说明模型没用结构。

2. **Structure Gain**
   - `score(seq+true_struct) - score(seq_only)`；
   - 衡量结构带来的净收益。

3. **Mirage Rate**
   - 无结构输入时，模型声称“based on structure”并给出具体结构证据的比例；
   - 直接借鉴 MIRAGE 的核心思想。

4. **Counterfactual Sensitivity**
   - domain swap、catalytic motif corruption 后，模型预测是否发生合理变化。

5. **Calibration / Abstention**
   - 错结构或缺结构时，模型是否降低置信度或承认信息不足。

6. **Evidence F1**
   - 模型指出的 domain/motif 是否和 InterPro、EC、binding site、active site 注释一致。

7. **Structure-token generation validity**
   - 如果生成结构 token，检查是否能 decode 成合理 backbone；
   - 用 RMSD/TM-score/pLDDT/ProteinMPNN sequence recovery/novelty/designability 等指标。

### 需要新增的 baseline

最低限度：

- Gemma text-only，输入 organism + sequence；
- ProteinChameleon sequence-only；
- ProteinChameleon structure-only；
- ProteinChameleon seq + true structure；
- ProteinChameleon seq + shuffled structure；
- ProteinChameleon no interleaved training；
- ProteinChameleon no warmup；
- ProteinChameleon full-chain structure vs domain-only structure；
- ProteinChameleon PT-BPE/GeoBPE token vs Foldseek 3Di token。

更强一点：

- SaProt / ProstT5 表征 + 线性/MLP classifier；
- Foldseek nearest-neighbor retrieval baseline；
- ESM2/ESM3-open sequence/structure baseline；
- ProTrek-style retrieval baseline，如果能跑；
- ProteinGPT/Prot2Chat/STELLA 如果代码或 checkpoint 可用，至少引用和定性比较。

### 为什么这条线适合 ICLR/NeurIPS/ICML

优点：

- 有明确 AI 问题：multimodal grounding and counterfactual evaluation；
- 有跨领域新意：把 MIRAGE 从视觉迁移到 protein structure；
- 有方法贡献：early-fusion discrete structure-token LM + grounding training；
- 有 benchmark 贡献：protein-specific counterfactual structure reliance 评测；
- 不需要和 ESM3 拼模型规模。

风险：

- 需要构造严谨的 held-out split；
- 需要证明 counterfactual 不是简单 distribution shift；
- 需要至少一个强 baseline；
- 如果只在 UniProt function text 上评估，会被认为生物任务太窄。

## 备选路线 A：Dynamic Protein Structure Interleaving

这是更偏 ICLR 的方法路线，借鉴 `Reasoning_Within_the_Mind_Dynamic_Multimodal_Interleaving_in_Latent_Space.pdf`。

核心想法：

> 不要把完整结构一直塞进上下文，而是让模型在需要结构证据时动态选择 domain/motif structure tokens 注入推理过程。

具体做法：

1. 先用 sequence/text prompt 生成初步 hidden state 或短推理。
2. 根据 token entropy、answer margin 或 learned gate 判断是否需要结构。
3. 从候选 domain/motif tokens 中检索最相关 span。
4. 把选中的 `<PROT_START> span <PROT_END>` 注入上下文或 latent prefix。
5. 继续生成答案。

可以提出：

- Dynamic Structure Injection, DSI；
- Protein latent think tokens；
- confidence-guided structure retrieval；
- token-budget-aware protein reasoning。

评测：

- 同样 accuracy 下结构 token 使用量下降多少；
- 同样 token budget 下是否优于 full-chain structure；
- 选择的 domain 是否和已知功能 domain 对齐；
- wrong-structure injection 时是否能拒绝。

难点：

- 需要改模型推理代码；
- 如果做 latent optimization，需要访问和稳定操控 hidden state；
- 工程风险比 counterfactual structure reliance 主线高。

适合作为主线的增强组件，而不是第一版 paper 的唯一核心。

## 备选路线 B：Any-to-any protein sequence-structure-text generation

借鉴 ESM3、Chameleon、AnyGPT、Yeti，把 ProteinChameleon 从 function annotation 扩展为：

- sequence -> structure tokens；
- structure tokens -> sequence；
- text -> sequence + structure；
- partial structure/text -> complete protein narrative；
- masked domain structure infilling；
- interleaved text+structure continuation。

这条线最像“protein Chameleon”，但要求很高：

- 必须能把生成的结构 token decode 回 backbone；
- 必须评估 designability、novelty、TM-score、RMSD、pLDDT；
- 需要 ProteinMPNN/ESMFold/AlphaFold 或等价 pipeline 做验证；
- 需要和 ESM3-open、DPLM、Yeti、GeoBPE SSLM 等生成模型比较。

如果已有 PT-BPE/GeoBPE decoder 能稳定 reconstruct backbone，这条线可以冲；如果不能，短期不建议把它作为主投稿核心。

## 备选路线 C：结构 tokenizer 和 LLM 兼容性研究

可以做一个偏 tokenizer 的 paper：

> Which protein structure tokens are language-model friendly?

比较：

- PT-BPE/GeoBPE；
- Foldseek 3Di；
- FoldToken/FoldToken2；
- Yeti-like LFQ；
- residue-level angle bins；
- domain-level BPE spans。

指标：

- reconstruction；
- structure-token perplexity；
- downstream function annotation；
- counterfactual sensitivity；
- token efficiency；
- long-context efficiency；
- LM codebook utilization。

这条线科学上扎实，但如果没有新 tokenizer 方法，只做评测可能更适合 benchmark/workshop 或 TMLR，而不是顶会主会。

## 建议的最小可执行版本

### 第 1 周：修基础可信度

1. 修 `PackingCollator` 的 cross-example attention leakage。
   - 简单方案：不 pack 不同样本，先牺牲效率换可信实验；
   - 更好方案：实现 block-diagonal causal mask 或用 FlashAttention varlen packing。

2. 改 split。
   - 用稳定 hash，例如 SHA256；
   - 加 sequence identity cluster split；
   - 加 Foldseek/CATH/domain-family split。

3. 加 baseline 数据模式。
   - `seq_only`；
   - `struct_only`；
   - `seq_true_struct`；
   - `seq_random_struct`；
   - `seq_same_length_struct`；
   - `seq_domain_dropout`。

4. 固化评测输出。
   - 每个样本保存 prompt type、accession、structure source、decoy accession、domain corruption 类型。

### 第 2-3 周：做 counterfactual structure reliance benchmark

1. 生成反事实 test set。
2. 写 `scripts/prepare_counterfactual_eval.py`。
3. 写 `scripts/eval_grounding.py`。
4. 增加 MCQA 版本，降低开放生成文本评分的不确定性。
5. 用现有 checkpoint 先跑：
   - seq-only；
   - seq+true；
   - seq+shuffled；
   - no-structure mirage。

这里会很快看出项目有没有真实信号。如果 `seq+true` 明显好于 `seq+shuffled`，而 no-structure mirage 低，就是强结果。如果差异不大，说明需要改训练。

### 第 4-6 周：做 grounding training

新增 Stage 2.5：

- true/corrupted pair training；
- modality dropout；
- domain dropout；
- structure denoising；
- refusal/uncertainty supervision。

训练不一定要全量 Gemma full fine-tune。可以先做 LoRA 或只训练新增结构 token + top layers，降低成本并支持更多 ablation。

### 第 7-8 周：强 baseline 和论文图表

必须有这些表：

1. Main benchmark table：
   - sequence-only、true structure、random structure、same-family decoy、domain dropout。

2. Mirage table：
   - 无结构输入时模型是否虚构结构证据。

3. Ablation table：
   - warmup、interleaved training、counterfactual training、domain dropout、structure denoising。

4. Tokenizer table：
   - PT-BPE/GeoBPE vs 3Di vs residue-level tokens。

5. Efficiency table：
   - full-chain structure vs domain-only vs dynamic injection。

6. Case studies：
   - 一个 enzyme active site；
   - 一个 multi-domain protein；
   - 一个结构相似但功能不同的 decoy；
   - 一个 sequence homolog 但结构/domain 不同的反例。

## 可以写成的论文题目

更稳的题目：

- **Counterfactual Structure Reliance in Protein Language Models**
- **Do Protein Language Models Really Use Structure? Counterfactual Evaluation with Discrete Structure Tokens**
- **PSG-Bench: A Counterfactual Benchmark for Protein Structure Grounding**
- **Do Protein Language Models Really Use Structure? A Counterfactual Study with Early-Fusion Structure Tokens**
- **Structure-Grounded ProteinChameleon: Early-Fusion Protein Language Modeling with Counterfactual Structure Interventions**

更方法导向的题目：

- **Dynamic Structure Interleaving for Grounded Protein Language Modeling**
- **When Should Protein LMs Look at Structure? Confidence-Guided Structural Token Injection**

更生成导向的题目：

- **A Unified Discrete Token Language for Protein Sequence, Structure, and Function**

我最推荐前两个。它们的 claim 更尖，也更能避开 Cuttlefish 已经覆盖的泛 structure-grounded hallucination 叙事。

## 投稿目标判断

### ICLR / NeurIPS / ICML

可行条件：

- 有明确的新 benchmark 或新 method；
- 反事实实验很扎实；
- baseline 充分；
- 证明不是普通 captioning；
- 有 scaling/ablation；
- 有开源数据构造和评测脚本。

最适合的包装是 counterfactual structure reliance benchmark + ProteinChameleon method。

### KDD / ACL / EMNLP Findings

如果主要是 instruction tuning、protein-text generation、function annotation，更适合 ACL/EMNLP/KDD 生物 NLP 方向。但这些 venue 对新方法要求可能比 ICLR 稍低，对数据/应用也比较接受。

### Nature Machine Intelligence / Nature Communications / Bioinformatics / Briefings in Bioinformatics

如果实验能证明对真实生物任务有用，例如新功能注释、remote homolog/domain function transfer、enzyme annotation、结构证据解释，这些期刊也可以考虑。

Bioinformatics 更偏应用和工具，要求比 ICLR 顶会低一些，但必须有可复现 pipeline 和明确生物学价值。

## 代码层面建议清单

优先级 P0：

- 修 packing attention leakage；
- 改稳定 split；
- 加 seq-only/struct-only/shuffled-struct eval；
- 加 counterfactual eval 数据生成脚本；
- 把 BLEU/ROUGE 之外的 MCQA、classification、calibration 指标加上。

优先级 P1：

- 加 modality dropout training；
- 加 domain dropout/corruption training；
- 加 structure-denoising objective；
- 加 evidence span 输出；
- 加 tokenizer ablation。

优先级 P2：

- 加 dynamic structure injection；
- 加 latent think tokens；
- 加 structure token decoding 和 designability pipeline；
- 加 retrieval-augmented Foldseek/ProTrek baseline。

## 最后的判断

ProteinChameleon 当前最值得保留的是“统一 causal LM + 离散结构 token + interleaved structure-text”。最需要改变的是论文问题定义。

不要把 paper 写成：

> 我们把蛋白结构 token 加到 Gemma 里，然后生成 function text。

应该写成：

> 当前 protein multimodal LLM 的高分可能来自语言和序列先验。我们提出 protein-specific counterfactual structure reliance 评测，系统测量模型是否真正使用结构；并提出 ProteinChameleon，一个早期融合的结构-token causal LM，通过反事实训练和结构 span interleaving 显著提升真实结构依赖、降低错误结构诱导的虚假推理，并在 held-out fold/domain/function split 上优于 sequence-only、projector-based 和 retrieval baselines。

这个版本才有比较清晰的 ICLR 级别故事。

# ProteinChameleon 文献综述

按照你的描述，ProteinChameleon 位于一个非常具体的交叉点上：它是一个 **decoder-only causal language model**，使用一个同时覆盖自然语言和蛋白相关符号的 **统一词表**，显式加入 **离散蛋白结构 tokens**，并采用 **两阶段训练**：第一阶段先学习新增蛋白结构 token 的 embedding 和输出头；第二阶段再混合结构条件下的蛋白功能监督数据，以及带有内联蛋白结构 token span 的科学文本。

目前最接近的文献通常并没有把这些要素全部合在一起。最接近的先例一般只贡献其中一两个核心思想：蛋白-文本交错建模、离散蛋白结构 tokenization、序列-结构-功能多模态学习，或者自由文本式的功能生成。

最重要的高层结论是：**ProteinChameleon 最像 ProtLLM、ESM3、SaProt，以及 FoldToken、GeoBPE、Yeti 等离散蛋白结构 tokenizer 工作的混合体**。ProtLLM 提供了 **LLM 内部交错建模蛋白和文本流** 的思路；ESM3 提供了 **序列、结构和功能都可以被转成离散 token track 并放进一个模型中** 的思路；SaProt 说明即使是简单的 **结构感知词表** 也能显著提升蛋白学习；FoldToken、GeoBPE、Yeti 这类 tokenizer 论文则说明：**离散结构字母表的质量很可能是多模态蛋白生成的瓶颈**。

## A. 最接近的 prior work

### ProtLLM

**Le Zhuo, Zewen Chi, Minghao Xu, Heyan Huang, Jianan Zhao, Heqi Zheng, Conghui He, Xian-Ling Mao, and Wentao Zhang. “ProtLLM: An Interleaved Protein-Language LLM with Protein-as-Word Pre-Training.” ACL 2024.**

ProtLLM 是在 **训练格式** 上最接近 ProteinChameleon 的论文。它是一个跨模态 LLM，能够处理 **交错的蛋白-文本输入和输出**，并提出了 **protein-as-word** 训练范式以及 **InterPT** 语料库。InterPT 由结构化蛋白注释数据和非结构化生物学论文共同构建。

它并不是严格意义上的“文本 + 结构 token 统一 decoder”，但它最直接地验证了一个想法：蛋白 span 可以被嵌入普通语言建模流中，而不是只能通过 late fusion 的方式处理。这对你的第二阶段数据，也就是“科学文本中带有 inline protein span”的设定特别相关。

它和 ProteinChameleon 的主要差距是：ProtLLM 的“protein token”不是 residue-level 或 structure-token sequence，也不是和文本 token 共享同一个底层词表的结构 token 序列。它依赖的是 protein mounting mechanism 和一种针对蛋白 identity 的专用 protein vocabulary。

**相关性：5/5。**

可以借鉴的点：InterPT 式数据构建、结构化/非结构化混合训练、显式支持任意交错形式。

仍然存在的差距：没有 PT-BPE/GeoBPE 风格的结构字母表，也没有在同一个 residue-aligned stream 中自由生成文本和结构 token。

### ESM3

**T. Hayes et al. “Simulating 500 million years of evolution with a language model.” Science 2025.**

ESM3 不是 decoder-only，但在概念上是 ProteinChameleon 最强的先例之一，因为它把 **序列、结构和功能视为一个模型内部的离散 tokenized modalities**。EvolutionaryScale 把 ESM3 描述为一个 **multi-track transformer**，其词表桥接序列、结构和功能，并通过 mask 和预测部分观测到的多模态 tokenized proteins 来训练。

他们明确指出，ESM3 必须把三维结构和功能都转换成离散字母表，并且模型可以在给定部分序列、结构、功能约束的情况下生成蛋白。这和 ProteinChameleon 的“所有模态都是 token”的哲学非常接近，虽然 ESM3 的训练目标是 masked multimodal generation，而不是 Gemma 风格的 causal LM。

**相关性：5/5。**

可以借鉴的点：multi-track prompting、离散结构/功能 tokenization、用合成数据增强稀缺的结构/功能标签、多模态可控生成。

仍然存在的差距：没有自然语言文档级 interleaving，也没有一个跨自由文本和蛋白结构 token 的 decoder-only 统一 token stream。

### SaProt

**Jin Su, Chenchen Han, Yuyang Zhou, Junjie Shan, Xibin Zhou, and Fajie Yuan. “SaProt: Protein Language Modeling with Structure-Aware Vocabulary.” ICLR 2024.**

SaProt 提出了一种 **structure-aware vocabulary**，把氨基酸残基 identity 和 Foldseek 派生的结构 token 结合起来，使得每个输入 token 已经同时融合序列和局部结构信息。它使用 ESM 风格的 encoder，并采用 BERT 风格的 masked objective，在大约 **4000 万个 sequence-structure pair** 上训练，并在 **10 个下游任务** 上报告了提升。

对 ProteinChameleon 来说，SaProt 很重要，因为它证明了：仅仅通过 **词表层面的改造**，就可以改变蛋白模型的 inductive bias，并带来实际收益。它也是一个提醒：即使没有复杂的多模态架构，structure-aware token 也能起作用。因此 ProteinChameleon 的价值很大程度上取决于你的 PT-BPE/GeoBPE token 是否比 3Di 风格的局部结构字母表编码了更丰富的结构信息。

**相关性：4/5。**

可以借鉴的点：残基 + 结构 token 融合、对低置信度结构区域进行 pLDDT-aware masking、大规模 AlphaFoldDB 预训练。

仍然存在的差距：没有文本模态，没有 autoregressive generation，没有 inline multimodal documents；结构被折叠进 residue token，而不是作为一个独立可生成的 token language 保留下来。

### ProTrek

**Jin Su, Yan He, Shiyang You, Shiyu Jiang, Xibin Zhou, Xuting Zhang, Yuxuan Wang, Xining Su, Igor Tolstoy, Xing Chang, Hongyuan Lu, and Fajie Yuan. “A trimodal protein language model enables advanced protein searches.” Nature Biotechnology 2025.**

ProTrek 通过 **trimodal contrastive learning** 统一 **序列、结构和自然语言功能描述**，实现任意两种模态之间的检索，并扩展到超过 **50 亿个蛋白** 的 embedding。

这很相关，因为 ProteinChameleon 的 alignment dataset，也就是“organism + amino acid sequence + structure tokens → function text”，本质上可以看作 ProTrek retrieval-oriented sequence-structure-function alignment 的生成式版本。换句话说，ProTrek 强烈说明存在一个 trimodal latent geometry；ProteinChameleon 的假设是，同样的 alignment 可以直接在一个 unified-token causal decoder 中实现。

**相关性：4/5。**

可以借鉴的点：强 sequence-structure-text alignment loss、任意模态对之间的 retrieval evaluation、大规模 embedding-based search 作为辅助 benchmark。

仍然存在的差距：ProTrek 不是生成式模型，不做文档 interleaving，也不把离散结构 token language 暴露给 decoder。

### FoldToken 和 FoldGPT

**Zhiyan Gao et al. “FoldToken: Learning Protein Language via Vector Quantization and Beyond.” AAAI 2025.**

FoldToken 是把蛋白结构转换成 **离散语言** 并用于 **GPT-style generation** 的最清晰先例之一。论文明确提出了一个 tokenizer，把 protein sequence-structure 投影到联合离散空间中，然后训练 **FoldGPT** 进行 sequence-structure co-generation。

这对 ProteinChameleon 非常核心，尤其是如果你的结构 token 不只是作为 conditioning context，而是也要作为可生成的输出空间。

**相关性：5/5。**

可以借鉴的点：VQ-trained structure language、joint sequence-structure code space、先做 structure-token warmup 再进入下游多模态使用。

仍然存在的差距：没有自然语言文本，没有蛋白功能 caption，也没有 interleaved scientific-document modeling。

## B. 蛋白结构 tokenization / structural alphabets

### Foldseek 和 3Di

**Milot Mirdita, Martin Steinegger, and colleagues. “Fast and accurate protein structure search with Foldseek.” Nature Biotechnology 2023.**

Foldseek 很基础，因为它的 **3Di alphabet** 成为后来很多蛋白模型使用的结构离散化方法。3Di 不是只编码局部 backbone conformation，而是描述每个残基与其空间最近邻之间的几何构象，形成一个 **20-state discrete structural alphabet**，主要用于快速结构检索。

这个表示对搜索非常强，而且足够简单，可以直接放进 language-model pipeline。ProteinChameleon 应该把 3Di 作为需要击败的 baseline structural alphabet，尤其是在 retrieval 和局部结构语义上。

**相关性：4/5。**

可以借鉴的点：用 retrieval、信息密度、对 noisy structures 的鲁棒性来评估 token 质量。

仍然存在的差距：3Di 是为搜索优化的，不一定适合长程生成一致性，也不一定适合 text-conditioned structure generation。

### ProstT5

**Michael Heinzinger et al. “Bilingual language model for protein sequence and structure.” NAR Genomics and Bioinformatics 2024.**

ProstT5 表明，氨基酸序列和 Foldseek 风格的结构字符串可以被当作两种“语言”，并在两者之间互相翻译。它尤其说明 structural alphabet 不仅可以用于搜索，也可以 **从序列中预测出来**，并用于 remote homology detection 等下游任务。

对 ProteinChameleon 来说，ProstT5 支持这样一个观点：离散结构字符串可以更像文本一样处理，而不是一定要被当作图或坐标来处理。

**相关性：4/5。**

可以借鉴的点：sequence ↔ structure translation task、用预测结构 token 做数据增强、评估 homology structure string 的 alignment sensitivity。

仍然存在的差距：没有文本模态，没有自由形式 annotation generation，也没有与自然语言共享的统一词表。

### Learning the Language of Protein Structure

**Benoit Gaujac et al. “Learning the Language of Protein Structure.” MLSB 2024 workshop / public preprint.**

这篇论文提出了一个用于蛋白结构的 **vector-quantized autoencoder**，codebook 大小从 **4096 到 64000 tokens**，并展示一个简单的 GPT 在这些 token 上训练后，可以生成 **novel, designable protein structures**。

它不是 protein-text 论文，但它清楚地证明了：蛋白坐标可以被压缩成支持 autoregressive modeling 的离散符号流。

**相关性：5/5。**

可以借鉴的点：先训练 tokenizer，再在 frozen codes 上训练 GPT；同时报告 reconstruction RMSD 和 codebook utilization；先测试一个小 causal LM 是否能学习 protein-only structure syntax，再引入文本。

仍然存在的差距：没有 sequence/text alignment，也没有和科学文本的 residue-level interleaving。

### FoldToken2 和 AIDO.StructureTokenizer

**Zhiyan Gao et al. “FoldToken2: Learning compact, invariant and generative protein structure language.” 2024 preprint.**

FoldToken2 在 FoldToken 基础上改进了 invariant encoder、更强的 vector quantization，以及更好的 equivariant decoding，并报告了对单链和多链结构更好的重建效果。

**Jiayou Zhang et al. “Balancing Locality and Reconstruction in Protein Structure Tokenizer.” 2024 preprint / AIDO.StructureTokenizer.**

AIDO.StructureTokenizer 明确研究了 **token locality** 和 **reconstruction quality** 之间的 trade-off，并报告说，更好的平衡可以提升与蛋白 sequence LM 的 alignment，以及下游 structure prediction accuracy。

这两篇论文一起强调了一个容易被忽视的点：**最适合 reconstruction 的 tokenizer，不一定最适合 multimodal alignment 或 generative downstream use**。这对 ProteinChameleon 非常关键。你的 PT-BPE/GeoBPE tokenizer 不应该只看 reconstruction，还应该评估 function-text conditioning、interleaving robustness 和 causal next-token predictability。

**相关性：4/5。**

可以借鉴的点：围绕 codebook size、locality、downstream text alignment 做 tokenizer ablation；使用 retrieval-based tokenizer evaluation；显式报告 token diversity 和 codebook utilization。

仍然存在的差距：这些论文都没有把自然语言监督直接整合进 tokenizer 本身。

### GeoBPE 和 Yeti

**Michael Sun, Weize Yuan, Gang Liu, Wojciech Matusik, and Marinka Zitnik. “Protein Structure Tokenization via Geometric Byte Pair Encoding.” ICLR 2026.**

GeoBPE 是与你的 PT-BPE / GeoBPE tokenizer 最直接相关的论文之一。它提出了一个 **hierarchical geometry-grounded BPE-style vocabulary**，由几何 primitive 组成，强调可解释性和多尺度控制，并在 **12 个任务和 24 个测试 split** 上报告提升；与 transformer 结合时，还可以做 unconditional backbone generation。

**Nabin Giri, Steven Farrell, and Kristofer E. Bouchard. “Yeti: A compact protein structure tokenizer for reconstruction and multi-modal generation.” arXiv 2026.**

Yeti 是另一个极其相关的 tokenizer，因为它明确为 **multimodal generation** 优化，而不仅仅是 reconstruction。它使用 lookup-free quantization 和 flow matching，报告了较高的 token diversity 与 codebook utilization，并展示了一个 compact multimodal model 可以从零开始联合生成合理的序列和结构。

这两篇可能是评估 PT-BPE/GeoBPE 设计选择时最直接的参考。

**相关性：二者均为 5/5。**

可以借鉴的点：hierarchical BPE-style structure vocabulary、可控尺度的 structural phrases、显式的 interpretability study，以及把“multimodal utility”作为 tokenizer 的真正目标。

仍然存在的差距：二者都没有像 ProteinChameleon 一样，在 decoder-only LM 中把结构 token 与自然语言文档交错起来。

## C. 多模态 protein-text-structure models

### ProtST

**Minghao Xu, Xinyu Yuan, Santiago Miret, and Jian Tang. “ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts.” ICML 2023.**

ProtST 从 Swiss-Prot 构建 **ProtDescribe** 数据集，并使用 sequence + biomedical text 进行预训练，目标包括 unimodal mask prediction 和 multimodal alignment。它没有 structure token，也不是 ProteinChameleon 意义上的生成式模型，但它是 protein-text alignment 领域的重要早期工作。

**相关性：3/5。**

可以借鉴的点：精心构建 sequence-text property descriptions、多模态 mask prediction、protein-text retrieval 作为 sanity check。

仍然存在的差距：没有结构 token，没有 decoder-only generation，也没有普通科学文本中的 inline protein spans。

### ProteinGPT

**Yijia Xiao, Edward Sun, Yiqiao Jin, Qifan Wang, and Wei Wang. “ProteinGPT: Multimodal LLM for Protein Property Prediction and Structure Understanding.” arXiv 2024.**

ProteinGPT 使用 sequence encoder、structure encoder、projection layer 和 LLM，并在一个由 RCSB-PDB 描述和 GPT 生成 QA 对构建的 **132,092 个蛋白 instruction-tuning 数据集** 上训练。

它对 ProteinChameleon 的重要性不在于架构接近，而在于数据和任务设计：它展示了一种把蛋白 sequence/structure record 转换成 **instruction-following QA 数据** 的方法。

**相关性：3/5。**

可以借鉴的点：从结构化蛋白描述中生成 QA-style supervision；在 instruction tuning 前先做 alignment stage；使用显式多模态 prompt。

仍然存在的差距：ProteinGPT 使用来自 frozen encoder 的 soft prompts，而不是统一 token vocabulary，因此没有检验“everything is tokens”这个更难但更优雅的假设。

### Prot2Chat 和 EvoLlama

**Zeyuan Wang et al. “Prot2Chat: Protein LLM with Early Fusion of Text, Sequence, and Structure.” Bioinformatics 2025.**

Prot2Chat 使用改造过的 ProteinMPNN-style encoder、text-aware adapter 和 LLaMA-family decoder。它在 **Mol-Instructions** 和 **UniProtQA** 上评估，指标包括 **BLEU-2、ROUGE-1/2/L**，以及类似人类排序的 KIMI evaluation；模型在 Mol-Instructions 上明显超过 baseline，并且在 UniProtQA 上微调后进一步提升。

**Ningwei Liu et al. “EvoLlama: Enhancing LLMs’ Understanding of Proteins via Multimodal Structure and Sequence Representations.” arXiv 2024.**

EvoLlama 结合 **ProteinMPNN structure encoder**、**ESM-2 sequence encoder**、multimodal projector 和 **Llama-3 decoder**，在蛋白 instruction data 和 verbalized property prediction tasks 上训练。

这两篇论文说明，sequence + structure + text fusion 已经可以在开放式 protein QA 和 explanation task 上有竞争力。

**相关性：Prot2Chat 4/5，EvoLlama 3/5。**

可以借鉴的点：open-ended QA benchmark、verbalized property prediction、early-fusion conditioning。

仍然存在的差距：二者都依赖 encoder/projector pipeline，而不是一个真正统一的、能够生成结构 token 的离散词表。

### SEPIT 和 STELLA

**Wei Wu et al. “Structure-Enhanced Protein Instruction Tuning: Towards General-Purpose Protein Understanding with LLMs.” accepted to KDD 2025.**

SEPIT 给 protein LM 加入 structure-aware module，先用 **contrastive learning 和 structure denoising** 进行 warmup，然后再做 instruction tuning，覆盖开放式生成和 closed-set answering。

**Hongwang Xiao et al. “STELLA: A Multimodal LLM for Protein Functional Annotation via Unified Sequence-Structure Encoding.” Findings of ACL 2026.**

STELLA 使用 **ESM3 作为统一 sequence-structure encoder**，并使用 **Llama-3.1-8B-Instruct** 来做 functional description 和 enzyme-reaction prediction。它的核心主张是 multimodal LLM 可以在 function annotation 上超过 sequence-only pLM。

这两篇对 ProteinChameleon 的 **应用目标** 非常接近：它们都关注 structure-informed function text generation。虽然它们采用的是更常见的 “encoder + projector + LLM” 路线。

**相关性：STELLA 4/5，SEPIT 3/5。**

可以借鉴的点：staged training、structure denoising、针对异质功能输出的 mixture-of-experts 或 specialized heads、把 “functional description prediction” 作为核心 benchmark。

仍然存在的差距：二者都没有学习一个可以嵌入普通科学文本中的 residue-level discrete structure language。

## D. 蛋白功能注释与 protein captioning

### Prot2Text

**Hadi Abdine, Michail Chatzianastasis, Costas Bouyioukos, and Michalis Vazirgiannis. “Prot2Text: Multimodal Protein’s Function Generation with GNNs and Transformers.” AAAI 2024.**

Prot2Text 是 **structure-conditioned free-text function generation** 最直接的先例之一。它把 **AlphaFold-derived protein graph 上的 RGCN** 与 **ESM sequence encoder** 结合，然后用 **GPT-2** 解码功能描述。

论文构建了一个包含 sequence、AlphaFold structure 和 text description 的 multimodal Swiss-Prot 数据集，并使用 **BLEU、ROUGE-1/2/L、BERTScore** 评估生成效果；其多模态版本优于仅用 sequence 或仅用 graph 的 baseline。

**相关性：5/5。**

可以借鉴的点：自由文本功能生成，而不只是 label prediction；graph + sequence fusion baseline；按与训练数据的 sequence similarity 分层评估。

仍然存在的差距：Prot2Text 使用连续 encoder fusion，而不是离散结构 token，也没有 interleaved document modeling。

### ProtT3

**Zhiyuan Liu, An Zhang, Hao Fei, Enzhi Zhang, Xiang Wang, Kenji Kawaguchi, and Tat-Seng Chua. “ProtT3: Protein-to-Text Generation for Text-based Protein Understanding.” ACL 2024.**

ProtT3 通过 **Q-Former** 把 protein LM 和 text LM 连接起来，并形式化了三个 protein-text task：**protein captioning、protein question answering、protein-text retrieval**。

它非常相关，因为它把 “protein-to-text generation” 作为中心问题，而不是把它当作分类任务的附属产物。

**相关性：4/5。**

可以借鉴的点：把 free-text generation task 作为主要 benchmark family；同时纳入 captioning、QA 和 retrieval；把 Q-Former / projector baseline 作为 unified-token model 的重要对照。

仍然存在的差距：protein 输入是 sequence-only，没有离散 structure-token modality。

### InstructProtein

**Zeyuan Wang, Qiang Zhang, Keyan Ding, Ming Qin, Xiang Zhuang, Xiaotong Li, and Huajun Chen. “InstructProtein: Aligning Human and Protein Language via Knowledge Instruction.” ACL 2024.**

InstructProtein 是一个双向模型，可以做 **protein sequence → textual function description**，也可以做 **text prompt → protein sequence**。它先在蛋白和自然语言语料上预训练，然后用 **knowledge-graph-driven instruction tuning** 对齐。

它对 ProteinChameleon 的价值在于：它把蛋白和文本都视为可以通过 instruction data 对齐的语言，而不仅仅依赖 contrastive learning。

**相关性：4/5。**

可以借鉴的点：KG-grounded instruction generation、功能解释的 causal modeling、双向评估。

仍然存在的差距：没有结构 token，也没有 interleaved text + protein document generation。

### FAPM 和 ProteinDT

**Wenkai Xiang, Zhaoping Xiong, and Huan Chen. “FAPM: functional annotation of proteins using multimodal models.” Bioinformatics 2024.**

FAPM 把 pretrained protein sequence model 和 LLM 连接起来，用 **GO-term text** 对齐 sequence representation，并生成自然语言功能标签。它在 Swiss-Prot 和 phage-protein annotation 上报告了较强性能，并且可以使用 taxonomy prompt。

**Shengchao Liu et al. “A Text-guided Protein Design Framework.” Nature Machine Intelligence 2025.**

ProteinDT 不是 annotation 模型，而是反方向任务：它对齐 protein 和 text representation，然后根据 **textual descriptions 生成 protein sequences**。它使用 **SwissProtCLAP** 数据集，其中包含 **441K text-protein pairs**。

这两篇一起说明：**function text 既可以作为监督信号，也可以作为控制信号**。

**相关性：FAPM 4/5，ProteinDT 3/5。**

可以借鉴的点：GO-term text generation、可选的 taxonomy/organism prompts、把 text ↔ protein 双向任务放进 benchmark suite。

仍然存在的差距：二者都没有达到“统一 decoder 同时处理文本和离散结构 token”的程度。

## E. 一般多模态 LLM / unified-token 启发

ProteinChameleon 的更广义设计强烈类似于生物学之外的 **unified-token multimodal LLM**。最明显的启发是 **Chameleon**：它使用 **early-fusion token-based mixed-modal modeling**，可以理解和生成任意顺序交错的 image-text sequence。

**AnyGPT** 从更抽象的角度提出了同样观点：如果所有模态都能被离散化成 token sequence，那么一个标准 LLM 就可以主要通过数据和 tokenization 来扩展到这些模态，而不一定需要为每个模态设计专门架构。

**MM-Interleaved** 也直接相关，因为它把 interleaved multimodal documents 当作原生训练格式，而不是只处理单个 paired input。

这些工作共同支持 ProteinChameleon 最强的概念主张：**新模态可以通过变成词表中的“另一种语言”来加入 autoregressive LM**。

在蛋白特定的 causal LM 背景中，重要的 sequence-only 先例是 **ProtGPT2** 和 **ProGen2**。ProtGPT2 证明，只在蛋白序列上训练 GPT-style autoregressive transformer，也能生成在序列统计和 AlphaFold 评估下看起来像蛋白的 de novo proteins。ProGen2 把这个范式扩展到 **最高 6.4B 参数**，并在 generation 和 zero-shot fitness prediction 上表现很强。

这些论文的重要性在于：它们说明 causal protein LM 在 sequence space 中已经有效；ProteinChameleon 可以被看作把这种成功扩展到一个更丰富的 token language，其中包括文本和结构。

在 structure-aware generative background 中，最相关的 inverse-folding 参考是 **ProteinMPNN** 和 **Masked Inverse Folding**。ProteinMPNN 是固定 backbone sequence design 的强 baseline；Masked Inverse Folding 则预训练一个 structured graph neural network，从 backbone 重构 sequence，并借助 sequence-only pLM 的 sequence-transfer input 提升性能。

这些不是 unified-token 模型，但它们是重要 baseline，因为它们可以回答：“显式结构到底在多大程度上帮助 sequence/function generation？”它们也提示 ProteinChameleon 应该和结构感知但非 token 化的方法做 ablation。

## F. Open research gaps 和 ProteinChameleon 的下一步实验

文献中的主要空白可以很简单地概括：**我没有找到一篇 prior protein paper 同时结合了以下四点**：

1. decoder-only causal LM；
2. 一个底层统一词表，覆盖自然语言和 **离散蛋白结构 tokens**；
3. 先在 **protein-only structure-token sequences** 上 warm up 新词表；
4. 再在 **带有 inline protein structure-token spans 的科学文本** 上训练。

ProtLLM 最接近 interleaving，ESM3 最接近 discrete multimodal tokenization，但 ProteinChameleon 的精确组合看起来仍然是新的。

因此，最重要的下一步实验是证明你的设计优于显而易见的替代方案。

第一，应该正面对比不同 **tokenizer families**：Foldseek/3Di、VQ tokenizer，例如 FoldToken 或 AIDO.StructureTokenizer，以及你的 PT-BPE/GeoBPE tokenizer。评估不要只看 reconstruction，还要看 **token perplexity、codebook utilization、retrieval accuracy、function-text generation quality、interleaved-document robustness**。GeoBPE 和 Yeti 都强烈暗示 tokenizer 质量是多模态生成的限制因素，所以这个 ablation 是核心实验，不是可选项。

第二，显式测试 **warmup stage** 的价值。可以比较：

1. joint-from-scratch model；
2. 只在 protein-only structure-token stream 上 warmup；
3. 在 mixed sequence + structure stream 上 warmup；
4. 加入短 natural-language protein captions 的 warmup。

这是你设计中最 novel 的部分之一。Learning the Language of Protein Structure、FoldToken 和 SEPIT 都暗示 staged training 很重要，但它们都没有测试你这种“先学新 token embedding/output head，再进入 mixed multimodal training”的具体方案。

第三，把 **interleaving** 当作一等能力来评估，而不仅仅当作一种额外训练数据。ProtLLM 和 MM-Interleaved 都说明，interleaving 可能带来 compositional reasoning 和 in-context learning 的收益，而不只是增加训练多样性。好的测试包括：

- document continuation：模型看到 inline structure-token span 后必须生成 function paragraph；
- paragraph editing：改变 structure span 后，预测 annotation 应该随之改变；
- retrieval-augmented reading：科学文本中同时引用多个 inline protein，模型需要综合理解。

第四，把 **alignment vs generation** 作为明确的 benchmark axis。可以把 ProteinChameleon 和 ProTrek/FAPM 风格的 contrastive system 比 retrieval 和 zero-shot function-term ranking；同时和 Prot2Text/ProtT3/STELLA 风格的 generative system 比 captioning 和 functional-description prediction。如果 ProteinChameleon 成功，它最好能用同一个模型同时在两类任务上表现不错。这比只超过一个 captioning baseline 更有说服力。

第五，加入 **structure-token generation audit**。如果模型可以 autoregressively 输出蛋白结构 token，就应该评估生成的 structure-token span 是否：

- 局部合法；
- 全局一致；
- 可解码成合理的 3D geometry。

Tokenizer 论文已经说明，局部 plausibility 和 reconstruction accuracy 不够；你的模型还需要测试 **long-range consistency**，尤其是在结构 span 被生成到更长文本文件内部时。

第六，可以考虑在 mixed training 中加入 **trimodal objective**。ProTrek 的成功说明，sequence、structure-token span 和 function text 之间的 retrieval-style alignment loss 可能会改善 decoder-only objective，尤其是在训练早期文本生成还不稳定的时候。一个有用实验是，在保留主要 causal-LM loss 的同时，对三种模态的 pooled representation 加一个 contrastive auxiliary loss。

简洁的实现优先级排序是：

**ProtLLM、ESM3、FoldToken/FoldToken2、GeoBPE、Yeti、SaProt、Prot2Text、ProtT3、ProTrek、STELLA。**

这些是最可能影响 ProteinChameleon 最终设计选择的论文。

## 开放问题与限制

有些细节在可直接访问的公开页面中仍然不完整。对于部分论文，尤其是需要订阅或 PDF 信息较多的论文，我可以确认核心架构、模态、训练目标和 venue，但不一定能完整确认每个 dataset split 或每个 metric 的全部细节。这主要影响一些不那么核心的条目，例如 SaProt、ProTrek 和某些较新的 multimodal protein instruction-tuning paper 的逐任务指标名称。

因此，报告中更强调 **高置信度的架构和实验事实**，并在概念比较时尽量谨慎。

ProteinChameleon 最大的未解问题不是“它能不能工作”，而是：

**哪一种结构 tokenization regime 最适合一个必须同时 interleave biology 和 prose 的 causal LM？**

现有文献强烈说明，这个问题仍然是开放的。

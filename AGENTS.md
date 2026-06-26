# ProteinChameleon 项目指南

## 项目目标

ProteinChameleon 是一个把蛋白质结构离散 token 接入 Gemma4 因果语言模型的实验项目。核心思路是把文本 token、蛋白结构边界 token 和 PT-BPE/GeoBPE 结构 token 放进同一个词表，让模型在同一个自回归 Transformer 里同时建模文本和蛋白结构片段。

统一序列的大致形式：

```text
[文本 token] <PROT_START> [蛋白结构 BPE token] <PROT_END> [文本 token]
```

训练分两阶段：

1. Stage 1 warmup：只训练新增 token 相关的 embedding / lm_head 行，让 `<PROT_START>`、`<PROT_END>` 和约 2100 个蛋白结构 token 有合理表示。
2. Stage 2 mixed training：从 Stage 1 checkpoint 出发，对 alignment 数据和 interleaved 数据做混合训练，让模型学习结构到功能文本，以及文本叙述中穿插结构片段的生成。

## 目录结构

```text
.
├── README.md                 # 当前只有项目名
├── requirements.txt          # Python 依赖
├── run_eval.sh               # Stage 2 多 GPU 生成评估 + PDF 可视化流水线
├── model/
│   ├── __init__.py           # 导出配置、tokenizer、模型类
│   ├── config.py             # ProteinChameleonConfig 和结构 token 常量
│   ├── tokenizer.py          # 文本 tokenizer 包装器，负责蛋白 token offset
│   └── model.py              # 扩展 Gemma4ForCausalLM 的主模型
└── scripts/
    ├── encode_structures.py
    ├── train_warmup.py
    ├── infer_warmup.py
    ├── download_structures.py
    ├── featurize_stage2.py
    ├── split_chunks.py
    ├── encode_stage2.py
    ├── prepare_alignment.py
    ├── prepare_domains.py
    ├── submit_interleaved_batch.py
    ├── process_interleaved_batch.py
    ├── train_stage2.py
    ├── clean_test_sets.py
    ├── eval_stage2.py
    └── visualize_eval.py
```

## 模型组件

- `model/config.py`
  - 定义 `ProteinChameleonConfig`，继承 `Gemma4Config`。
  - `BPE_VOCAB_SIZE = 2100`，对应 600 个 motif token 和 1500 个 angle bin。
  - 新增特殊 token：`<PROT_START>`、`<PROT_END>`。
  - 维护 `protein_token_offset`，即结构 token 在统一词表中的起始 ID。

- `model/tokenizer.py`
  - `ProteinChameleonTokenizer` 包装 Hugging Face `AutoTokenizer`。
  - 在文本 tokenizer 后追加两个结构边界特殊 token。
  - 将原始 PT-BPE token `i` 映射为 `protein_token_offset + i`。
  - 提供 `encode_text`、`shift_protein_ids`、`encode_mixed`、`decode` 和 `apply_to_config`。

- `model/model.py`
  - `ProteinChameleonForCausalLM` 继承 `Gemma4ForCausalLM`。
  - 扩展 `embed_tokens`、`embed_tokens_per_layer` 和 `lm_head` 到统一词表大小。
  - `from_gemma()` 会从 Gemma4 checkpoint 读取文本权重，复制原始文本 embedding/head，并初始化新增蛋白 token 行。
  - `tie_weights()` 保留 Gemma 文本部分权重绑定，但避免覆盖蛋白 token 的 lm_head 行。

## 数据与训练流程

### Stage 1 warmup

输入是已经 PT-BPE 编码好的结构 token：

```text
encoded/warmup.npz
  token_ids: object array，每个元素是一条蛋白的原始 BPE token 序列
  fnames:    来源文件名
```

常见流程：

```bash
python scripts/encode_structures.py --workers 8
python scripts/train_warmup.py --base-model google/gemma-4-E4B --encoded-file encoded/warmup.npz
python scripts/infer_warmup.py --mode both --ckpt-dir checkpoints/warmup/final
```

Stage 1 只更新新增 token 相关行，原 Gemma 参数冻结；loss 只计算结构 token 的 next-token prediction。

### Stage 2 mixed training

Stage 2 训练两类数据：

- Alignment：`Organism + Sequence + <PROT_START> structure <PROT_END> -> function_text`，loss 只落在功能文本上。
- Interleaved：包含文本和多个内联结构片段的叙述，loss 覆盖文本和结构 token。

常见流程：

```bash
python scripts/download_structures.py
python scripts/featurize_stage2.py
python scripts/split_chunks.py --feat-dir /path/to/featurized --part-size 500
python scripts/encode_stage2.py --feat-dir /path/to/featurized --out-dir /path/to/encoded --merge
python scripts/prepare_alignment.py --stage2-npz /path/to/stage2.npz --out-dir /path/to/stage2-alignment
python scripts/prepare_domains.py --feat-dir /path/to/featurized --out-dir /path/to/stage2
python scripts/submit_interleaved_batch.py --out-dir /path/to/stage2
python scripts/process_interleaved_batch.py --out-dir /path/to/stage2
python scripts/train_stage2.py --warmup-ckpt checkpoints/warmup/final --align-dir /path/to/stage2-alignment --interleaved-dir /path/to/stage2
```

`train_stage2.py` 会把 alignment 和 interleaved 数据按预设 token 总量比例采样，并使用固定长度窗口进行 greedy packing。

## 脚本功能速查

- `scripts/encode_structures.py`
  - 把 PT-BPE `Tokenizer` pkl 文件量化成 `warmup.npz`。
  - 用于 Stage 1 warmup 数据准备。

- `scripts/train_warmup.py`
  - 从 Gemma4 初始化 ProteinChameleon。
  - 冻结 Gemma 主体，只训练新增 token 对应的 embedding/lm_head 行。
  - 输出 `checkpoints/warmup/final`。

- `scripts/infer_warmup.py`
  - 加载 warmup checkpoint。
  - 支持 `perplexity`、`generate`、`both`。
  - 输出 perplexity JSON/CSV、生成 token NPZ/TXT 和运行 metadata。

- `scripts/download_structures.py`
  - 根据 `proteins.csv` 下载 AlphaFold PDB，失败时用 RCSB PDB ID 兜底。
  - 输出 `{accession}.pdb` 和失败日志。

- `scripts/featurize_stage2.py`
  - 把 PDB 结构转换成 foldingdiff/GeoBPE 所需的 backbone geometry dict。
  - 按 chunk 保存 `chunk_XXXX.pkl`。

- `scripts/split_chunks.py`
  - 把较大的 `chunk_XXXX.pkl` 切成 `chunk_XXXX_part_MM.pkl`。
  - 原始 chunk 会移动到 `_originals/`，便于降低后续多进程编码内存峰值。

- `scripts/encode_stage2.py`
  - 使用 PT-BPE checkpoint 把 featurized chunk 编成结构 token。
  - 支持子 chunk、heartbeat、断点续跑和 `--merge`/`--merge-only`。
  - 合并后生成 `stage2.npz`。

- `scripts/prepare_alignment.py`
  - 从 `stage2.npz` 和 `proteins.csv` 过滤出有足够 function_text、长度合适且有 AA sequence 的样本。
  - 生成 `alignment_train.npz`、`alignment_val.npz`、`alignment_test.npz`。

- `scripts/prepare_domains.py`
  - 根据 InterPro domain 注释切出结构片段，并编码成 domain token。
  - 生成 `domain_tokens.npz`，供 interleaved 数据替换结构占位符使用。

- `scripts/submit_interleaved_batch.py`
  - 生成 OpenAI Batch API 请求，让 GPT-4.1 基于蛋白功能、序列和 domain 信息写带 `[IPRxxxxxx]` 占位符的科学叙述。
  - 输出 batch input、metadata 和 batch info。

- `scripts/process_interleaved_batch.py`
  - 下载或读取 GPT batch 结果。
  - 把 `[IPRxxxxxx]` 替换成 `[PROT_START] token... [PROT_END]`。
  - 生成 `interleaved_train.npz`、`interleaved_val.npz`、`interleaved_test.npz`。

- `scripts/train_stage2.py`
  - 加载 Stage 1 checkpoint，做 alignment + interleaved 混合全量微调。
  - 使用 per-token `loss_mask` 控制哪些位置参与 loss。
  - 输出 `checkpoints/stage2/final`。

- `scripts/clean_test_sets.py`
  - 删除 test set 中和 train/val sequence 重复的样本。
  - 输出 `_clean.npz` 测试集。

- `scripts/eval_stage2.py`
  - 计算 alignment/interleaved perplexity。
  - 在 alignment prompt 上生成 function text，并计算 BLEU、ROUGE、BERTScore（依赖可选）。
  - 支持 interleaved 生成和多 shard 并行。

- `scripts/visualize_eval.py`
  - 把评估 JSON 转成每个样本一个 PDF。
  - 可渲染结构片段、文本对比、BERTScore/ROUGE、Ramachandran 统计。

- `run_eval.sh`
  - 封装 Stage 2 多 GPU 生成评估和 PDF 可视化。
  - 注意当前脚本里 alignment/interleaved 推理逻辑有重复段落，维护时应小心避免同一任务被跑两遍。

## 本机可用数据和 checkpoint

当前仓库本身不包含训练数据，但这台机器上已经有一套准备好的 `.npz` 数据，位于 `/data2/steven/data`：

```text
/data2/steven/data/
├── warmup/
│   └── warmup.npz
└── stage2/
    ├── alignment/
    │   ├── alignment_train.npz
    │   ├── alignment_val.npz
    │   ├── alignment_test.npz
    │   └── alignment_test_clean.npz
    ├── interleaved_train.npz
    ├── interleaved_val.npz
    ├── interleaved_test.npz
    └── interleaved_test_clean.npz
```

这些文件字段与训练脚本匹配：

- `warmup/warmup.npz`：`token_ids`、`fnames`，用于 `train_warmup.py`。
- `stage2/alignment/*.npz`：`token_ids`、`accessions`、`sequences`、`function_text`、`organism`，用于 alignment 训练/评估。
- `stage2/interleaved_*.npz`：`accessions`、`sequences`、`organism`、`narratives`、`n_domains`，用于 interleaved 训练/评估。

这些数据名的含义：

- `warmup`：Stage 1 使用的纯结构 token 预热数据。每条样本是一条蛋白结构被 PT-BPE/GeoBPE 编码后的离散 token 序列，用来先学习新增结构 token 的 embedding 和输出头。
- `stage1`：也就是 warmup 训练阶段。Gemma 主体基本冻结，主要让 `<PROT_START>`、`<PROT_END>` 和结构 token 先具备基本语言建模能力。
- `alignment`：Stage 2 的结构/序列/文本对齐数据。每条样本包含物种、氨基酸序列、结构 token 和功能描述；训练时模型看到 `organism + sequence + structure tokens`，loss 主要落在 `function_text` 上。
- `interleaved`：Stage 2 的交错多模态叙述数据。每条 `narrative` 中自然语言和结构 token span 交替出现，例如 `text <PROT_START> structure tokens <PROT_END> text`，用于训练模型处理/生成内联结构片段。
- `stage2`：从 Stage 1 checkpoint 出发，用 `alignment` 和 `interleaved` 两类数据混合全量微调，是最终主训练阶段。

当前数据量统计：

| 数据文件 | 用途 | 文件大小 | 样本数 | 主要规模 |
|---|---|---:|---:|---:|
| `/data2/steven/data/warmup/warmup.npz` | Stage 1 warmup | 115M | 26,911 | 29,223,403 个结构 token，平均 1,086/token 序列 |
| `/data2/steven/data/stage2/alignment/alignment_train.npz` | Stage 2 alignment train | 1.5G | 243,215 | 342,491,607 个结构 token，平均 1,408/token 序列 |
| `/data2/steven/data/stage2/alignment/alignment_val.npz` | alignment val | 83M | 13,351 | 18,731,287 个结构 token，平均 1,403/token 序列 |
| `/data2/steven/data/stage2/alignment/alignment_test.npz` | alignment test | 84M | 13,571 | 19,058,091 个结构 token，平均 1,404/token 序列 |
| `/data2/steven/data/stage2/alignment/alignment_test_clean.npz` | 去重后 alignment test | 70M | 10,719 | 15,803,335 个结构 token，平均 1,474/token 序列 |
| `/data2/steven/data/stage2/interleaved_train.npz` | Stage 2 interleaved train | 493M | 64,641 | 481,868,941 个 narrative 字符，150,403 个 domain span |
| `/data2/steven/data/stage2/interleaved_val.npz` | interleaved val | 27M | 3,520 | 26,356,578 个 narrative 字符，8,179 个 domain span |
| `/data2/steven/data/stage2/interleaved_test.npz` | interleaved test | 26M | 3,465 | 25,855,129 个 narrative 字符，8,076 个 domain span |
| `/data2/steven/data/stage2/interleaved_test_clean.npz` | 去重后 interleaved test | 23M | 2,902 | 21,760,730 个 narrative 字符，6,726 个 domain span |
 
汇总：

- `warmup`：26,911 条样本，约 2,922 万结构 token。
- `alignment`：train + val + test 共 270,137 条样本；如果使用 clean test，则 train + val + clean test 共 267,285 条样本。
- `interleaved`：train + val + test 共 71,626 条样本；如果使用 clean test，则 train + val + clean test 共 71,063 条样本。

这台机器上也有训练好的 checkpoint，位于 `/data2/steven/checkpoints`：

```text
/data2/steven/checkpoints/
├── warmup/final/
└── stage2/
    ├── final/
    └── checkpoint-4200 ... checkpoint-10000
```

常用命令应覆盖脚本里的旧默认路径，例如：

```bash
python scripts/train_stage2.py \
  --warmup-ckpt /data2/steven/checkpoints/warmup/final \
  --align-dir /data2/steven/data/stage2/alignment \
  --interleaved-dir /data2/steven/data/stage2 \
  --out-dir /data2/steven/checkpoints/stage2_new

python scripts/eval_stage2.py \
  --ckpt /data2/steven/checkpoints/stage2/final \
  --align-test /data2/steven/data/stage2/alignment/alignment_test_clean.npz \
  --interleaved-test /data2/steven/data/stage2/interleaved_test_clean.npz
```

## 重要运行注意事项

- 很多脚本默认路径写死为作者机器路径，例如 `/data/steven/...`、`/data2/steven/...`、`/home/steven/PT-BPE`、`/home/ubuntu/...`。在当前机器运行前，优先通过 CLI 参数覆盖路径。
- 结构编码依赖外部 PT-BPE/GeoBPE/foldingdiff 代码库和对应 conda 环境；`requirements.txt` 只覆盖主训练代码的基础依赖，不包含所有可视化和结构处理依赖。
- `submit_interleaved_batch.py` 和 `process_interleaved_batch.py` 需要 `OPENAI_API_KEY`。
- Stage 1/2 都默认使用 wandb；如果不想上报，需要改脚本里的 `report_to` 或设置环境变量。
- 训练脚本默认使用 bf16、gradient checkpointing 和 `device_map="auto"`，适合多 GPU/大显存环境。
- 数据 split 使用 Python 内置 `hash(acc) % 100`，如果需要跨进程或跨机器严格可复现，应设置固定 `PYTHONHASHSEED` 或改成稳定哈希函数。

## 代码修改建议

- 改模型词表或 tokenizer 逻辑时，必须同时检查 `ProteinChameleonConfig`、`ProteinChameleonTokenizer.apply_to_config()` 和 `ProteinChameleonForCausalLM.from_gemma()`。
- 改数据字段时，必须同步检查 `prepare_alignment.py`、`train_stage2.py`、`eval_stage2.py` 和 `visualize_eval.py`。
- 改结构 token 格式时，必须同步检查 interleaved 生成链路里的 `[PROT_START] ... [PROT_END]` 正则解析。
- 大规模脚本优先先用 `--limit`、`--max-proteins`、`--n` 或小 batch 做 smoke test，再跑全量任务。

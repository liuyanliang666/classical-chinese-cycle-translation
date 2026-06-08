# 古文今译：基于回环一致性约束的古今中文双向翻译系统

本项目将视觉领域 CycleGAN 的 **Cycle Consistency** 思想迁移到古今中文翻译任务：训练一个双向 Seq2Seq 模型，在标准翻译损失之外引入 **回环一致性损失（Cycle Consistency Loss）** 作为辅助训练信号，促使模型生成信息保留更完整、可逆性更强的翻译结果。

核心思路：

```text
古文 A  →  模型翻译  →  现代文 B  →  模型翻译  →  古文 A'

总损失 = 翻译损失 + λ × 回环损失(A, A')
```

与 CycleGAN 的区别在于：图像是连续信号，可以直接用像素级 L1 loss 反向传播；文本生成中间有离散采样（argmax），梯度不连续。本项目采用 **Soft Token 近似** 解决这一问题——将第一次翻译的 softmax 概率分布作为连续输入喂给第二次翻译，使回环损失可以端到端反向传播。

## 项目阶段

```text
阶段 1：双向翻译基线
模型：Randeng-BART-139M / mT5-small（对照组）
数据：NiuTrans/Classical-Modern 平行语料
目标：标准 Seq2Seq 微调，建立翻译基线

阶段 1.5：回环一致性约束训练（核心创新）
方法：在阶段 1 的训练循环中加入回环损失
技术：Soft Token 近似（softmax 概率 × embedding 矩阵）绕过离散采样不可导问题
对比：有/无回环约束的翻译质量差异、回环一致性分数变化

阶段 2：回环一致性分析
指标：字面一致性（ChrF / 编辑距离）+ 语义一致性（BERTScore）+ 结构一致性（长度比 / 句数差）
分析维度：按文本长度、来源书籍、虚词比例分组，定位信息损失规律
```

## Soft Token 近似原理

标准 Seq2Seq 生成文本时，每一步通过 argmax 选出一个离散 token，梯度在这里断开。Soft Token 近似的做法是：

```text
第一次前向（古→今）：
  logits_B = Model(A)
  soft_embeds_B = softmax(logits_B) × Embedding_Matrix   ← 连续，可导

第二次前向（今→古）：
  logits_A' = Model(inputs_embeds=soft_embeds_B)          ← 直接接受连续输入

回环损失：
  cycle_loss = CrossEntropy(logits_A', token_ids_A)       ← 对原始 A 的 token 计算交叉熵
```

Transformer 的 forward 方法原生支持 `inputs_embeds` 参数，因此无需修改模型架构，只需在训练循环中增加一次前向推理。

## 目录结构

```text
.
├── app.py                       # Streamlit Demo
├── environment.yml              # Conda 环境
├── requirements.txt             # Python 依赖
├── scripts/
│   ├── prepare_data.py           # CSV / NiuTrans 数据转换
│   └── run_eval.py               # 简单评估脚本
├── src/ccnlp/
│   ├── preprocess.py             # 数据清洗与任务构造
│   ├── inference.py              # 当前规则基线推理
│   ├── evaluate.py               # Exact Match / ChrF
│   ├── train_seq2seq.py          # 第一阶段 Seq2Seq 微调
│   └── examples.py               # Demo 示例
└── tests/
    └── test_core.py
```

## 1. 安装 Conda 环境

第一次创建环境：

```bash
cd /Users/mac/NLP/classical_chinese_project
conda env create -f environment.yml
conda activate classical-chinese-nlp
```

如果环境已经存在，只更新依赖：

```bash
cd /Users/mac/NLP/classical_chinese_project
conda activate classical-chinese-nlp
pip install -r requirements.txt
```

## 2. 运行当前 Demo

当前 Demo 使用规则基线，不依赖已经训练好的模型：

```bash
cd /Users/mac/NLP/classical_chinese_project
conda activate classical-chinese-nlp
streamlit run app.py
```

浏览器打开终端显示的 `Local URL`，通常是：

```text
http://localhost:8501
```

## 3. 准备 NiuTrans 平行语料

下载数据集：

```bash
cd /Users/mac/NLP/classical_chinese_project/data/raw
git clone https://github.com/NiuTrans/Classical-Modern.git
```

NiuTrans 数据的核心目录是：

```text
data/raw/Classical-Modern/双语数据
```

里面按书籍/章节递归存放：

```text
source.txt  # 文言文
target.txt  # 现代白话文
bitext.txt  # 对照文本
```

## 4. 转换为双向训练 JSONL

先建议抽取 5 万条平行句对，避免第一次训练数据过大：

```bash
cd /Users/mac/NLP/classical_chinese_project

python scripts/prepare_data.py \
  --format niutrans \
  --input "data/raw/Classical-Modern/双语数据" \
  --output_dir data/processed \
  --max_examples 50000
```

脚本会先去重，在「句对」层级划分 train/validation/test（默认各 1% 给 val/test），再把每个句对扩展成两个方向，并附带 `book`/`length` 元信息（供阶段2分组分析）：

```json
{"task": "classical_to_modern", "source": "古文翻今：学而时习之，不亦说乎", "target": "学习后按时温习，不也很快乐吗", "book": "论语", "length": 9}
{"task": "modern_to_classical", "source": "今文翻古：学习后按时温习，不也很快乐吗", "target": "学而时习之，不亦说乎", "book": "论语", "length": 9}
```

输出三个文件：`data/processed/{train,validation,test}.jsonl`。每个句对扩展成两条，所以 `50000` 个句对约产生 `100000` 条样本。划分在句对层级进行，保证同一句对的两个方向不会跨 split，避免泄漏。可用 `--val_ratio` / `--test_ratio` / `--seed` 调整。

如果机器资源足够，可以去掉 `--max_examples` 使用完整数据。

## 5. 第一阶段：Seq2Seq 翻译微调

第一阶段目标是训练：

```text
古文 -> 白话文
白话文 -> 古文
```

推荐主模型：

```text
IDEA-CCNL/Randeng-BART-139M-SUMMARY
```

原因：这是中文生成模型，比 mT5 这种多语言模型更适合作为中文翻译主模型。mT5 可以作为对照组。

### 5.1 冒烟测试

先跑一个很小的训练，确认环境、模型下载、数据格式都没问题：

```bash
cd /Users/mac/NLP/classical_chinese_project
conda activate classical-chinese-nlp

python -m ccnlp.train_seq2seq \
  --train_file data/processed/train.jsonl \
  --model_name IDEA-CCNL/Randeng-BART-139M-SUMMARY \
  --output_dir outputs/checkpoints/debug-randeng-bart \
  --epochs 1 \
  --batch_size 2 \
  --max_train_samples 200
```

如果显存或内存不足，把 `--batch_size 2` 改成 `1`。

### 5.2 正式训练

冒烟测试通过后，再启动正式训练：

```bash
python -m ccnlp.train_seq2seq \
  --train_file data/processed/train.jsonl \
  --validation_file data/processed/validation.jsonl \
  --model_name IDEA-CCNL/Randeng-BART-139M-SUMMARY \
  --output_dir outputs/checkpoints/randeng-bart-niutrans \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5
```

训练完成后 checkpoint 会保存在：

```text
outputs/checkpoints/randeng-bart-niutrans
```

### 5.3 mT5 对照组

做报告对比时，可以把模型换成 mT5：

```bash
python -m ccnlp.train_seq2seq \
  --train_file data/processed/train.jsonl \
  --validation_file data/processed/validation.jsonl \
  --model_name google/mt5-small \
  --output_dir outputs/checkpoints/mt5-small-niutrans \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5
```

对照目的：

```text
比较中文中心模型和多语言模型在古今翻译任务上的差异。
```

## 5.4 鲁迅风格：Qwen3-4B LoRA 微调

鲁迅风格改写不再建议用 `IDEA-CCNL/Randeng-BART-139M-SUMMARY` 作为主模型。推荐使用：

```text
Qwen/Qwen3-4B + QLoRA
```

如果服务器 `data` 目录下已有 `luxun_plain_pairs.filtered.jsonl`，先转换成训练/验证/测试三份 JSONL：

```bash
python scripts/build_luxun_style_data.py \
  --input data/luxun_plain_pairs.filtered.jsonl \
  --output_dir data/processed/luxun_style
```

输出文件：

```text
data/processed/luxun_style/train.jsonl
data/processed/luxun_style/validation.jsonl
data/processed/luxun_style/test.jsonl
```

冒烟训练：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m ccnlp.train_causal_lora \
  --train_file data/processed/luxun_style/train.jsonl \
  --validation_file data/processed/luxun_style/validation.jsonl \
  --model_name Qwen/Qwen3-4B \
  --output_dir outputs/checkpoints/debug-qwen3-luxun-lora \
  --max_train_samples 16 \
  --epochs 0.05 \
  --batch_size 1 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 384
```

正式训练（双卡 12GB RTX 3080 Ti 可先这样跑；如果 DDP + QLoRA 在环境中不稳定，先退回单卡）：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 -m ccnlp.train_causal_lora \
  --train_file data/processed/luxun_style/train.jsonl \
  --validation_file data/processed/luxun_style/validation.jsonl \
  --model_name Qwen/Qwen3-4B \
  --output_dir outputs/checkpoints/qwen3-4b-luxun-lora \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_seq_length 512 \
  --lora_r 16
```

单句推理：

```bash
PYTHONPATH=src python -m ccnlp.generate \
  --backend causal_lora \
  --base_model Qwen/Qwen3-4B \
  --adapter_dir outputs/checkpoints/qwen3-4b-luxun-lora \
  --task luxun_style \
  --input "街上很多人沉默地走过，没有人愿意先开口。"
```

测试集评估：

```bash
PYTHONPATH=src python -m ccnlp.eval_causal_lora \
  --base_model Qwen/Qwen3-4B \
  --adapter_dir outputs/checkpoints/qwen3-4b-luxun-lora \
  --test_file data/processed/luxun_style/test.jsonl \
  --output outputs/checkpoints/qwen3-4b-luxun-lora/test_predictions.jsonl
```

## 6. 回环一致性约束训练（核心创新）

### 6.1 实验设计

本项目的核心对比实验：

```text
实验组 A（基线）：标准 Seq2Seq 翻译损失
实验组 B（本文方法）：翻译损失 + λ × 回环一致性损失

对比维度：
  - 翻译质量：BLEU / ChrF / BERTScore
  - 回环一致性：RTC 分数（见下文）
  - 信息保留：按文本类型分组的回环损失分布
```

### 6.2 训练循环伪代码

```python
for batch in dataloader:
    # ---- 标准翻译损失 ----
    outputs = model(input_ids=batch["source"], labels=batch["target"])
    translation_loss = outputs.loss

    # ---- 回环一致性损失（Soft Token 近似）----
    # 第一次前向：源语言 → 目标语言的 logits
    logits_forward = outputs.logits
    # 用 softmax 概率加权 embedding，得到连续表示（绕过 argmax 不可导）
    soft_embeds = logits_forward.softmax(dim=-1) @ model.get_input_embeddings().weight
    # 第二次前向：连续表示 → 尝试恢复源语言
    cycle_outputs = model(inputs_embeds=soft_embeds, labels=batch["source"])
    cycle_loss = cycle_outputs.loss

    # ---- 合并 ----
    total_loss = translation_loss + lambda_cycle * cycle_loss
    total_loss.backward()
```

### 6.3 与 CycleGAN 的类比

本方法借鉴了视觉领域 CycleGAN (Zhu et al., 2017) 的核心思想：

```text
CycleGAN（图像）：            本项目（文本）：
域A图像 → G → 域B图像          古文 → Seq2Seq → 现代文
域B图像 → F → 域A'图像         现代文 → Seq2Seq → 古文'
L_cycle = ‖A - A'‖₁           L_cycle = CE(logits_A', tokens_A)

图像是连续信号 → 直接 L1       文本是离散信号 → Soft Token 近似
```

关键区别：文本生成的离散性使得 CycleGAN 的 L1 loss 无法直接使用。本项目通过 Soft Token 近似将离散采样替换为连续的 softmax 概率分布，使梯度可以端到端传播。

## 7. 回环一致性评估体系

### 7.1 三层一致性指标

对测试集执行 A → B → A' 回环翻译后，从三个层次度量 A 与 A' 的一致性：

```text
字面一致性（权重 0.2）：
  ChrF(A, A') + 归一化编辑距离
  衡量原文在字符级别的恢复程度

语义一致性（权重 0.6）：
  BERTScore F1(A, A')
  衡量语义信息的保留程度（允许同义改写）

结构一致性（权重 0.2）：
  长度比 len(A')/len(A) + 句数差
  衡量文本结构是否发生变形

综合回环一致性分数：
  RTC = 0.2 × 字面 + 0.6 × 语义 + 0.2 × 结构
```

### 7.2 分组分析维度

```text
按文本长度分组：短句（<20字）/ 中句（20-50字）/ 长句（>50字）
按来源书籍分组：论语 / 史记 / 庄子 / 诗经 等
按虚词比例分组：虚词密度高 / 低
按翻译方向分组：古→今→古 vs 今→古→今
```

目标：定位哪类文本的信息损失最严重，分析回环约束是否有效缓解了特定类型的信息丢失。

## 8. 当前限制

- Streamlit Demo 当前仍使用 `BaselineGenerator` 规则基线，训练完成后需接入模型。
- 回环一致性约束训练的自定义 training step 尚未实现。
- 评估指标需补充 BLEU 和 BERTScore。
- 分组分析模块尚未实现。

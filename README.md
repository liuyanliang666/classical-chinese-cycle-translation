# 古文今译：回环一致性约束的古今中文双向翻译

本项目研究两个相关的中文生成任务：古文与现代汉语之间的双向翻译，以及现代汉语到鲁迅风格文本的改写。古今互译部分将视觉领域 CycleGAN 的 **Cycle Consistency** 思想迁移到文本生成任务中，目标不是只追求单向 BLEU，而是让译文在语义、结构和关键信息上更可逆，减少古今互译中的信息丢失和幻觉；鲁迅风格任务使用 Qwen3-4B LoRA 进行风格迁移训练与推理。

核心方法：

```text
古文 A -> 模型翻译 -> 现代文 B -> 模型回译 -> 古文 A'

总损失 = 翻译损失 + lambda_cycle * 回环一致性损失(A, A')
```

文本生成中的 `argmax` 离散采样不可导，因此本项目使用 **Soft Token 近似**：第一次翻译得到的 softmax 分布与 embedding 矩阵相乘，形成连续表示，再输入第二次翻译模型计算回环损失。

## 当前内容

- 古文 -> 白话文、白话文 -> 古文双向 Seq2Seq 训练与推理。
- `lambda_cycle` 回环一致性训练实验。
- RTC 回环一致性、复制率、幻觉率等评估脚本。
- Streamlit 本地前端 + FastAPI 云端 GPU 推理后端。
- 鲁迅风格改写的 Qwen3-4B LoRA 数据处理、训练、推理和评估入口。

## 目录结构

```text
.
├── app.py                         # Streamlit 前端
├── scripts/
│   ├── prepare_data.py             # NiuTrans 数据转换
│   ├── run_eval.py                 # 翻译质量评估入口
│   ├── plot_ablation.py            # lambda 消融绘图
│   ├── faithful_eval.py            # 抗幻觉解码评估
│   ├── build_luxun_style_data.py   # 鲁迅风格数据构造
│   ├── modernize_luxun_with_api.py # 调用大模型生成鲁迅白话对照
│   ├── prepare_luxun_textgrid.py   # 鲁迅语料切分与清洗
│   └── dir_bertscore.py            # 目录级 BERTScore 计算
├── src/ccnlp/
│   ├── train_seq2seq.py            # Seq2Seq / 回环一致性训练
│   ├── train_causal_lora.py        # Qwen LoRA 风格迁移训练
│   ├── generate.py                 # 模型推理 CLI
│   ├── api_service.py              # FastAPI 后端模型路由与懒加载
│   ├── api_server.py               # /health 与 /generate HTTP 接口
│   ├── inference.py                # 模型推理封装
│   ├── ui_config.py                # 前端任务和样式配置
│   ├── eval_runner.py              # BLEU / ChrF / ExactMatch
│   ├── eval_causal_lora.py         # 鲁迅 LoRA 评估
│   ├── rtc_eval.py                 # 回环一致性评估
│   ├── copy_rate.py                # 复制坍缩诊断
│   ├── hallucination.py            # 数字与内容字幻觉诊断
│   ├── faithful_decode.py          # 抗幻觉解码实验
│   ├── causal_sft.py               # Causal LM SFT 数据处理
│   └── luxun_textgrid.py           # 鲁迅文本切分工具
├── output/doc/
│   └── TransVerse_期末报告.docx     # 项目期末报告
├── tests/                          # 单元测试
├── environment.yml                 # Conda 环境
└── requirements.txt                # Python 依赖
```

## 安装

```bash
conda env create -f environment.yml
conda activate classical-chinese-nlp
pip install -r requirements.txt
```

如果环境已存在：

```bash
conda activate classical-chinese-nlp
pip install -r requirements.txt
```

## 运行 Demo

当前 Demo 采用前后端分离方式运行：GPU 服务器启动 FastAPI 推理后端，本机启动 Streamlit 前端。前端只负责输入、风格选择和展示后端返回结果；文言文风格和鲁迅风格的真实模型推理由后端完成。

### 1. 在 GPU 服务器启动后端

在服务器上进入项目目录，并设置模型路径。下面的路径都是示例，需要替换成自己服务器上的实际项目目录和模型目录：

```bash
cd /path/to/classical_chinese_project
conda activate classical-chinese-nlp

export CCNLP_SEQ2SEQ_MODEL=/path/to/classical_chinese_project/outputs/checkpoints/<seq2seq_checkpoint>
export CCNLP_QWEN_BASE_MODEL=/path/to/classical_chinese_project/models/<qwen_base_model>
export CCNLP_LUXUN_ADAPTER=/path/to/classical_chinese_project/outputs/checkpoints/<luxun_lora_adapter>

PYTHONPATH=src uvicorn ccnlp.api_server:app --host 127.0.0.1 --port 8000
```

服务器路径示例：

```bash
cd /root/classical_chinese_project
conda activate classical-chinese-nlp

export CCNLP_SEQ2SEQ_MODEL=/root/classical_chinese_project/outputs/checkpoints/randeng-bart-modern-to-classical-100k-bs16
export CCNLP_QWEN_BASE_MODEL=/root/classical_chinese_project/models/Qwen3-4B-Instruct-2507
export CCNLP_LUXUN_ADAPTER=/root/classical_chinese_project/outputs/checkpoints/qwen3-4b-instruct-modern-to-luxun-api-lora-fast/checkpoint-2364

PYTHONPATH=src uvicorn ccnlp.api_server:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

生成接口是 `POST /generate`，请求体示例：

```json
{
  "text": "要转换的内容",
  "task": "luxun_style",
  "style_strength": 0.65
}
```

其中 `task` 可取：

```text
modern_to_classical
luxun_style
```

鲁迅风格前端滑块是 `0` 到 `1`，后端推理时会映射到 `1.0` 到 `1.3` 的 LoRA 风格强度。

### 2. 在本机打通 SSH 隧道

如果后端只监听服务器的 `127.0.0.1:8000`，本机需要开一个 SSH 隧道：

```bash
ssh -p <SSH_PORT> -L 8000:127.0.0.1:8000 <SSH_USER>@<SERVER_DOMAIN>
```

其中 `<SERVER_DOMAIN>` 只是服务器域名地址示例占位符，需要替换成自己的 GPU 服务器域名或 IP；`<SSH_PORT>` 和 `<SSH_USER>` 也需要按实际服务器配置填写。

这个终端需要保持打开。若后端已经通过公网地址暴露，则可以跳过隧道，直接把 `CCNLP_API_URL` 设置成公网 API 地址。

### 3. 在本机启动前端

另开一个本机终端：

```bash
cd /path/to/local/classical_chinese_project
conda activate classical-chinese-nlp

export CCNLP_API_URL=http://127.0.0.1:8000
streamlit run app.py
```

默认前端地址通常是：

```bash
http://localhost:8501
```

如果本机没有使用 SSH 隧道，而是直接访问云端 API，则改成：

```bash
export CCNLP_API_URL=http://<SERVER_DOMAIN_OR_IP>:8000
streamlit run app.py
```

## 准备古今平行语料

下载 NiuTrans/Classical-Modern 数据后，将原始目录转换为双向 JSONL：

```bash
git clone https://github.com/NiuTrans/Classical-Modern.git data/raw/Classical-Modern

python scripts/prepare_data.py \
  --format niutrans \
  --input "data/raw/Classical-Modern/双语数据" \
  --output_dir data/processed \
  --max_examples 50000
```

输出：

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
```

每个句对会扩展为两个方向：

```json
{"task": "classical_to_modern", "source": "古文翻今：学而时习之，不亦说乎", "target": "学习后按时温习，不也很快乐吗"}
{"task": "modern_to_classical", "source": "今文翻古：学习后按时温习，不也很快乐吗", "target": "学而时习之，不亦说乎"}
```

## 训练 Seq2Seq 基线

推荐从中文生成模型 `IDEA-CCNL/Randeng-BART-139M-SUMMARY` 开始：

```bash
PYTHONPATH=src python -m ccnlp.train_seq2seq \
  --train_file data/processed/train.jsonl \
  --validation_file data/processed/validation.jsonl \
  --model_name IDEA-CCNL/Randeng-BART-139M-SUMMARY \
  --output_dir outputs/checkpoints/randeng-bart-niutrans \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5
```

小样本冒烟测试：

```bash
PYTHONPATH=src python -m ccnlp.train_seq2seq \
  --train_file data/processed/train.jsonl \
  --model_name IDEA-CCNL/Randeng-BART-139M-SUMMARY \
  --output_dir outputs/checkpoints/debug-randeng-bart \
  --epochs 1 \
  --batch_size 2 \
  --max_train_samples 200
```

## 回环一致性训练

建议先训练一个可用的标准翻译模型，再在该 checkpoint 上继续进行回环一致性微调。不要从零训练时直接开启较大的 `lambda_cycle`，否则容易诱发复制坍缩。

```bash
PYTHONPATH=src python -m ccnlp.train_seq2seq \
  --train_file data/processed/train.jsonl \
  --validation_file data/processed/validation.jsonl \
  --model_name outputs/checkpoints/randeng-bart-niutrans \
  --output_dir outputs/checkpoints/randeng-bart-cycle-lambda0.1 \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --lambda_cycle 0.1
```

当前结果显示：`lambda_cycle=0.1` 在 BLEU、BERTScore 和幻觉率上最稳；更大的 lambda 可进一步提升回环可逆性，但会带来更多复制风险和轻微单向译质损失。

## 评估

翻译质量：

```bash
PYTHONPATH=src python -m ccnlp.eval_runner \
  --model_dir outputs/checkpoints/randeng-bart-niutrans \
  --test_file data/processed/test.jsonl \
  --output outputs/predictions.jsonl
```

回环一致性：

```bash
PYTHONPATH=src python -m ccnlp.rtc_eval \
  --model_dir outputs/checkpoints/randeng-bart-niutrans \
  --test_file data/processed/test.jsonl \
  --output outputs/rtc_roundtrip.jsonl
```

复制率诊断：

```bash
PYTHONPATH=src python -m ccnlp.copy_rate \
  --from_file outputs/rtc_roundtrip.jsonl
```

幻觉诊断：

```bash
PYTHONPATH=src python -m ccnlp.hallucination \
  --from_file outputs/rtc_roundtrip.jsonl
```

## 鲁迅风格 LoRA

将鲁迅风格平行数据转换为训练集：

```bash
python scripts/build_luxun_style_data.py \
  --input data/luxun_plain_pairs.filtered.jsonl \
  --output_dir data/processed/luxun_style
```

训练 Qwen3-4B LoRA：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m ccnlp.train_causal_lora \
  --train_file data/processed/luxun_style/train.jsonl \
  --validation_file data/processed/luxun_style/validation.jsonl \
  --model_name Qwen/Qwen3-4B \
  --output_dir outputs/checkpoints/qwen3-4b-luxun-lora \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_seq_length 512
```

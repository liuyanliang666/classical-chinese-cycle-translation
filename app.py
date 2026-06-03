from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ccnlp.examples import EXAMPLES
from ccnlp.inference import BaselineGenerator, TaskType


TASK_LABELS = {
    "古文翻今": TaskType.CLASSICAL_TO_MODERN,
    "今文翻古": TaskType.MODERN_TO_CLASSICAL,
    "鲁迅风格": TaskType.LUXUN_STYLE,
    "唐诗体": TaskType.TANG_POEM_STYLE,
}


@st.cache_resource
def load_generator() -> BaselineGenerator:
    return BaselineGenerator()


st.set_page_config(page_title="古文今译与风格迁移系统", page_icon="文", layout="wide")
st.title("古文今译 + 风格迁移双向系统")
st.caption("Classical Chinese ↔ Modern Chinese Translation & Style Transfer MVP")

generator = load_generator()

with st.sidebar:
    st.header("示例库")
    selected_example = st.selectbox("选择示例", EXAMPLES, format_func=lambda item: item["title"])
    if st.button("载入示例", use_container_width=True):
        st.session_state["input_text"] = selected_example["text"]
        st.session_state["task"] = selected_example["task"]

default_task = st.session_state.get("task", TaskType.CLASSICAL_TO_MODERN.value)
task_keys = list(TASK_LABELS.keys())
default_index = next(
    (idx for idx, key in enumerate(task_keys) if TASK_LABELS[key].value == default_task),
    0,
)

left, right = st.columns([1, 1])
with left:
    task_label = st.radio("任务类型", task_keys, index=default_index, horizontal=True)
    style_strength = st.slider("风格强度", 0.0, 1.0, 0.5, 0.1)
    input_text = st.text_area(
        "输入文本",
        value=st.session_state.get("input_text", "学而时习之，不亦说乎"),
        height=180,
    )
    run = st.button("生成", type="primary", use_container_width=True)

with right:
    st.subheader("生成结果")
    if run or input_text:
        result = generator.generate_with_metadata(input_text, TASK_LABELS[task_label], style_strength)
        st.write(result.output_text)
        st.info(result.note)

st.divider()
st.subheader("模型对比占位")
st.dataframe(
    [
        {"模型": "规则基线", "状态": "已接入", "说明": "保证无 GPU 时也能完成演示"},
        {"模型": "mT5 / BART 微调", "状态": "训练脚本骨架", "说明": "接入 Seq2SeqTrainer 后替换规则基线"},
        {"模型": "LoRA-Qwen 风格迁移", "状态": "后续扩展", "说明": "用于鲁迅风格、唐诗体等可控生成"},
    ],
    use_container_width=True,
)

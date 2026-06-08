from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ccnlp.inference import BaselineGenerator, TaskType
from ccnlp.ui_config import DEFAULT_INPUT, DEMO_EXAMPLES, STYLE_CARDS, STYLE_TASKS


TASK_BY_LABEL = {task.label: task.task for task in STYLE_TASKS}
TASK_META = {task.task: task for task in STYLE_TASKS}


@st.cache_resource
def load_generator() -> BaselineGenerator:
    return BaselineGenerator()


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #f6f5ef;
            --surface: #ffffff;
            --surface-soft: #f9faf7;
            --ink: #18202f;
            --muted: #667085;
            --line: rgba(24, 32, 47, 0.12);
            --accent: #9d3d2e;
            --accent-strong: #7f2f24;
            --jade: #2f6f67;
        }

        [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(180deg, rgba(157, 61, 46, 0.06), rgba(47, 111, 103, 0.06)),
                var(--paper);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background: #f0f2ed !important;
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] > div,
        div[data-testid="stSidebarUserContent"] {
            padding-left: 1.15rem !important;
            padding-right: 1.15rem !important;
        }

        section[data-testid="stSidebar"] * {
            color: var(--ink) !important;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted) !important;
        }

        .demo-hero {
            padding: 0.35rem 0 1.3rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.3rem;
        }

        .demo-kicker {
            color: var(--accent) !important;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 0.45rem;
        }

        .demo-title {
            color: var(--ink) !important;
            font-size: clamp(2rem, 3vw, 3.1rem);
            line-height: 1.12;
            font-weight: 800;
            letter-spacing: 0;
            margin: 0;
        }

        .demo-subtitle {
            color: var(--muted) !important;
            max-width: 780px;
            font-size: 1rem;
            line-height: 1.7;
            margin-top: 0.75rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 14px 36px rgba(24, 32, 47, 0.08);
        }

        .panel-title {
            color: var(--ink) !important;
            font-size: 1.05rem;
            font-weight: 750;
            margin: 0 0 0.25rem;
        }

        .panel-note {
            color: var(--muted) !important;
            font-size: 0.88rem;
            line-height: 1.55;
            margin-bottom: 1rem;
        }

        .result-box {
            min-height: 228px;
            color: var(--ink) !important;
            background: var(--surface-soft) !important;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            font-size: 1.05rem;
            line-height: 1.95;
            white-space: pre-wrap;
        }

        .result-placeholder {
            color: #8a92a1 !important;
        }

        .model-badge {
            display: inline-block;
            color: var(--accent-strong) !important;
            background: rgba(157, 61, 46, 0.1) !important;
            border: 1px solid rgba(157, 61, 46, 0.18);
            border-radius: 999px;
            padding: 0.2rem 0.62rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .model-card {
            min-height: 142px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
        }

        .model-card-title {
            color: var(--ink) !important;
            font-weight: 750;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }

        .model-card-backend {
            color: var(--jade) !important;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .model-card-description {
            color: var(--muted) !important;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .stTextArea textarea {
            min-height: 220px;
            color: var(--ink) !important;
            background: #ffffff !important;
            border: 1px solid rgba(24, 32, 47, 0.18) !important;
            border-radius: 8px;
            line-height: 1.75;
            font-size: 1rem;
        }

        .stTextArea textarea::placeholder {
            color: #8a92a1 !important;
        }

        .stRadio > label,
        .stTextArea > label,
        .stSlider > label {
            color: var(--ink) !important;
            font-weight: 650;
        }

        .stButton > button {
            min-height: 2.8rem;
            border-radius: 8px;
            border: 1px solid var(--accent-strong);
            background: var(--accent);
            color: white;
            font-weight: 750;
            letter-spacing: 0;
        }

        .stButton > button:hover {
            border-color: var(--accent-strong);
            background: var(--accent-strong);
            color: white;
        }

        .stButton > button:focus {
            box-shadow: 0 0 0 3px rgba(157, 61, 46, 0.2);
        }

        [data-baseweb="radio"] {
            background: rgba(255, 255, 255, 0.65);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.25rem 0.7rem;
            margin-right: 0.35rem;
        }

        [data-baseweb="radio"] *,
        div[data-baseweb="slider"] *,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3 {
            color: inherit;
        }

        [data-baseweb="radio"] span,
        [data-baseweb="radio"] p,
        [data-baseweb="radio"] div {
            color: var(--ink) !important;
        }

        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border-color: rgba(24, 32, 47, 0.18) !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {
            color: var(--ink) !important;
        }

        html body [data-testid="stAppViewContainer"] [data-baseweb="radio"] p,
        html body [data-testid="stAppViewContainer"] [data-baseweb="radio"] span,
        html body [data-testid="stAppViewContainer"] [data-baseweb="radio"] div,
        html body [data-testid="stAppViewContainer"] div[data-baseweb="slider"] p,
        html body [data-testid="stAppViewContainer"] div[data-baseweb="slider"] span,
        html body [data-testid="stAppViewContainer"] div[data-baseweb="slider"] div {
            color: var(--ink) !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    st.session_state.setdefault("input_text", DEFAULT_INPUT)
    st.session_state.setdefault("task", TaskType.MODERN_TO_CLASSICAL.value)
    st.session_state.setdefault("output_text", "")
    st.session_state.setdefault("output_note", "")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("展示样例")
        st.caption("样例均为现代文输入，可快速切换目标风格。")
        selected_example = st.selectbox(
            "选择样例",
            DEMO_EXAMPLES,
            format_func=lambda item: item.title,
        )
        if st.button("载入样例", use_container_width=True):
            st.session_state["input_text"] = selected_example.text
            st.session_state["task"] = selected_example.task.value
            st.session_state["output_text"] = ""
            st.session_state["output_note"] = ""

        st.divider()
        st.caption("当前版本先保留规则基线输出；真实 BART 和 Qwen 模型可在同一界面入口替换接入。")


def render_hero() -> None:
    st.markdown(
        """
        <div class="demo-hero">
          <div class="demo-kicker">CLASSICAL CHINESE NLP DEMO</div>
          <h1 class="demo-title">现代文风格转换系统</h1>
          <div class="demo-subtitle">
            输入一段现代中文，选择目标风格，系统输出对应的文言文表达或鲁迅式改写。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel(generator: BaselineGenerator) -> None:
    labels = [task.label for task in STYLE_TASKS]
    current_task = TaskType(st.session_state.get("task", TaskType.MODERN_TO_CLASSICAL.value))
    current_label = TASK_META.get(current_task, STYLE_TASKS[0]).label
    default_index = labels.index(current_label) if current_label in labels else 0

    left, right = st.columns([1.02, 0.98], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">输入与风格</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-note">输入保持为现代文，展示时只切换输出风格。</div>',
                unsafe_allow_html=True,
            )
            selected_label = st.radio(
                "输出风格",
                labels,
                index=default_index,
                horizontal=True,
            )
            selected_task = TASK_BY_LABEL[selected_label]
            st.session_state["task"] = selected_task.value

            style_strength = st.slider(
                "鲁迅风格强度",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05,
                help="仅影响鲁迅风格输出；文言文风格会忽略该值。",
            )
            input_text = st.text_area(
                "现代文输入",
                value=st.session_state["input_text"],
                placeholder="请输入一段现代中文。",
            )
            st.session_state["input_text"] = input_text

            run = st.button("生成结果", type="primary", use_container_width=True)
            if run:
                result = generator.generate_with_metadata(input_text, selected_task, style_strength)
                st.session_state["output_text"] = result.output_text
                st.session_state["output_note"] = result.note

    with right:
        with st.container(border=True):
            selected_task = TaskType(st.session_state.get("task", TaskType.MODERN_TO_CLASSICAL.value))
            meta = TASK_META.get(selected_task, STYLE_TASKS[0])
            output_text = st.session_state.get("output_text", "")
            output_note = st.session_state.get("output_note", "")

            st.markdown('<div class="panel-title">生成结果</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="model-badge">{html.escape(meta.backend)}</div>',
                unsafe_allow_html=True,
            )
            if output_text:
                safe_output = html.escape(output_text)
                st.markdown(f'<div class="result-box">{safe_output}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="result-box result-placeholder">点击“生成结果”后在这里查看输出。</div>',
                    unsafe_allow_html=True,
                )
            if output_note:
                st.info(output_note)


def render_model_plan() -> None:
    st.markdown("### 模型路线")
    columns = st.columns(len(STYLE_CARDS), gap="large")
    for column, card in zip(columns, STYLE_CARDS):
        with column:
            st.markdown(
                f"""
                <div class="model-card">
                  <div class="model-card-title">{html.escape(card.title)}</div>
                  <div class="model-card-backend">{html.escape(card.backend)}</div>
                  <div class="model-card-description">{html.escape(card.description)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(page_title="现代文风格转换系统", page_icon="文", layout="wide")
    apply_page_styles()
    initialize_state()

    generator = load_generator()
    render_sidebar()
    render_hero()
    render_input_panel(generator)
    render_model_plan()


if __name__ == "__main__":
    main()

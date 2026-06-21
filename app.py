from __future__ import annotations

import html
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ccnlp.inference import GenerationResult, TaskType
from ccnlp.ui_config import STYLE_CARDS, STYLE_TASKS


TASK_BY_LABEL = {task.label: task.task for task in STYLE_TASKS}
TASK_META = {task.task: task for task in STYLE_TASKS}
DEFAULT_TASK = TaskType.MODERN_TO_CLASSICAL
DEFAULT_API_URL = "http://127.0.0.1:8000"


class ApiGenerationClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 180.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("CCNLP_API_URL") or DEFAULT_API_URL).rstrip("/")
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen

    def generate_with_metadata(
        self,
        text: str,
        task: TaskType | str,
        style_strength: float = 1.0,
    ) -> GenerationResult:
        task_type = TaskType(task)
        payload = {
            "text": text,
            "task": task_type.value,
            "style_strength": style_strength,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/generate",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            response = self.opener(request, timeout=self.timeout)
            try:
                raw_body = response.read()
            finally:
                close = getattr(response, "close", None)
                if close is not None:
                    close()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接云端推理服务：{exc.reason}") from exc

        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"云端推理服务返回了无效 JSON：{exc}") from exc

        output = str(parsed.get("output", "")).strip()
        if not output:
            raise RuntimeError("云端推理服务返回了空结果")
        return GenerationResult(
            task=task_type,
            input_text=text,
            output_text=output,
            note=str(parsed.get("note") or "云端模型输出。"),
        )


@st.cache_resource
def load_generator() -> ApiGenerationClient:
    return ApiGenerationClient()


def task_label_for(task_value: str | TaskType) -> str:
    try:
        task = task_value if isinstance(task_value, TaskType) else TaskType(task_value)
    except ValueError:
        task = DEFAULT_TASK
    return TASK_META.get(task, STYLE_TASKS[0]).label


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@500;600;700;900&family=Noto+Sans+SC:wght@400;500;700&display=swap');
        :root {
            --paper: #f7f3e8;
            --paper-deep: #ebe1cc;
            --surface: #fffdf8;
            --surface-soft: #f8f4ea;
            --inkstone: #1f2733;
            --ink-soft: #47505f;
            --muted: #717989;
            --line: rgba(42, 37, 30, 0.14);
            --line-strong: rgba(42, 37, 30, 0.22);
            --cinnabar: #a33a2b;
            --cinnabar-deep: #76281e;
            --jade: #2f6f67;
            --gold: #b98b35;
            --shadow: 0 18px 42px rgba(31, 39, 51, 0.10);
            --shadow-soft: 0 10px 26px rgba(31, 39, 51, 0.07);
            --radius: 10px;
            --radius-lg: 14px;
            --ease: cubic-bezier(0.22, 0.61, 0.36, 1);
            --font-serif: "Noto Serif SC", "Source Han Serif SC", "Songti SC", "STSong", "SimSun", serif;
            --font-sans: "Noto Sans SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
        }

        [data-testid="stAppViewContainer"],
        section[data-testid="stSidebar"],
        [data-testid="stMarkdownContainer"],
        .stTextArea textarea,
        .stButton > button,
        .stDownloadButton > button {
            font-family: var(--font-sans);
        }

        .demo-title,
        .panel-title,
        .route-section-title,
        .sidebar-title,
        .route-card-title,
        .result-box {
            font-family: var(--font-serif);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 18% 10%, rgba(163, 58, 43, 0.08), transparent 28rem),
                linear-gradient(135deg, rgba(47, 111, 103, 0.08), transparent 34%),
                linear-gradient(180deg, rgba(255, 253, 248, 0.88), rgba(247, 243, 232, 0.96)),
                var(--paper);
        }

        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(31, 39, 51, 0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(31, 39, 51, 0.025) 1px, transparent 1px);
            background-size: 36px 36px;
            mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.72), transparent 78%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .main .block-container {
            max-width: 1220px;
            padding-top: 1.4rem;
            padding-bottom: 2.8rem;
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 253, 248, 0.95), rgba(235, 225, 204, 0.82)) !important;
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] > div,
        div[data-testid="stSidebarUserContent"] {
            padding-left: 1.15rem !important;
            padding-right: 1.15rem !important;
        }

        section[data-testid="stSidebar"] * {
            color: var(--inkstone) !important;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted) !important;
        }

        .sidebar-brand {
            border-bottom: 1px solid var(--line);
            margin-bottom: 1rem;
            padding-bottom: 0.95rem;
        }

        .sidebar-title {
            color: var(--inkstone) !important;
            font-size: 1.1rem;
            font-weight: 800;
            margin: 0 0 0.28rem;
        }

        .sidebar-copy {
            color: var(--muted) !important;
            font-size: 0.86rem;
            line-height: 1.6;
            margin: 0;
        }

        .demo-shell {
            position: relative;
            padding: 1.25rem 0 1.7rem;
            margin-bottom: 1.35rem;
            animation: fadeRise 0.6s var(--ease) both;
        }

        .demo-shell::before {
            content: "";
            position: absolute;
            left: 0;
            top: 1.35rem;
            width: 4px;
            height: 5.7rem;
            border-radius: 999px;
            background: linear-gradient(180deg, var(--cinnabar), var(--jade));
        }

        .demo-shell::after {
            content: "";
            position: absolute;
            left: 1.2rem;
            right: 0;
            bottom: 0;
            height: 2px;
            border-radius: 999px;
            background: linear-gradient(90deg,
                var(--cinnabar) 0%, var(--gold) 22%, var(--jade) 46%, transparent 82%);
            opacity: 0.6;
        }

        .seal {
            position: absolute;
            top: 1.2rem;
            right: 0.1rem;
            width: 3.5rem;
            height: 3.5rem;
            display: grid;
            place-items: center;
            transform: rotate(-7deg);
            border-radius: var(--radius);
            color: #fff;
            background: linear-gradient(160deg, var(--cinnabar), var(--cinnabar-deep));
            box-shadow: 0 8px 18px rgba(118, 40, 30, 0.28),
                inset 0 0 0 2px rgba(255, 255, 255, 0.38);
            font-family: var(--font-serif);
            font-weight: 900;
            font-size: 1.18rem;
            line-height: 1.02;
            letter-spacing: 0.05em;
            text-align: center;
            animation: sealStamp 0.6s var(--ease) both;
        }

        .demo-hero {
            padding-left: 1.2rem;
        }

        .demo-kicker {
            color: var(--cinnabar) !important;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 0.52rem;
        }

        .demo-title {
            color: var(--inkstone) !important;
            font-size: clamp(2.25rem, 3.2vw, 3.55rem);
            line-height: 1.14;
            font-weight: 800;
            letter-spacing: 0.02em;
            margin: 0;
        }

        .demo-subtitle {
            color: var(--ink-soft) !important;
            max-width: 780px;
            font-size: 1rem;
            line-height: 1.8;
            margin-top: 0.75rem;
        }

        .demo-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
        }

        .demo-stat {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            color: var(--ink-soft) !important;
            background: rgba(255, 253, 248, 0.68);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 0.42rem 0.7rem;
            font-size: 0.82rem;
            font-weight: 650;
            transition: transform 0.18s var(--ease), box-shadow 0.18s var(--ease),
                border-color 0.18s var(--ease);
        }

        .demo-stat:hover {
            transform: translateY(-1px);
            border-color: var(--gold);
            box-shadow: var(--shadow-soft);
        }

        .demo-stat-mark {
            color: var(--gold) !important;
            font-weight: 900;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 253, 248, 0.88);
            border: 1px solid var(--line-strong);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }

        .panel-title {
            color: var(--inkstone) !important;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0 0 0.25rem;
        }

        .panel-note {
            color: var(--muted) !important;
            font-size: 0.88rem;
            line-height: 1.55;
            margin-bottom: 1rem;
        }

        .result-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            margin-bottom: 0.85rem;
        }

        .result-style {
            color: var(--ink-soft) !important;
            font-size: 0.86rem;
            font-weight: 700;
        }

        .result-box {
            min-height: 250px;
            color: var(--inkstone) !important;
            background:
                linear-gradient(180deg, rgba(255, 253, 248, 0.96), rgba(248, 244, 234, 0.96)) !important;
            border: 1px solid var(--line-strong);
            border-radius: var(--radius);
            padding: 1.15rem 1.2rem;
            font-size: 1.08rem;
            line-height: 2.05;
            letter-spacing: 0.03em;
            white-space: pre-wrap;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.55);
            animation: fadeRise 0.5s var(--ease) both;
        }

        .result-placeholder {
            color: #8a8176 !important;
        }

        .result-loading {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.95rem;
        }

        .loading-ring {
            width: 2.7rem;
            height: 2.7rem;
            border-radius: 50%;
            border: 3px solid rgba(163, 58, 43, 0.18);
            border-top-color: var(--cinnabar);
            animation: spin 0.8s linear infinite;
        }

        .loading-text {
            color: var(--ink-soft) !important;
            font-size: 0.92rem;
            letter-spacing: 0.12em;
        }

        .loading-dots {
            display: inline-block;
            margin-left: 0.1em;
        }

        .loading-dots i {
            font-style: normal;
            opacity: 0;
            animation: dotPulse 1.4s ease-in-out infinite;
        }

        .loading-dots i:nth-child(2) {
            animation-delay: 0.2s;
        }

        .loading-dots i:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @keyframes dotPulse {
            0%, 100% { opacity: 0; }
            50% { opacity: 1; }
        }

        .model-badge {
            display: inline-flex;
            align-items: center;
            color: var(--cinnabar-deep) !important;
            background: rgba(163, 58, 43, 0.1) !important;
            border: 1px solid rgba(163, 58, 43, 0.2);
            border-radius: 999px;
            padding: 0.24rem 0.66rem;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .route-section-title {
            color: var(--inkstone) !important;
            font-size: 1.18rem;
            font-weight: 820;
            margin: 1.6rem 0 0.8rem;
        }

        .route-card {
            position: relative;
            overflow: hidden;
            min-height: 150px;
            background: rgba(255, 253, 248, 0.76);
            border: 1px solid var(--line-strong);
            border-radius: var(--radius);
            padding: 1.05rem 1.1rem;
            box-shadow: var(--shadow-soft);
            transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease),
                border-color 0.2s var(--ease);
        }

        .route-card:hover {
            transform: translateY(-3px);
            border-color: rgba(185, 139, 53, 0.55);
            box-shadow: 0 16px 34px rgba(31, 39, 51, 0.12);
        }

        .route-card::after {
            content: "";
            position: absolute;
            top: -1px;
            right: -1px;
            width: 2.6rem;
            height: 2.6rem;
            background: linear-gradient(225deg, rgba(185, 139, 53, 0.45), transparent 62%);
            border-top-right-radius: var(--radius);
            pointer-events: none;
        }

        .route-card-title {
            color: var(--inkstone) !important;
            font-weight: 800;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }

        .route-card-backend {
            color: var(--jade) !important;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .route-card-description {
            color: var(--muted) !important;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .stTextArea textarea {
            min-height: 245px;
            color: var(--inkstone) !important;
            background: var(--surface) !important;
            border: 1px solid var(--line-strong) !important;
            border-radius: var(--radius);
            line-height: 1.75;
            font-size: 1rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
        }

        .stTextArea textarea::placeholder {
            color: #8a8176 !important;
        }

        .stRadio > label,
        .stTextArea > label,
        .stSlider > label {
            color: var(--inkstone) !important;
            font-weight: 650;
        }

        .stButton > button {
            min-height: 2.8rem;
            border-radius: var(--radius);
            border: 1px solid var(--cinnabar-deep);
            background: linear-gradient(180deg, var(--cinnabar), var(--cinnabar-deep));
            color: white !important;
            font-weight: 750;
            letter-spacing: 0.02em;
            box-shadow: 0 9px 18px rgba(118, 40, 30, 0.18);
            transition: transform 0.18s var(--ease), box-shadow 0.18s var(--ease),
                background 0.18s var(--ease);
        }

        .stButton > button *,
        section[data-testid="stSidebar"] .stButton > button * {
            color: white !important;
        }

        .stButton > button:hover {
            border-color: var(--cinnabar-deep);
            background: var(--cinnabar-deep);
            color: white !important;
            transform: translateY(-1px);
            box-shadow: 0 13px 24px rgba(118, 40, 30, 0.26);
        }

        .stButton > button:active {
            transform: translateY(0);
            box-shadow: 0 6px 12px rgba(118, 40, 30, 0.2);
        }

        .stButton > button:focus {
            box-shadow: 0 0 0 3px rgba(163, 58, 43, 0.2);
        }

        [data-baseweb="radio"] {
            background: rgba(255, 253, 248, 0.74);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 0.3rem 0.72rem;
            margin-right: 0.35rem;
            transition: border-color 0.18s var(--ease), background 0.18s var(--ease),
                box-shadow 0.18s var(--ease);
        }

        [data-baseweb="radio"]:hover {
            border-color: var(--cinnabar);
            background: rgba(163, 58, 43, 0.06);
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
            color: var(--inkstone) !important;
        }

        div[data-baseweb="select"] > div {
            background: var(--surface) !important;
            border-color: var(--line-strong) !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {
            color: var(--inkstone) !important;
        }

        html body [data-testid="stAppViewContainer"] [data-baseweb="radio"] p,
        html body [data-testid="stAppViewContainer"] [data-baseweb="radio"] span,
        html body [data-testid="stAppViewContainer"] [data-baseweb="radio"] div,
        html body [data-testid="stAppViewContainer"] div[data-baseweb="slider"] p,
        html body [data-testid="stAppViewContainer"] div[data-baseweb="slider"] span,
        html body [data-testid="stAppViewContainer"] div[data-baseweb="slider"] div {
            color: var(--inkstone) !important;
        }

        div[data-testid="stAlert"] {
            border-radius: var(--radius);
            border-color: rgba(47, 111, 103, 0.22);
            background: rgba(47, 111, 103, 0.08);
        }

        .char-count {
            margin-top: -0.35rem;
            text-align: right;
            color: var(--muted) !important;
            font-size: 0.78rem;
            letter-spacing: 0.02em;
        }

        .stDownloadButton > button {
            min-height: 2.5rem;
            margin-top: 0.7rem;
            border-radius: var(--radius);
            border: 1px solid var(--line-strong);
            background: var(--surface);
            color: var(--ink-soft) !important;
            font-weight: 700;
            transition: transform 0.18s var(--ease), border-color 0.18s var(--ease),
                color 0.18s var(--ease), box-shadow 0.18s var(--ease);
        }

        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: var(--jade);
            color: var(--jade) !important;
            box-shadow: var(--shadow-soft);
        }

        @keyframes fadeRise {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes sealStamp {
            0% { opacity: 0; transform: scale(1.3) rotate(-7deg); }
            55% { opacity: 1; }
            100% { opacity: 1; transform: scale(1) rotate(-7deg); }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.001ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.001ms !important;
            }

            .loading-ring {
                animation-duration: 0.8s !important;
                animation-iteration-count: infinite !important;
            }
        }

        @media (max-width: 760px) {
            .demo-shell {
                padding-top: 0.6rem;
            }

            .demo-shell::before {
                height: 4.6rem;
            }

            .demo-title {
                font-size: 2.05rem;
            }

            .result-toolbar {
                align-items: flex-start;
                flex-direction: column;
            }

            .seal {
                display: none;
            }

            .demo-shell::after {
                left: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    st.session_state.setdefault("input_text", "")
    st.session_state.setdefault("task", DEFAULT_TASK.value)
    st.session_state.setdefault("task_label", task_label_for(st.session_state["task"]))
    st.session_state.setdefault("output_text", "")
    st.session_state.setdefault("output_note", "")


def render_hero() -> None:
    st.markdown(
        """
        <div class="demo-shell">
          <div class="seal" aria-hidden="true">古今</div>
          <div class="demo-hero">
            <div class="demo-kicker">CLASSICAL CHINESE NLP DEMO</div>
            <h1 class="demo-title">现代文风格转换系统</h1>
            <div class="demo-subtitle">
              输入一段现代中文，选择目标风格，系统输出对应的文言文表达或鲁迅式改写。
            </div>
            <div class="demo-stats">
              <span class="demo-stat"><span class="demo-stat-mark">01</span> 现代文输入</span>
              <span class="demo-stat"><span class="demo-stat-mark">02</span> 风格选择</span>
              <span class="demo-stat"><span class="demo-stat-mark">03</span> 生成展示</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel(generator: ApiGenerationClient) -> None:
    labels = [task.label for task in STYLE_TASKS]
    current_label = st.session_state.get("task_label", task_label_for(st.session_state.get("task", DEFAULT_TASK.value)))
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
                key="task_label",
            )
            selected_task = TASK_BY_LABEL[selected_label]
            st.session_state["task"] = selected_task.value

            style_strength = 0.65
            if selected_task == TaskType.LUXUN_STYLE:
                style_strength = st.slider(
                    "鲁迅风格强度",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.65,
                    step=0.05,
                    help="仅在选择鲁迅风格时显示，用于调节鲁迅风格化强度。",
                )
            input_text = st.text_area(
                "现代文输入",
                value=st.session_state["input_text"],
                placeholder="请输入一段现代中文。",
            )
            st.session_state["input_text"] = input_text
            st.markdown(
                f'<div class="char-count">{len(input_text.strip())} 字</div>',
                unsafe_allow_html=True,
            )

            run = st.button("生成结果", type="primary", use_container_width=True)

    with right:
        with st.container(border=True):
            selected_task = TaskType(st.session_state.get("task", TaskType.MODERN_TO_CLASSICAL.value))
            meta = TASK_META.get(selected_task, STYLE_TASKS[0])

            st.markdown('<div class="panel-title">生成结果</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="result-toolbar">
                  <div class="result-style">{html.escape(meta.label)}</div>
                  <div class="model-badge">{html.escape(meta.backend)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            result_slot = st.empty()
            if run:
                with result_slot.container():
                    st.markdown(
                        """
                        <div class="result-box result-loading">
                          <div class="loading-ring"></div>
                          <div class="loading-text">正在生成<span class="loading-dots"><i>.</i><i>.</i><i>.</i></span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                try:
                    result = generator.generate_with_metadata(input_text, selected_task, style_strength)
                except Exception as exc:  # noqa: BLE001 - surface remote inference failures in the UI.
                    st.session_state["output_text"] = ""
                    st.session_state["output_note"] = ""
                    st.error(f"生成失败：{exc}")
                else:
                    st.session_state["output_text"] = result.output_text
                    st.session_state["output_note"] = result.note

            output_text = st.session_state.get("output_text", "")
            output_note = st.session_state.get("output_note", "")
            with result_slot.container():
                if output_text:
                    safe_output = html.escape(output_text)
                    st.markdown(f'<div class="result-box">{safe_output}</div>', unsafe_allow_html=True)
                    st.download_button(
                        "下载结果",
                        data=output_text,
                        file_name="style_transfer_output.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-box result-placeholder">点击“生成结果”后在这里查看输出。</div>',
                        unsafe_allow_html=True,
                    )
            if output_note:
                st.info(output_note)


def render_model_plan() -> None:
    st.markdown('<div class="route-section-title">模型路线</div>', unsafe_allow_html=True)
    columns = st.columns(len(STYLE_CARDS), gap="large")
    for column, card in zip(columns, STYLE_CARDS):
        with column:
            st.markdown(
                f"""
                <div class="route-card">
                  <div class="route-card-title">{html.escape(card.title)}</div>
                  <div class="route-card-backend">{html.escape(card.backend)}</div>
                  <div class="route-card-description">{html.escape(card.description)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(page_title="现代文风格转换系统", page_icon="文", layout="wide")
    apply_page_styles()
    initialize_state()

    generator = load_generator()
    render_hero()
    render_input_panel(generator)
    render_model_plan()


if __name__ == "__main__":
    main()

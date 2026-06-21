import sys
import json
from types import SimpleNamespace

import pytest

from ccnlp.inference import TaskType


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


@pytest.fixture
def app_module(monkeypatch):
    fake_streamlit = SimpleNamespace(
        cache_resource=lambda func: func,
        markdown=lambda *args, **kwargs: None,
        set_page_config=lambda *args, **kwargs: None,
        session_state={},
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    sys.modules.pop("app", None)

    import app

    return app


def test_page_styles_include_refined_classical_theme(monkeypatch, app_module):
    rendered = []

    def fake_markdown(body, **kwargs):
        rendered.append(body)

    monkeypatch.setattr(app_module.st, "markdown", fake_markdown)

    app_module.apply_page_styles()

    css = "\n".join(rendered)

    assert "--cinnabar" in css
    assert "--inkstone" in css
    assert ".demo-shell" in css
    assert ".result-toolbar" in css
    assert ".route-card" in css


def test_hero_uses_refined_demo_shell(monkeypatch, app_module):
    rendered = []

    def fake_markdown(body, **kwargs):
        rendered.append(body)

    monkeypatch.setattr(app_module.st, "markdown", fake_markdown)

    app_module.render_hero()

    html = "\n".join(rendered)

    assert "demo-shell" in html
    assert "demo-stats" in html
    assert "现代文风格转换系统" in html


def test_initialize_state_keeps_task_label_in_sync(app_module):
    app_module.st.session_state.clear()
    app_module.st.session_state["task"] = TaskType.LUXUN_STYLE.value

    app_module.initialize_state()

    assert app_module.st.session_state["task_label"] == "鲁迅风格"
    assert app_module.st.session_state["input_text"] == ""


@pytest.mark.parametrize(
    ("selected_label", "expected_slider_calls"),
    [
        ("文言文风格", 0),
        ("鲁迅风格", 1),
    ],
)
def test_strength_slider_only_shows_for_luxun_style(monkeypatch, app_module, selected_label, expected_slider_calls):
    slider_calls = []

    app_module.st.session_state.clear()
    app_module.st.session_state["input_text"] = "测试文本"
    app_module.st.session_state["task_label"] = selected_label
    app_module.st.session_state["task"] = app_module.TASK_BY_LABEL[selected_label].value

    monkeypatch.setattr(app_module.st, "columns", lambda *args, **kwargs: [_Context(), _Context()], raising=False)
    monkeypatch.setattr(app_module.st, "container", lambda *args, **kwargs: _Context(), raising=False)
    monkeypatch.setattr(app_module.st, "radio", lambda *args, **kwargs: selected_label, raising=False)
    monkeypatch.setattr(app_module.st, "text_area", lambda *args, **kwargs: "测试文本", raising=False)
    monkeypatch.setattr(app_module.st, "button", lambda *args, **kwargs: False, raising=False)
    monkeypatch.setattr(app_module.st, "empty", lambda: SimpleNamespace(container=lambda: _Context()), raising=False)
    monkeypatch.setattr(app_module.st, "download_button", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(app_module.st, "info", lambda *args, **kwargs: None, raising=False)

    def fake_slider(*args, **kwargs):
        slider_calls.append((args, kwargs))
        return kwargs["value"]

    monkeypatch.setattr(app_module.st, "slider", fake_slider, raising=False)

    app_module.render_input_panel(generator=SimpleNamespace())

    assert len(slider_calls) == expected_slider_calls
    if slider_calls:
        assert slider_calls[0][0][0] == "鲁迅风格强度"


def test_api_generation_client_posts_generate_request(app_module):
    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps(
                {"output": "市中众人默然而过。", "note": "云端模型输出。"},
                ensure_ascii=False,
            ).encode("utf-8")

        def close(self):
            captured["closed"] = True

    def fake_opener(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["content_type"] = request.get_header("Content-type")
        return FakeResponse()

    client = app_module.ApiGenerationClient(
        base_url="http://127.0.0.1:8000/",
        timeout=9.0,
        opener=fake_opener,
    )

    result = client.generate_with_metadata(
        "街上很多人沉默地走过。",
        TaskType.LUXUN_STYLE,
        style_strength=0.8,
    )

    assert captured["url"] == "http://127.0.0.1:8000/generate"
    assert captured["timeout"] == 9.0
    assert captured["payload"] == {
        "text": "街上很多人沉默地走过。",
        "task": "luxun_style",
        "style_strength": 0.8,
    }
    assert captured["content_type"] == "application/json"
    assert captured["closed"] is True
    assert result.task == TaskType.LUXUN_STYLE
    assert result.output_text == "市中众人默然而过。"
    assert result.note == "云端模型输出。"

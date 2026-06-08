import pytest

from ccnlp.inference import apply_lora_style_strength, capture_lora_scaling_state


def test_apply_lora_style_strength_scales_from_original_values():
    model = FakeModel()
    state = capture_lora_scaling_state(model)

    apply_lora_style_strength(state, 0.5)
    assert model.lora.scaling == {"default": 4.0, "aux": 1.0}

    apply_lora_style_strength(state, 1.5)
    assert model.lora.scaling == {"default": 12.0, "aux": 3.0}


def test_apply_lora_style_strength_rejects_negative_values():
    model = FakeModel()
    state = capture_lora_scaling_state(model)

    with pytest.raises(ValueError, match="non-negative"):
        apply_lora_style_strength(state, -0.1)


class FakeModel:
    def __init__(self):
        self.lora = FakeLoraLayer()

    def named_modules(self):
        return [("plain", object()), ("lora", self.lora)]


class FakeLoraLayer:
    def __init__(self):
        self.scaling = {"default": 8.0, "aux": 2.0}

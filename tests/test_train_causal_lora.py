from ccnlp import train_causal_lora


def test_train_causal_lora_parser_defaults_to_qwen3_qlora():
    parser = train_causal_lora.build_parser()

    args = parser.parse_args(
        [
            "--train_file",
            "data/processed/luxun_style/train.jsonl",
            "--output_dir",
            "outputs/checkpoints/qwen3-4b-luxun-lora",
        ]
    )

    assert args.model_name == "Qwen/Qwen3-4B"
    assert args.load_in_4bit is True
    assert args.max_seq_length == 512
    assert args.batch_size == 1
    assert args.gradient_accumulation_steps == 8
    assert args.gradient_checkpointing_use_reentrant is False
    assert args.lora_r == 16
    assert "q_proj" in args.lora_target_modules
    assert "down_proj" in args.lora_target_modules


def test_train_causal_lora_parser_accepts_luxun_dataset_format():
    parser = train_causal_lora.build_parser()

    args = parser.parse_args(
        [
            "--train_file",
            "data/luxun_dataset.jsonl",
            "--output_dir",
            "outputs/checkpoints/debug",
            "--dataset_format",
            "luxungpt",
        ]
    )

    assert args.dataset_format == "luxungpt"


def test_device_map_uses_local_rank_in_distributed_training(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert train_causal_lora.resolve_device_map() == {"": 1}


def test_device_map_uses_auto_outside_distributed_training(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert train_causal_lora.resolve_device_map() == "auto"


def test_prepare_model_for_4bit_passes_non_reentrant_checkpointing_kwargs():
    parser = train_causal_lora.build_parser()
    args = parser.parse_args(
        [
            "--train_file",
            "data/processed/luxun_style/train.jsonl",
            "--output_dir",
            "outputs/checkpoints/qwen3-4b-luxun-lora",
        ]
    )
    model = FakeModel()
    calls = []

    def fake_prepare(model_arg, **kwargs):
        calls.append(kwargs)
        return model_arg

    result = train_causal_lora.prepare_model_for_training(model, args, fake_prepare)

    assert result is model
    assert model.config.use_cache is False
    assert calls == [
        {
            "use_gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }
    ]
    assert model.gradient_checkpointing_calls == []


def test_prepare_model_without_4bit_enables_non_reentrant_checkpointing_directly():
    parser = train_causal_lora.build_parser()
    args = parser.parse_args(
        [
            "--train_file",
            "data/processed/luxun_style/train.jsonl",
            "--output_dir",
            "outputs/checkpoints/qwen3-4b-luxun-lora",
            "--no-load_in_4bit",
        ]
    )
    model = FakeModel()

    result = train_causal_lora.prepare_model_for_training(
        model,
        args,
        lambda model_arg, **kwargs: model_arg,
    )

    assert result is model
    assert model.config.use_cache is False
    assert model.gradient_checkpointing_calls == [
        {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    ]


class FakeConfig:
    use_cache = True


class FakeModel:
    def __init__(self):
        self.config = FakeConfig()
        self.gradient_checkpointing_calls = []

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing_calls.append(kwargs)

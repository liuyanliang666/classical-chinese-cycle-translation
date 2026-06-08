from ccnlp import generate


def test_generate_cli_accepts_luxun_style_task():
    parser = generate.build_parser()

    args = parser.parse_args(
        [
            "--model_dir",
            "outputs/checkpoints/randeng-bart-luxun-style",
            "--input",
            "街上很多人沉默地走过。",
            "--task",
            "luxun_style",
        ]
    )

    assert args.task == "luxun_style"


def test_generate_cli_has_luxun_demo_label():
    assert generate._ARROW["luxun_style"] == "今→鲁迅"


def test_generate_cli_accepts_causal_lora_backend():
    parser = generate.build_parser()

    args = parser.parse_args(
        [
            "--backend",
            "causal_lora",
            "--base_model",
            "Qwen/Qwen3-4B",
            "--adapter_dir",
            "outputs/checkpoints/qwen3-4b-luxun-lora",
            "--input",
            "街上很多人沉默地走过。",
            "--task",
            "luxun_style",
        ]
    )

    assert args.backend == "causal_lora"
    assert args.base_model == "Qwen/Qwen3-4B"
    assert args.adapter_dir == "outputs/checkpoints/qwen3-4b-luxun-lora"


def test_generate_cli_accepts_lora_style_strength():
    parser = generate.build_parser()

    args = parser.parse_args(
        [
            "--backend",
            "causal_lora",
            "--base_model",
            "Qwen/Qwen3-4B",
            "--adapter_dir",
            "outputs/checkpoints/qwen3-4b-luxun-lora",
            "--input",
            "街上很多人沉默地走过。",
            "--task",
            "luxun_style",
            "--style_strength",
            "0.6",
        ]
    )

    assert args.style_strength == 0.6

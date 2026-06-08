from ccnlp import eval_causal_lora


def test_eval_causal_lora_parser_accepts_qwen_lora_inputs():
    parser = eval_causal_lora.build_parser()

    args = parser.parse_args(
        [
            "--base_model",
            "Qwen/Qwen3-4B",
            "--adapter_dir",
            "outputs/checkpoints/qwen3-4b-luxun-lora",
            "--test_file",
            "data/processed/luxun_style/test.jsonl",
            "--output",
            "outputs/checkpoints/qwen3-4b-luxun-lora/test_predictions.jsonl",
            "--no_bertscore",
        ]
    )

    assert args.base_model == "Qwen/Qwen3-4B"
    assert args.adapter_dir == "outputs/checkpoints/qwen3-4b-luxun-lora"
    assert args.task == "luxun_style"
    assert args.no_bertscore is True

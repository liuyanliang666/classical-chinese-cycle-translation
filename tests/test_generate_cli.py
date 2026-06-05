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

from ccnlp import train_seq2seq


def test_seq2seq_wandb_reporting_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_RUN_NAME", raising=False)
    monkeypatch.delenv("WANDB_WATCH", raising=False)

    args = train_seq2seq.build_parser().parse_args(
        ["--train_file", "data/processed/train.jsonl"]
    )

    report_to, run_name = train_seq2seq.configure_wandb_reporting(args)

    assert report_to == "none"
    assert run_name is None
    assert "WANDB_PROJECT" not in __import__("os").environ
    assert "WANDB_RUN_NAME" not in __import__("os").environ


def test_seq2seq_wandb_reporting_sets_project_and_run_name(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_RUN_NAME", raising=False)
    monkeypatch.delenv("WANDB_WATCH", raising=False)

    args = train_seq2seq.build_parser().parse_args(
        [
            "--train_file",
            "data/processed/train.jsonl",
            "--use_wandb",
            "--wandb_project",
            "classical-chinese-bart",
            "--wandb_run_name",
            "bart-cycle-lambda0.1-3ep",
        ]
    )

    report_to, run_name = train_seq2seq.configure_wandb_reporting(args)

    assert report_to == ["wandb"]
    assert run_name == "bart-cycle-lambda0.1-3ep"
    assert __import__("os").environ["WANDB_PROJECT"] == "classical-chinese-bart"
    assert __import__("os").environ["WANDB_RUN_NAME"] == "bart-cycle-lambda0.1-3ep"
    assert __import__("os").environ["WANDB_WATCH"] == "false"

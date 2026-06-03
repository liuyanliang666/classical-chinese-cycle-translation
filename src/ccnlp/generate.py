"""快速查看微调模型的翻译效果。

用法：
    PYTHONPATH=src python -m ccnlp.generate --model_dir <checkpoint目录>          # 跑内置示例
    PYTHONPATH=src python -m ccnlp.generate --model_dir <...> --input 学而时习之   # 单句
    PYTHONPATH=src python -m ccnlp.generate --model_dir <...> --interactive       # 交互输入
"""

from __future__ import annotations

import argparse

from ccnlp.inference import ModelGenerator

# (task, 文本) —— 两个方向各几条，直观看双向翻译质量。
DEMOS = [
    ("classical_to_modern", "学而时习之，不亦说乎"),
    ("classical_to_modern", "温故而知新，可以为师矣"),
    ("classical_to_modern", "三人行，必有我师焉"),
    ("modern_to_classical", "学习后按时温习，不也是很快乐的吗"),
    ("modern_to_classical", "自己不愿意的，不要强加给别人"),
]

_ARROW = {"classical_to_modern": "古→今", "modern_to_classical": "今→古"}


def main() -> None:
    parser = argparse.ArgumentParser(description="查看微调模型的翻译效果")
    parser.add_argument("--model_dir", required=True, help="训练输出目录或某个 checkpoint")
    parser.add_argument("--input", default=None, help="单句输入；不给则跑内置示例")
    parser.add_argument(
        "--task",
        choices=["classical_to_modern", "modern_to_classical"],
        default="classical_to_modern",
        help="翻译方向（仅对 --input 生效）",
    )
    parser.add_argument("--interactive", action="store_true", help="进入交互模式")
    parser.add_argument("--num_beams", type=int, default=4)
    args = parser.parse_args()

    generator = ModelGenerator(args.model_dir, num_beams=args.num_beams)
    print(f"已加载模型：{args.model_dir}（设备：{generator.device}）\n")

    if args.input is not None:
        print(generator.generate(args.input, args.task))
        return

    if args.interactive:
        print("交互模式。输入文本翻译；命令 :c 切古→今，:m 切今→古，:q 退出。")
        task = "classical_to_modern"
        while True:
            try:
                text = input(f"[{_ARROW[task]}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text == ":q":
                break
            if text == ":c":
                task = "classical_to_modern"
                continue
            if text == ":m":
                task = "modern_to_classical"
                continue
            if text:
                print(f"    → {generator.generate(text, task)}\n")
        return

    for task, text in DEMOS:
        print(f"[{_ARROW[task]}] {text}\n    → {generator.generate(text, task)}\n")


if __name__ == "__main__":
    main()

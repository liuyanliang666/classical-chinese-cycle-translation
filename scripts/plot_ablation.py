"""绘制 lambda_cycle 的受控消融曲线（修正版：平台，无悬崖）。

所有点均为「从对照组 A 的 3-epoch 权重继续微调 1 epoch、batch 16、lr 1e-5」，
只改变 lambda_cycle（变量受控）。因此整条曲线是一次干净的 λ 扫描。

生成两栏纵向子图（共享 x 轴 = lambda_cycle）：
  上：翻译质量 BLEU（左轴）+ 复制率（右轴）——BLEU 全程平稳、复制率温和上升
  下：完整 RTC 与三层一致性——随 λ 单调小幅上升（真实、非复制增益）
另用红色离群标记叠加「旧 B：从零训练（混淆变量）」一点，展示它如何脱离受控曲线。

用法：
  python scripts/plot_ablation.py
输出：figures/lambda_ablation.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 受控消融数据（均从 A + 1 epoch，只扫 lambda_cycle）----
# λ=0 用 matched 控制 Aplus1（A+1ep@λ0），λ=0.5 用 matched L05m（A+1ep@λ0.5）
LAM = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
BLEU = [25.13, 25.32, 25.20, 25.04, 25.08, 24.89]
COPY = [7.4, 9.6, 11.0, 11.7, 11.7, 13.2]            # 复制率 %
RTC = [0.804, 0.8123, 0.8174, 0.8219, 0.8263, 0.8289]
LITERAL = [0.477, 0.4951, 0.5058, 0.5156, 0.5241, 0.5305]
SEMANTIC = [0.8811, 0.8877, 0.8920, 0.8957, 0.8992, 0.9011]

# ---- 旧 B：从零初始权重训练 1 epoch（混淆变量），离群对照 ----
B_LAM, B_BLEU, B_COPY, B_RTC = 0.5, 22.60, 30.7, 0.8795

BLUE, RED, GREEN, GRAY = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"


def main() -> None:
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"hspace": 0.12}
    )

    # ---- 上栏：BLEU + 复制率 ----
    l1, = ax_top.plot(LAM, BLEU, "-o", color=BLUE, lw=2, label="BLEU（翻译质量，受控）")
    ax_top.set_ylabel("BLEU", color=BLUE)
    ax_top.tick_params(axis="y", labelcolor=BLUE)
    ax_top.set_ylim(21.5, 26.5)
    # 旧 B 的 BLEU 离群点
    bx, = ax_top.plot([B_LAM], [B_BLEU], "X", color=RED, ms=12, mew=2,
                      label="旧 B：从零训练（混淆变量）")

    ax_top_r = ax_top.twinx()
    l2, = ax_top_r.plot(LAM, COPY, "-s", color=RED, lw=2, label="复制率（%，受控）")
    ax_top_r.plot([B_LAM], [B_COPY], "X", color=RED, ms=12, mew=2)
    ax_top_r.set_ylabel("复制率 (%)", color=RED)
    ax_top_r.tick_params(axis="y", labelcolor=RED)
    ax_top_r.set_ylim(0, 35)

    ax_top.set_title("回环约束权重 λ 的受控消融（均从 A + 1 epoch 微调）",
                     fontsize=13, pad=10)
    ax_top.legend(handles=[l1, l2, bx], loc="center left", frameon=False, fontsize=9)
    ax_top_r.annotate(
        "旧 B 从零训练：复制率 30.7% / BLEU 崩盘\n（脱离受控曲线，非 λ=0.5 的真实表现）",
        xy=(0.5, 30.7), xytext=(0.10, 31),
        fontsize=9, color=RED, ha="left",
        arrowprops=dict(arrowstyle="->", color=RED),
    )

    # ---- 下栏：RTC 与三层一致性 ----
    ax_bot.plot(LAM, RTC, "-o", color=GREEN, lw=2.2, label="完整 RTC（受控）")
    ax_bot.plot(LAM, SEMANTIC, "-^", color=GRAY, lw=1.5, alpha=0.8, label="语义一致性")
    ax_bot.plot(LAM, LITERAL, "-v", color="#9467bd", lw=1.5, alpha=0.8, label="字面一致性")
    ax_bot.plot([B_LAM], [B_RTC], "X", color=RED, ms=12, mew=2, label="旧 B（复制虚高）")
    ax_bot.set_ylabel("一致性分数")
    ax_bot.set_xlabel("lambda_cycle（回环一致性损失权重）")
    ax_bot.set_ylim(0.45, 0.97)
    ax_bot.legend(loc="center left", frameon=False, fontsize=9)
    ax_bot.annotate(
        "受控 RTC 随 λ 单调小幅上升\n（真实信息保留，非复制）",
        xy=(0.5, 0.8289), xytext=(0.06, 0.74),
        fontsize=9, color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN),
    )
    ax_bot.annotate(
        "旧 B 的 RTC 由 30.7% 复制虚高",
        xy=(0.5, 0.8795), xytext=(0.18, 0.93),
        fontsize=9, color=RED, ha="left",
        arrowprops=dict(arrowstyle="->", color=RED),
    )

    out = Path("figures/lambda_ablation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"图已保存：{out}")


if __name__ == "__main__":
    main()

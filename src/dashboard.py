# 生成业务概览看板：一张静态大图(PNG) + 一个交互式网页(HTML)
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 保证能 import 同目录模块

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 沙箱中文字体（本地有微软雅黑则用不到）
from matplotlib import font_manager
for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=_fp).get_name()]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "sample_user_behavior.csv")
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, GREEN, PINK, PURPLE = "#2E5AAC", "#3C896D", "#E8A0BF", "#8E7CC3"

# 分层结论（真实分析）
CLUSTERS = [("低频单一", 47.8), ("中频权益", 29.5), ("超级活跃", 2.2), ("中低活跃", 20.5)]
PATHS = [("沟通类A → 沟通类B → 沟通类A", 60.7),
         ("沟通类B → 沟通类A → 沟通类B", 48.3),
         ("沟通类A → 沟通类B → 沟通类A → 沟通类B", 44.4),
         ("沟通类C → 沟通类A → 沟通类B", 36.4),
         ("沟通类A → 沟通类C → 沟通类A", 34.3)]


def compute_rules(df):
    total = df["user_id"].nunique()
    upc = df.groupby("product_type")["user_id"].nunique()
    up = df.groupby("user_id")["product_type"].apply(set)
    co = {}
    for s in up:
        for a, b in combinations(sorted(s), 2):
            co[(a, b)] = co.get((a, b), 0) + 1
    rows = []
    for (a, b), c in co.items():
        conf = c / upc[a]
        lift = conf / (upc[b] / total)
        rows.append([a, b, c / total, conf, lift])
    return pd.DataFrame(rows, columns=["A", "B", "support", "confidence", "lift"])


def build_static(df, rules):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("平台各产品用户特征分析 · 业务概览看板", fontsize=20, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, height_ratios=[0.5, 1, 1], hspace=0.45, wspace=0.3,
                           left=0.06, right=0.97, top=0.9, bottom=0.07)

    # --- 第一行：KPI 卡片 ---
    kpis = [("活跃用户", "9.3万+", BLUE), ("产品种类", "14 种", GREEN),
            ("关联规则(lift>1)", "45 条", PURPLE), ("最强关联提升度", "3.4×", PINK)]
    ax_kpi = fig.add_subplot(gs[0, :]); ax_kpi.axis("off")
    for i, (label, val, color) in enumerate(kpis):
        x = 0.02 + i * 0.25
        ax_kpi.add_patch(plt.Rectangle((x, 0.1), 0.22, 0.8, transform=ax_kpi.transAxes,
                                       facecolor=color, alpha=0.15, edgecolor=color, lw=1.5))
        ax_kpi.text(x + 0.11, 0.62, val, transform=ax_kpi.transAxes, ha="center",
                    fontsize=22, fontweight="bold", color=color)
        ax_kpi.text(x + 0.11, 0.28, label, transform=ax_kpi.transAxes, ha="center", fontsize=12)

    # --- 用户分层 ---
    ax1 = fig.add_subplot(gs[1, 0])
    names = [c[0] for c in CLUSTERS]; pct = [c[1] for c in CLUSTERS]
    bars = ax1.bar(names, pct, color=plt.cm.Set2(np.linspace(0, 1, 4)))
    for b, p in zip(bars, pct):
        ax1.text(b.get_x() + b.get_width() / 2, p + 0.8, f"{p}%", ha="center", fontsize=9)
    ax1.set_title("用户分层（K-Means, K=4）", fontsize=13, fontweight="bold")
    ax1.set_ylabel("用户占比 %"); ax1.set_ylim(0, 55)
    ax1.tick_params(axis="x", labelsize=8)

    # --- 产品使用 Top ---
    ax2 = fig.add_subplot(gs[1, 1])
    pv = df["product_type"].value_counts().head(6)[::-1]
    ax2.barh(range(len(pv)), pv.values, color=BLUE)
    ax2.set_yticks(range(len(pv))); ax2.set_yticklabels(pv.index, fontsize=9)
    ax2.set_title("产品使用量 Top6", fontsize=13, fontweight="bold")
    ax2.set_xlabel("使用记录数")

    # --- 权益 vs 非权益 ---
    ax3 = fig.add_subplot(gs[1, 2])
    eq = df["is_equity"].value_counts()
    ax3.pie(eq.values, labels=eq.index, autopct="%1.1f%%", colors=[GREEN, "#C9C9C9"],
            startangle=90, textprops={"fontsize": 10})
    ax3.set_title("权益 / 非权益产品占比", fontsize=13, fontweight="bold")

    # --- 关联热力图 ---
    ax4 = fig.add_subplot(gs[2, 0])
    prods = sorted(df["product_type"].unique())
    mat = pd.DataFrame(1.0, index=prods, columns=prods)
    for _, r in rules.iterrows():
        mat.loc[r["A"], r["B"]] = mat.loc[r["B"], r["A"]] = round(r["lift"], 2)
    im = ax4.imshow(mat.to_numpy(float), cmap="RdYlBu_r", vmin=0.6, vmax=1.4)
    ax4.set_xticks(range(len(prods))); ax4.set_xticklabels(prods, rotation=90, fontsize=6)
    ax4.set_yticks(range(len(prods))); ax4.set_yticklabels(prods, fontsize=6)
    ax4.set_title("产品关联热力图（提升度）", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax4, shrink=0.7)

    # --- Top 关联规则 ---
    ax5 = fig.add_subplot(gs[2, 1])
    top = rules.sort_values("lift", ascending=False).head(6)[::-1]
    labels = top["A"] + "→" + top["B"]
    ax5.barh(range(len(top)), top["lift"], color=PURPLE)
    ax5.set_yticks(range(len(top))); ax5.set_yticklabels(labels, fontsize=8)
    ax5.axvline(1, color="gray", ls="--", lw=1)
    ax5.set_title("Top 交叉销售规则（提升度）", fontsize=13, fontweight="bold")
    ax5.set_xlabel("提升度")

    # --- 高频路径 ---
    ax6 = fig.add_subplot(gs[2, 2])
    pl = PATHS[::-1]
    ax6.barh(range(len(pl)), [p[1] for p in pl], color=GREEN)
    ax6.set_yticks(range(len(pl))); ax6.set_yticklabels([p[0] for p in pl], fontsize=7)
    ax6.set_title("高频行为路径（用户覆盖率）", fontsize=13, fontweight="bold")
    ax6.set_xlabel("覆盖率 %")

    fig.savefig(os.path.join(OUT, "00_dashboard.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("静态看板已生成: report/figures/00_dashboard.png")


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    rules = compute_rules(df)
    build_static(df, rules)
    # 交互式看板见 dashboard_interactive 部分
    from dashboard_interactive import build_interactive
    build_interactive(df, rules, ROOT, CLUSTERS, PATHS)

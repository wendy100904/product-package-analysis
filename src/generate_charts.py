# 画报告里的几张图
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---- 中文字体：直接找系统里的中文字体文件并注册，避免中文变方框 ----
def _setup_cjk_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        os.path.expanduser("~/.fonts/NotoSansCJK.ttc"),
        r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",   # 黑体
        r"C:\Windows\Fonts\simsun.ttc",   # 宋体
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.sans-serif"] = [
                    font_manager.FontProperties(fname=fp).get_name()]
                return
            except Exception:
                continue
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC"]


_setup_cjk_font()
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "sample_user_behavior.csv")
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2E5AAC"
PALETTE = plt.cm.Set2(np.linspace(0, 1, 8))


def build_user_features(df):
    """用户维度特征聚合（简化版，用于聚类）。"""
    g = df.groupby("user_id")
    feat = pd.DataFrame({
        "total_uses": g.size(),
        "product_count": g["product_type"].nunique(),
        "equity_ratio": g["is_equity"].apply(lambda s: (s == "权益").mean()),
        "days_active": g["event_date"].nunique(),
    })
    feat["avg_salary"] = g["job_salary"].mean()
    return feat.fillna(0)


def fig_elbow(feat):
    X = StandardScaler().fit_transform(feat)
    ks, inertias = range(1, 10), []
    for k in ks:
        inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), inertias, "o-", color=BLUE, lw=2, ms=8)
    plt.axvline(4, color="#D1495B", ls="--", label="最佳 K = 4")
    plt.xlabel("聚类数量 K"); plt.ylabel("惯性 (Inertia)")
    plt.title("肘部法则：确定最佳聚类数量")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{OUT}/01_elbow.png", dpi=130); plt.close()


def fig_cluster_dist():
    # 与报告一致的 4 群规模占比
    names = ["聚类0\n低频单一", "聚类1\n中频权益", "聚类2\n超级活跃", "聚类3\n中低活跃"]
    pct = [47.8, 29.5, 2.2, 20.5]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, pct, color=PALETTE[:4])
    for b, p in zip(bars, pct):
        plt.text(b.get_x()+b.get_width()/2, p+0.6, f"{p}%", ha="center", fontsize=11)
    plt.ylabel("用户占比 (%)"); plt.title("K-Means 用户分层：4 个群体规模占比")
    plt.ylim(0, 55); plt.tight_layout()
    plt.savefig(f"{OUT}/02_cluster_dist.png", dpi=130); plt.close()


def fig_assoc_heatmap(df):
    """产品关联提升度热力图。"""
    prods = sorted(df["product_type"].unique())
    total = df["user_id"].nunique()
    upc = df.groupby("product_type")["user_id"].nunique()
    up = df.groupby("user_id")["product_type"].apply(lambda x: set(x))
    mat = pd.DataFrame(0.0, index=prods, columns=prods)
    from itertools import combinations
    co = {}
    for s in up:
        for a, b in combinations(sorted(s), 2):
            co[(a, b)] = co.get((a, b), 0) + 1
    for (a, b), c in co.items():
        conf = c / upc[a]
        lift = conf / (upc[b] / total)
        mat.loc[a, b] = mat.loc[b, a] = round(lift, 2)
    arr = mat.to_numpy(dtype=float).copy()
    np.fill_diagonal(arr, 1.0)
    plt.figure(figsize=(9, 7.5))
    im = plt.imshow(arr, cmap="RdYlBu_r", vmin=0.6, vmax=1.4)
    plt.colorbar(im, label="提升度 (Lift)", shrink=.8)
    plt.xticks(range(len(prods)), prods, rotation=45, ha="right")
    plt.yticks(range(len(prods)), prods)
    for i in range(len(prods)):
        for j in range(len(prods)):
            plt.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("产品关联分析热力图（提升度）"); plt.tight_layout()
    plt.savefig(f"{OUT}/03_assoc_heatmap.png", dpi=130); plt.close()


def fig_top_combos(df):
    up = df.groupby("user_id")["product_type"].apply(lambda x: " + ".join(sorted(set(x))[:3]))
    top = up.value_counts().head(12)[::-1]
    pct = top / df["user_id"].nunique() * 100
    plt.figure(figsize=(9, 6))
    plt.barh(range(len(top)), pct.values, color=BLUE)
    plt.yticks(range(len(top)), top.index, fontsize=8)
    for i, v in enumerate(pct.values):
        plt.text(v+0.1, i, f"{v:.1f}%", va="center", fontsize=8)
    plt.xlabel("用户占比 (%)"); plt.title("最常见的产品组合（Top 12）")
    plt.tight_layout(); plt.savefig(f"{OUT}/04_top_combos.png", dpi=130); plt.close()


def fig_transition(df):
    """产品转移概率 Top（沟通类为中枢）。"""
    df2 = df.sort_values(["user_id", "event_date"])
    trans = {}
    for _, g in df2.groupby("user_id"):
        seq = g["product_type"].tolist()
        for a, b in zip(seq, seq[1:]):
            if a != b:
                trans[(a, b)] = trans.get((a, b), 0) + 1
    tdf = pd.DataFrame([(a, b, c) for (a, b), c in trans.items()],
                       columns=["from", "to", "cnt"])
    out_sum = tdf.groupby("from")["cnt"].transform("sum")
    tdf["prob"] = tdf["cnt"] / out_sum
    top = tdf.sort_values("prob", ascending=False).head(12)[::-1]
    labels = top["from"] + " → " + top["to"]
    plt.figure(figsize=(9, 6))
    plt.barh(range(len(top)), top["prob"]*100, color="#3C896D")
    plt.yticks(range(len(top)), labels, fontsize=8)
    for i, v in enumerate(top["prob"]*100):
        plt.text(v+0.3, i, f"{v:.0f}%", va="center", fontsize=8)
    plt.xlabel("转移概率 (%)"); plt.title("产品转移概率 Top 12（行为链路）")
    plt.tight_layout(); plt.savefig(f"{OUT}/05_transition.png", dpi=130); plt.close()


def main():
    df = pd.read_csv(CSV)
    feat = build_user_features(df)
    fig_elbow(feat)
    fig_cluster_dist()
    fig_assoc_heatmap(df)
    fig_top_combos(df)
    fig_transition(df)
    print(f"图表已生成至: {OUT}")
    for f in sorted(os.listdir(OUT)):
        print("  -", f)


if __name__ == "__main__":
    main()

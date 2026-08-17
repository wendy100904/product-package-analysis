# 交互式看板（Plotly），导出单个 HTML
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_interactive(df, rules, root, clusters, paths):
    prods = sorted(df["product_type"].unique())
    mat = pd.DataFrame(1.0, index=prods, columns=prods)
    for _, r in rules.iterrows():
        mat.loc[r["A"], r["B"]] = mat.loc[r["B"], r["A"]] = round(r["lift"], 2)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("用户分层（K-Means, K=4）", "产品关联热力图（提升度）",
                        "Top 交叉销售规则（提升度）", "高频行为路径（用户覆盖率）"),
        specs=[[{"type": "bar"}, {"type": "heatmap"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.14, horizontal_spacing=0.12)

    # 1. 分层
    fig.add_trace(go.Bar(
        x=[c[0] for c in clusters], y=[c[1] for c in clusters],
        marker_color=["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"],
        text=[f"{c[1]}%" for c in clusters], textposition="outside",
        hovertemplate="%{x}<br>占比 %{y}%<extra></extra>"), row=1, col=1)

    # 2. 热力图
    fig.add_trace(go.Heatmap(
        z=mat.to_numpy(float), x=prods, y=prods, colorscale="RdYlBu_r",
        zmid=1, zmin=0.6, zmax=1.4, colorbar=dict(title="提升度", len=0.4, y=0.8),
        hovertemplate="%{y} → %{x}<br>提升度 %{z}<extra></extra>"), row=1, col=2)

    # 3. Top 规则
    top = rules.sort_values("lift", ascending=False).head(8)[::-1]
    fig.add_trace(go.Bar(
        x=top["lift"], y=(top["A"] + " → " + top["B"]), orientation="h",
        marker_color="#8E7CC3",
        customdata=top[["support", "confidence"]].round(3).values,
        hovertemplate="%{y}<br>提升度 %{x:.2f}<br>支持度 %{customdata[0]}<br>置信度 %{customdata[1]}<extra></extra>"),
        row=2, col=1)

    # 4. 路径
    pl = paths[::-1]
    fig.add_trace(go.Bar(
        x=[p[1] for p in pl], y=[p[0] for p in pl], orientation="h",
        marker_color="#3C896D", text=[f"{p[1]}%" for p in pl], textposition="outside",
        hovertemplate="%{y}<br>覆盖率 %{x}%<extra></extra>"), row=2, col=2)

    fig.update_layout(
        title=dict(text="平台各产品用户特征分析 · 交互式看板",
                   font=dict(size=22), x=0.5),
        showlegend=False, height=820, width=1200,
        font=dict(family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif"),
        margin=dict(t=90, l=60, r=40, b=40))
    fig.update_xaxes(title_text="用户占比 %", row=1, col=1)
    fig.update_xaxes(title_text="提升度", row=2, col=1)
    fig.update_xaxes(title_text="覆盖率 %", row=2, col=2)

    out = os.path.join(root, "report", "dashboard.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print("交互式看板已生成: report/dashboard.html")

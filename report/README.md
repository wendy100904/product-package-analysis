# 报告

| 文件 | 内容 |
|---|---|
| [`analysis_report.md`](./analysis_report.md) | 分析报告：分层 / 关联 / 路径 / 建议 |
| [`appendix_rules.md`](./appendix_rules.md) | 45 条关联规则明细，按提升度排序 |
| [`figures/`](./figures) | 图表 |

`figures/` 里的图由 [`src/generate_charts.py`](../src/generate_charts.py) 生成。产品名用代号（沟通类/触达类/推广类/增值类/功能类），数字用占比。

重新生成：`python src/data/generate_sample_data.py && python src/generate_charts.py`

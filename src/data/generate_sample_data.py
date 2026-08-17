# -*- coding: utf-8 -*-
"""
脱敏模拟数据生成脚本（Synthetic Dataset）
========================================
生成一份结构与生产数据完全一致、但内容完全随机的示例数据，
使任何人 clone 仓库后都能直接跑通 K-Means 聚类与产品关联分析。

数据零泄密：所有 ID 为随机生成的伪 ID，主体名称均为代号（Company_*, City_Tier_*）。

用法:
    python src/data/generate_sample_data.py
    # 输出 -> data/sample_user_behavior.csv
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_USERS = 800          # 用户数
MAX_EVENTS_PER_USER = 25  # 每用户最多行为记录数

PRODUCTS = ["职位发布", "简历搜索", "在线沟通", "简历查看", "权益礼包",
            "急聘推广", "置顶刷新", "企业认证"]
EQUITY_MAP = {"权益礼包": "权益", "急聘推广": "权益", "置顶刷新": "权益"}
CITY_TIERS = ["一线城市", "新一线城市", "二线城市", "三线城市", "四线城市", "五线城市"]
ORG_TYPES = ["KA", "SME"]
COMPANY_SCALES = ["1-49人", "50-99人", "100-499人", "500-999人",
                  "1000-2000人", "2000-5000人", "5000-10000人", "10000人以上"]
JOB_TITLES = ["销售", "研发", "运营", "市场", "客服", "行政", "财务", "产品"]


def pseudo_id(prefix, i):
    """生成伪 ID（不含任何真实标识）。"""
    return f"{prefix}_{i:06d}"


def gen():
    rows = []
    base = pd.Timestamp("2026-01-01")
    for u in range(N_USERS):
        uid = pseudo_id("USR", u)
        n_events = np.random.randint(1, MAX_EVENTS_PER_USER)
        # 用户级属性（同一用户保持一致）
        city = np.random.choice(CITY_TIERS)
        org = np.random.choice(ORG_TYPES, p=[0.25, 0.75])
        scale = np.random.choice(COMPANY_SCALES)
        title = np.random.choice(JOB_TITLES)
        for _ in range(n_events):
            prod = np.random.choice(PRODUCTS)
            day = int(np.random.randint(0, 60))
            dt = base + pd.Timedelta(days=day)
            active = np.random.rand() < 0.7
            rows.append({
                "user_id": uid,
                "active_user_id": uid if active else np.nan,
                "job_post_user_id": uid if prod == "职位发布" else np.nan,
                "search_user_id": uid if prod == "简历搜索" else np.nan,
                "chat_user_id": uid if prod == "在线沟通" else np.nan,
                "resume_view_user_id": uid if prod == "简历查看" else np.nan,
                "event_date": dt.strftime("%Y-%m-%d"),
                "stat_date": dt.strftime("%Y-%m-%d"),
                "publish_date": dt.strftime("%Y-%m-%d"),
                "product_type": prod,
                "is_equity": EQUITY_MAP.get(prod, "非权益"),
                "job_salary": round(float(np.random.exponential(scale=18) + 5), 1),
                "city_tier_level": city,
                "org_type_name": org,
                "company_scale_type": scale,
                "jobtitle1": title,
            })
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_user_behavior.csv")
    df = gen()
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已生成脱敏示例数据: {out_path}")
    print(f"形状: {df.shape}, 用户数: {df['user_id'].nunique()}")

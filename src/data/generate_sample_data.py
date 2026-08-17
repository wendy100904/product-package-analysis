# 造一批测试用的假数据，跑通整套流程用
import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_USERS = 800          # 用户数
MAX_EVENTS_PER_USER = 25  # 每用户最多行为记录数

PRODUCTS = ["沟通类A", "沟通类B", "沟通类C", "触达类A", "触达类B",
            "推广类A", "推广类B", "增值类A", "增值类B", "功能类A"]
# 权益产品（与报告口径一致）：沟通类B/C、推广类A/B、触达类B 视为权益
EQUITY_MAP = {"沟通类B": "权益", "沟通类C": "权益", "推广类A": "权益",
              "推广类B": "权益", "触达类B": "权益"}
CITY_TIERS = ["一线城市", "新一线城市", "二线城市", "三线城市", "四线城市", "五线城市"]
ORG_TYPES = ["KA", "SME"]
COMPANY_SCALES = ["1-49人", "50-99人", "100-499人", "500-999人",
                  "1000-2000人", "2000-5000人", "5000-10000人", "10000人以上"]
JOB_TITLES = ["销售", "研发", "运营", "市场", "客服", "行政", "财务", "产品"]


def make_id(prefix, i):
    return f"{prefix}_{i:06d}"


def gen():
    rows = []
    base = pd.Timestamp("2026-01-01")
    for u in range(N_USERS):
        uid = make_id("USR", u)
        n_events = np.random.randint(1, MAX_EVENTS_PER_USER)
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
                "job_post_user_id": uid if prod == "触达类A" else np.nan,
                "search_user_id": uid if prod == "功能类A" else np.nan,
                "chat_user_id": uid if prod in ("沟通类A", "沟通类B", "沟通类C") else np.nan,
                "resume_view_user_id": uid if prod == "增值类B" else np.nan,
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
    print(f"已生成: {out_path}")
    print(f"形状: {df.shape}, 用户数: {df['user_id'].nunique()}")

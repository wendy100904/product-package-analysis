"""
数据加载模块
=============
生产环境通过 Spark + Hive 读取数据；作品集环境通过环境变量指向脱敏的
示例 CSV（见 data/generate_sample_data.py 生成的 sample_user_behavior.csv）。

数据安全说明：
- 不含任何真实数据库连接串、IP、账号密码或内部临时表名。
- 生产表名统一由环境变量 SOURCE_TABLE 注入，默认走本地示例数据。
"""
import os
import pandas as pd

# 生产表名通过环境变量注入，默认占位（脱敏）
SOURCE_TABLE = os.getenv("SOURCE_TABLE", "dw_layer.fact_user_daily_behavior")
# 作品集复现路径：默认读取本地脱敏示例数据
DATA_PATH = os.getenv("ANALYTICS_DATA_PATH", "./data/sample_user_behavior.csv")


def load_from_spark():
    """生产环境：从 Spark/Hive 读取数据（作品集中通常不执行此分支）。"""
    from pyspark.sql import SparkSession
    from pyspark import SparkConf

    spark_conf = SparkConf()
    spark_conf.setAppName("b2b_user_behavior_analysis")
    spark = (
        SparkSession.builder.config(conf=spark_conf)
        .enableHiveSupport()
        .getOrCreate()
    )
    print(f"1. 从 Spark 读取数据: {SOURCE_TABLE}")
    return spark.sql(f"SELECT * FROM {SOURCE_TABLE}").toPandas()


def load_data():
    """
    统一数据入口。
    优先读取本地脱敏示例 CSV，保证任何人 clone 仓库即可复现整条分析链路。
    """
    print("1. 数据加载...")
    if os.path.exists(DATA_PATH):
        print(f"  读取示例数据: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("  未找到示例数据，尝试从 Spark 读取（需生产环境）...")
        df = load_from_spark()

    print(f"数据形状: {df.shape}")
    print(f"数据列名: {df.columns.tolist()}")
    return df

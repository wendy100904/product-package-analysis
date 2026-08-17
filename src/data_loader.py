import os
import pandas as pd

# 生产环境读 Hive，本地跑读 csv
SOURCE_TABLE = os.getenv("SOURCE_TABLE", "dw_layer.fact_user_daily_behavior")
DATA_PATH = os.getenv("ANALYTICS_DATA_PATH", "./data/sample_user_behavior.csv")


def load_from_spark():
    from pyspark.sql import SparkSession
    from pyspark import SparkConf

    conf = SparkConf()
    conf.setAppName("user_behavior_analysis")
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    print(f"从 Spark 读取: {SOURCE_TABLE}")
    return spark.sql(f"SELECT * FROM {SOURCE_TABLE}").toPandas()


def load_data():
    print("1. 数据加载...")
    if os.path.exists(DATA_PATH):
        print(f"  读取: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        df = load_from_spark()

    print(f"数据形状: {df.shape}")
    print(f"数据列名: {df.columns.tolist()}")
    return df

# 用户分层 + 产品关联分析主流程
# 跑之前先执行 src/data/generate_sample_data.py 生成数据
import matplotlib
matplotlib.use("Agg")  # 不弹窗
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocessing import preprocess_data, create_features
from src.clustering import perform_clustering_auto, analyze_clusters
from src.labeling import (
    create_complete_labeling_system,
    print_cluster_characteristics,
    export_cluster_reports,
    analyze_product_preference_by_cluster,
)
from src.association import analyze_product_associations_simple

# 中文字体：按系统常见字体依次尝试
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC",
                                   "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# main 只负责跑分析，不出图（图表统一由 src/generate_charts.py 生成）。
# 这里屏蔽 plt.show()，避免运行时弹窗卡住。
plt.show = lambda *a, **k: plt.close("all")


def main():
    """主执行函数"""
    print("开始用户分层与产品偏好分析")
    print("=" * 80)

    try:
        # 1. 加载数据
        df = load_data()

        # 2. 数据预处理
        df_processed = preprocess_data(df)

        # 3. 特征工程
        print("\n3. 特征工程...")
        user_features = create_features(df_processed, sample_size=None)

        # 4. 用户分层聚类（肘部法则自动确定 K）
        user_with_clusters, kmeans_model, optimal_k = perform_clustering_auto(
            user_features
        )

        # 5. 聚类结果分析
        analyze_clusters(user_with_clusters, df_processed)

        # 6. 用户标签系统
        labeled_users = create_complete_labeling_system(
            user_with_clusters, df_processed
        )

        # 7. 产品关联分析（支持度 / 置信度 / 提升度）
        association_results = analyze_product_associations_simple(
            df_processed, labeled_users
        )

        # 8. 分聚类产品偏好
        analyze_product_preference_by_cluster(df_processed, labeled_users)

        # 9. 各类别用户特征详情
        print_cluster_characteristics(labeled_users, df_processed, num_clusters=optimal_k)

        # 10. 导出聚类特征报告 CSV
        export_cluster_reports(labeled_users, df_processed, num_clusters=optimal_k)

        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        return labeled_users, association_results

    except Exception as e:
        print(f"\n分析过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

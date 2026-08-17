# K-Means 聚类，肘部法则选 K，PCA 画图
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def perform_clustering_auto(user_features, max_clusters=15):
    """
    使用肘部法则自动确定最佳聚类数量
    
    参数:
    user_features: 用户特征数据框
    max_clusters: 最大聚类数
    
    返回:
    user_with_clusters: 带有聚类标签的用户数据
    kmeans_model: 最终的KMeans模型
    optimal_k: 最佳聚类数
    """
    print("\n4. 自动确定最佳聚类数...")
    
    # 选择用于聚类的特征（使用标准化后的特征）
    feature_cols = [col for col in user_features.columns if col.endswith('_scaled')]
    
    if not feature_cols:
        print("  没有找到标准化特征，将使用原始数值特征")
        # 如果没有标准化特征，使用原始数值特征并标准化
        numeric_cols = user_features.select_dtypes(include=[np.number]).columns.tolist()
        # 排除ID列和非特征列
        exclude_cols = ['user_id']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(user_features[feature_cols])
    else:
        X_scaled = user_features[feature_cols].values
    
    # 肘部法则：计算不同K值的惯性
    inertias = []
    K_range = range(1, min(max_clusters + 1, len(X_scaled) + 1))
    
    for k in K_range:
        print(f"  测试 K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # 寻找最佳K值（肘部点） - 使用简单的肘部法则替代kneed
    if len(inertias) >= 3:
        # 计算曲率变化（二阶差分）
        diff1 = np.diff(inertias)  # 一阶差分
        diff2 = np.diff(diff1)     # 二阶差分
        
        # 找到惯性下降明显变缓的点
        elbow_point = None
        for i in range(1, len(diff2)):
            # 如果下降幅度显著变小
            if abs(diff2[i]) < abs(diff2[i-1]) * 0.5 and i >= 2:
                elbow_point = i + 1  # 加1因为diff2比原数组短2
                break
        
        if elbow_point is None:
            # 如果没有明显肘部点，使用启发式方法
            optimal_k = min(9, max(3, len(X_scaled) // 100))
        else:
            optimal_k = elbow_point
    else:
        optimal_k = min(9, max(3, len(X_scaled) // 100))
    
    print(f"  最佳聚类数: {optimal_k}")
    
    # 使用最佳K值进行最终聚类
    print(f"  使用 K={optimal_k} 进行聚类...")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    user_features['cluster'] = final_kmeans.fit_predict(X_scaled)
    
    # 可视化肘部法则
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7, label=f'最佳K值: {optimal_k}')
    plt.xlabel('聚类数量 (K)', fontsize=12)
    plt.ylabel('惯性 (Inertia)', fontsize=12)
    plt.title('肘部法则: 确定最佳聚类数量', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 可视化聚类结果（使用PCA降维）
    from sklearn.decomposition import PCA
    
    if X_scaled.shape[1] > 2:
        print("  使用PCA可视化聚类结果...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=user_features['cluster'], 
                             cmap='tab20', s=50, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # 添加聚类中心
        cluster_centers_pca = pca.transform(final_kmeans.cluster_centers_)
        plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
                   c='red', marker='X', s=200, alpha=0.8, label='聚类中心')
        
        plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        plt.title(f'用户聚类可视化 (K={optimal_k})', fontsize=14)
        plt.colorbar(scatter, label='聚类标签')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    print(f"  聚类完成: {optimal_k} 个聚类")
    print(f"  聚类分布:\n{user_features['cluster'].value_counts().sort_index()}")
    
    return user_features, final_kmeans, optimal_k

# 还需要添加 analyze_clusters 函数


def analyze_clusters(user_with_clusters, original_df):
    """
    分析聚类结果
    
    参数:
    user_with_clusters: 带有聚类标签的用户特征数据
    original_df: 原始数据
    
    返回:
    cluster_summary: 聚类摘要
    """
    print("\n5. 聚类结果分析...")
    
    # 1. 聚类大小分布
    cluster_distribution = user_with_clusters['cluster'].value_counts().sort_index()
    total_users = len(user_with_clusters)
    
    print(f"聚类分布:")
    for cluster_id, count in cluster_distribution.items():
        percentage = count / total_users * 100
        print(f"  聚类 {cluster_id}: {count} 用户 ({percentage:.1f}%)")
    
    # 2. 计算每个聚类的平均特征
    print("\n聚类特征对比:")
    
    # 选择要分析的特征
    feature_cols = [
        'total_uses', 'product_count', 'days_active', 'equity_ratio',
        'recency', 'avg_salary', 'unique_job_functions', 'activity_score'
    ]
    
    # 只保留实际存在的特征
    existing_features = [col for col in feature_cols if col in user_with_clusters.columns]
    
    if existing_features:
        # 计算每个聚类的特征平均值
        cluster_means = user_with_clusters.groupby('cluster')[existing_features].mean().round(2)
        
        # 计算每个特征在所有聚类中的标准差，用于衡量区分度
        feature_variability = cluster_means.std()
        
        print("\n各聚类特征平均值:")
        print(cluster_means)
        
        print("\n特征区分度（标准差越大，区分度越高）:")
        for feature in existing_features:
            std_value = feature_variability[feature]
            if std_value > cluster_means[feature].mean() * 0.3:  # 阈值
                print(f"  {feature}: {std_value:.2f} (高区分度)")
            else:
                print(f"  {feature}: {std_value:.2f}")
    
    # 3. 可视化聚类特征对比
    if existing_features:
        # 选择前4个特征进行可视化
        vis_features = existing_features[:4]
        if len(vis_features) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.ravel()
            
            for idx, feature in enumerate(vis_features):
                if idx < len(axes):
                    # 按聚类分组计算平均值
                    feature_by_cluster = user_with_clusters.groupby('cluster')[feature].mean()
                    
                    axes[idx].bar(feature_by_cluster.index, feature_by_cluster.values, 
                                 color=plt.cm.Set3(np.arange(len(feature_by_cluster))/len(feature_by_cluster)))
                    axes[idx].set_xlabel('聚类')
                    axes[idx].set_ylabel(feature)
                    axes[idx].set_title(f'{feature} 按聚类分布')
                    axes[idx].grid(True, alpha=0.3)
            
            plt.suptitle('聚类特征对比', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show()
    
    # 4. 创建聚类摘要
    cluster_summary = {}
    for cluster_id in sorted(user_with_clusters['cluster'].unique()):
        cluster_data = user_with_clusters[user_with_clusters['cluster'] == cluster_id]
        cluster_summary[cluster_id] = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / total_users * 100,
            'avg_total_uses': cluster_data['total_uses'].mean() if 'total_uses' in cluster_data.columns else 0,
            'avg_product_count': cluster_data['product_count'].mean() if 'product_count' in cluster_data.columns else 0,
            'avg_equity_ratio': cluster_data['equity_ratio'].mean() if 'equity_ratio' in cluster_data.columns else 0,
            'avg_activity_score': cluster_data['activity_score'].mean() if 'activity_score' in cluster_data.columns else 0,
        }
    
    return cluster_summary

# 1. 首先需要定义 analyze_product_associations_simple 函数

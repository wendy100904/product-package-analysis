# -*- coding: utf-8 -*-
"""数据预处理与特征工程模块（脱敏版）。"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """数据预处理函数"""
    print("\n2. 数据预处理...")
    
    # 复制数据避免修改原数据
    data = df.copy()
    
    # 2.1 数据类型转换
    # 日期转换
    date_columns = ['event_date', 'stat_date', 'publish_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # 2.2 处理缺失值
    print("处理缺失值...")
    # 对于分类变量，用众数填充
    categorical_cols = ['city_tier_level', 'org_type_name', 'company_scale_type', 
                       'final_segmentation_level', 'job_sal', 'job_edu_level_code',
                       'product_type', 'is_equity']
    for col in categorical_cols:
        if col in data.columns and data[col].isnull().any():
            mode_val = data[col].mode()[0] if not data[col].mode().empty else '未知'
            data[col] = data[col].fillna(mode_val)
            print(f"  {col}: 填充了 {data[col].isnull().sum()} 个缺失值")
    
    # 2.3 创建新的特征
    print("创建新特征...")
    # 最近使用时间
    if 'event_date' in data.columns:
        latest_date = data['event_date'].max()
        data['days_since_last_use'] = (latest_date - data['event_date']).dt.days
    
    # 用户活跃度标记 - 整合多个活跃指标
    # 判断是否活跃：有搜索、IM、查看简历、或7日内有新发职位的用户
    
    # 首先创建各个活跃指标
    if 'active_user_id' in data.columns:
        data['is_active_user'] = data['active_user_id'].apply(lambda x: 1 if pd.notnull(x) else 0)
    
    if 'job_post_user_id' in data.columns:
        data['is_new_job_user'] = data['job_post_user_id'].apply(lambda x: 1 if pd.notnull(x) else 0)
    
    if 'search_user_id' in data.columns:
        data['is_searcher'] = data['search_user_id'].apply(lambda x: 1 if pd.notnull(x) else 0)
    
    if 'chat_user_id' in data.columns:
        data['is_communicator'] = data['chat_user_id'].apply(lambda x: 1 if pd.notnull(x) else 0)
    
    if 'resume_view_user_id' in data.columns:
        data['is_viewer'] = data['resume_view_user_id'].apply(lambda x: 1 if pd.notnull(x) else 0)
    
    # 综合活跃度指标：如果满足任一活跃条件即为活跃用户（包括新发职位用户）
    active_conditions = []
    if 'is_active_user' in data.columns:
        active_conditions.append('is_active_user')
    if 'is_new_job_user' in data.columns:
        active_conditions.append('is_new_job_user')  # 新发职位用户也算活跃用户
    if 'is_searcher' in data.columns:
        active_conditions.append('is_searcher')
    if 'is_communicator' in data.columns:
        active_conditions.append('is_communicator')
    if 'is_viewer' in data.columns:
        active_conditions.append('is_viewer')
    
    if active_conditions:
        data['is_active_combined'] = data[active_conditions].max(axis=1)
    else:
        data['is_active_combined'] = 0
    
    # 2.4 薪资处理 - 根据说明，job_salary字段是具体的薪资数值，单位为万
    # 例如：35.6代表35.6万
    # 按照 0, 8, 15, 25, 40, 50 进行分段
    # 重要：需要按用户计算平均薪资，避免重复计算
    if 'job_salary' in data.columns:
        print("处理薪资数据...")
        
        def map_salary(salary):
            """将薪资转换为数值型（单位为万）"""
            if pd.isna(salary):
                return np.nan
            
            try:
                # 如果是字符串，清理并转换
                if isinstance(salary, str):
                    # 移除空格、逗号等
                    salary_clean = salary.replace(',', '').replace('，', '').strip()
                    
                    # 检查是否有"万"字，有则移除
                    if '万' in salary_clean:
                        salary_clean = salary_clean.replace('万', '')
                    # 检查是否有"k"或"K"，有则转换为万（假设1k=0.1万）
                    elif 'k' in salary_clean.lower():
                        salary_clean = str(float(salary_clean.lower().replace('k', '')) / 10)
                    # 检查是否有"千"字，有则转换为万
                    elif '千' in salary_clean:
                        salary_clean = str(float(salary_clean.replace('千', '')) / 10)
                    
                    return float(salary_clean)
                else:
                    # 已经是数值型，直接返回
                    return float(salary)
            except Exception as e:
                # 如果转换失败，返回NaN
                return np.nan
        
        # 先转换薪资数据
        data['salary_num'] = data['job_salary'].apply(map_salary)
        
        # 按用户计算平均薪资（避免重复计算）
        print("\n  按用户计算薪资统计...")
        
        # 获取每个用户的平均薪资
        user_salary_stats = data[['user_id', 'salary_num']].copy()
        
        # 筛选有效的薪资数据
        valid_salary_data = user_salary_stats[user_salary_stats['salary_num'].notna()]
        
        if len(valid_salary_data) > 0:
            # 按用户分组计算平均薪资
            user_avg_salary = valid_salary_data.groupby('user_id')['salary_num'].agg(['mean', 'count']).reset_index()
            user_avg_salary = user_avg_salary.rename(columns={'mean': 'avg_salary_num', 'count': 'salary_records'})
            
            print(f"  有薪资数据的用户数: {len(user_avg_salary)}")
            print(f"  总用户数: {data['user_id'].nunique()}")
            print(f"  有薪资数据的用户比例: {len(user_avg_salary)/data['user_id'].nunique()*100:.1f}%")
            
            # 计算用户平均薪资的统计信息
            min_salary = user_avg_salary['avg_salary_num'].min()
            max_salary = user_avg_salary['avg_salary_num'].max()
            avg_salary = user_avg_salary['avg_salary_num'].mean()
            median_salary = user_avg_salary['avg_salary_num'].median()
            
            print(f"  用户平均薪资范围: {min_salary:.1f}万 ~ {max_salary:.1f}万")
            print(f"  用户平均薪资均值: {avg_salary:.1f}万")
            print(f"  用户平均薪资中位数: {median_salary:.1f}万")
            
            # 按照指定的分段点创建薪资等级
            # 分段点: 0, 8, 15, 25, 40, 50
            bins = [0, 8, 15, 25, 40, 50, float('inf')]
            labels = [
                '低薪(<8万)', 
                '中低薪(8-15万)', 
                '中等薪(15-25万)', 
                '中高薪(25-40万)', 
                '高薪(40-50万)', 
                '超高薪(>50万)'
            ]
            
            # 为用户平均薪资创建分箱
            user_avg_salary['salary_level'] = pd.cut(
                user_avg_salary['avg_salary_num'], 
                bins=bins, 
                labels=labels, 
                right=False,  # 左闭右开 [a, b)
                include_lowest=True  # 包含最小值
            )
            
            # 统计各薪资等级分布（按用户）
            print("\n  用户薪资等级分布（按0,8,15,25,40,50分段）:")
            salary_dist = user_avg_salary['salary_level'].value_counts().sort_index()
            
            # 按分段顺序显示
            for label in labels:
                if label in salary_dist.index:
                    user_count = salary_dist[label]
                    percentage = user_count / len(user_avg_salary) * 100
                    # 提取薪资范围用于显示
                    if '<' in label:
                        range_str = label.split('(')[1].replace(')', '')
                    elif '>' in label:
                        range_str = label.split('(')[1].replace(')', '')
                    else:
                        range_str = label.split('(')[1].replace(')', '')
                    
                    print(f"    {range_str}: {user_count}人 ({percentage:.1f}%)")
                else:
                    range_str = label.split('(')[1].replace(')', '')
                    print(f"    {range_str}: 0人 (0.0%)")
            
            # 将用户平均薪资和等级合并回原始数据
            # 首先创建一个映射字典
            salary_mapping = user_avg_salary.set_index('user_id')[['avg_salary_num', 'salary_level']].to_dict(orient='index')
            
            # 为原始数据添加平均薪资和等级
            def get_user_salary_info(user_id):
                if user_id in salary_mapping:
                    return (
                        salary_mapping[user_id]['avg_salary_num'],
                        salary_mapping[user_id]['salary_level']
                    )
                return (np.nan, '未知')
            
            # 应用映射
            data[['user_avg_salary', 'salary_level']] = data['user_id'].apply(
                lambda x: pd.Series(get_user_salary_info(x))
            )
            
            # 显示各分段的详细统计（按用户）
            print("\n  各薪资分段详细统计（按用户）:")
            for i in range(len(bins)-1):
                lower = bins[i]
                upper = bins[i+1] if bins[i+1] != float('inf') else "∞"
                label = labels[i]
                
                # 获取该分段的用户数据
                if upper == "∞":
                    segment_users = user_avg_salary[(user_avg_salary['avg_salary_num'] >= lower)]
                else:
                    segment_users = user_avg_salary[(user_avg_salary['avg_salary_num'] >= lower) & (user_avg_salary['avg_salary_num'] < upper)]
                
                if len(segment_users) > 0:
                    seg_min = segment_users['avg_salary_num'].min()
                    seg_max = segment_users['avg_salary_num'].max()
                    seg_avg = segment_users['avg_salary_num'].mean()
                    seg_median = segment_users['avg_salary_num'].median()
                    user_count = len(segment_users)
                    percentage = user_count / len(user_avg_salary) * 100
                    
                    print(f"    {label}:")
                    print(f"      用户数: {user_count} ({percentage:.1f}%)")
                    print(f"      薪资范围: {seg_min:.1f}~{seg_max:.1f}万")
                    print(f"      平均薪资: {seg_avg:.1f}万")
                    print(f"      中位数: {seg_median:.1f}万")
                    
                    # 显示平均薪资记录数
                    avg_records = segment_users['salary_records'].mean()
                    print(f"      平均薪资记录数: {avg_records:.1f}")
        else:
            print("  警告: 没有有效的薪资数据")
            data['user_avg_salary'] = np.nan
            data['salary_level'] = '未知'
    else:
        print("  警告: 数据中没有job_salary字段")
        data['user_avg_salary'] = np.nan
        data['salary_level'] = '未知'
    
    # 2.5 处理其他数值字段
    # 如果有其他需要处理的数值字段，可以在这里添加
    
    print(f"\n预处理后数据形状: {data.shape}")
    print(f"总用户数: {data['user_id'].nunique()}")
    
    # 显示预处理后的数据示例
    print("\n预处理后的数据示例（前3行）:")
    display_columns = ['user_id', 'product_type', 'is_equity', 'salary_num', 'user_avg_salary', 'salary_level']
    available_columns = [col for col in display_columns if col in data.columns]
    
    if available_columns:
        print(data[available_columns].head(3))
    else:
        print(data.head(3))
    
    # 检查薪资数据完整性
    if 'user_avg_salary' in data.columns:
        valid_salary_users = data['user_avg_salary'].notna().sum()
        total_users = data['user_id'].nunique()
        print(f"\n薪资数据完整性检查:")
        print(f"  有平均薪资数据的用户数: {valid_salary_users}")
        print(f"  总用户数: {total_users}")
        print(f"  覆盖率: {valid_salary_users/total_users*100:.1f}%")
    
    return data

# 添加 create_features 函数


def create_features(df, sample_size=None):
    """
    从原始数据中创建用户特征
    
    参数:
    df: 预处理后的数据框
    sample_size: 采样大小（None表示不采样）
    
    返回:
    user_features: 用户特征数据框
    """
    print("创建用户特征...")
    
    # 如果需要采样
    if sample_size is not None and sample_size < len(df):
        print(f"  对数据进行采样: {sample_size} 条记录")
        df = df.sample(sample_size, random_state=42)
    
    # 确保有用户ID列
    if 'user_id' not in df.columns:
        raise ValueError("数据中必须包含 'user_id' 列")
    
    # 1. 基础使用特征
    print("  提取基础使用特征...")
    user_features = df.groupby('user_id').agg({
        # 使用频率特征
        'user_id': 'count',  # 总使用次数
        'product_type': 'nunique',  # 使用的产品种类数
        'event_date': ['min', 'max'],  # 首次和最后使用日期
        # 活跃度特征
        'is_active_combined': 'max',  # 是否活跃（整合后的）
        'is_new_job_user': 'max',  # 是否新发职位用户
        'is_searcher': 'max',  # 是否搜索用户
        'is_communicator': 'max',  # 是否沟通用户
        'is_viewer': 'max',  # 是否查看简历用户
        # 产品类型特征
        'is_equity': lambda x: (x == '权益').mean(),  # 权益产品使用比例
    }).reset_index()
    
    # 重命名列
    user_features.columns = [
        'user_id',
        'total_uses',
        'product_count',
        'first_use_date',
        'last_use_date',
        'is_active',
        'is_new_job_user',
        'is_searcher',
        'is_communicator',
        'is_viewer',
        'equity_ratio'
    ]
    
    # 2. 计算活跃天数
    print("  计算活跃天数...")
    # 获取每个用户有活动的日期数量
    user_dates = df.groupby('user_id')['event_date'].apply(lambda x: x.dt.date.nunique()).reset_index()
    user_dates.columns = ['user_id', 'days_active']
    user_features = user_features.merge(user_dates, on='user_id', how='left')
    
    # 3. 计算RFM特征
    print("  计算RFM特征...")
    # R(最近使用时间): 距离分析日期的天数
    analysis_date = df['event_date'].max()
    user_features['recency'] = (analysis_date - user_features['last_use_date']).dt.days
    user_features['recency'] = user_features['recency'].fillna(365)  # 如果缺失，设为365天
    
    # F(使用频率): 使用次数
    user_features['frequency'] = user_features['total_uses']
    
    # M(使用价值): 这里用使用的产品种类数和权益产品比例作为价值指标
    user_features['monetary'] = user_features['product_count'] * (1 + user_features['equity_ratio'])
    
    # 4. 提取客户属性特征
    print("  提取客户属性特征...")
    # 获取每个用户的最新客户属性
    if 'org_type_name' in df.columns:
        # 对于每个用户，取出现次数最多的组织类型
        org_types = df.groupby('user_id')['org_type_name'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else '未知').reset_index()
        org_types.columns = ['user_id', 'org_type']
        user_features = user_features.merge(org_types, on='user_id', how='left')
    
    # 城市级别
    if 'city_tier_level' in df.columns:
        city_levels = df.groupby('user_id')['city_tier_level'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else '未知').reset_index()
        city_levels.columns = ['user_id', 'city_level']
        user_features = user_features.merge(city_levels, on='user_id', how='left')
    
    # 行业信息（如果有）
    if 'jobtitle1' in df.columns:
        # 取用户最常发布的职位领域作为行业标签
        industries = df.groupby('user_id')['jobtitle1'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else '未知'
        ).reset_index()
        industries.columns = ['user_id', 'industry']
        user_features = user_features.merge(industries, on='user_id', how='left')
    
    # 企业规模
    if 'company_scale_type' in df.columns:
        company_sizes = df.groupby('user_id')['company_scale_type'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else '未知'
        ).reset_index()
        company_sizes.columns = ['user_id', 'company_size']
        user_features = user_features.merge(company_sizes, on='user_id', how='left')
    
    # 5. 提取职位特征
    print("  提取职位特征...")
    # 薪资信息
    if 'salary_num' in df.columns:
        salary_stats = df.groupby('user_id')['salary_num'].agg(['mean', 'max', 'min']).reset_index()
        salary_stats.columns = ['user_id', 'avg_salary', 'max_salary', 'min_salary']
        user_features = user_features.merge(salary_stats, on='user_id', how='left')
        user_features['avg_salary'] = user_features['avg_salary'].fillna(0)
        user_features['max_salary'] = user_features['max_salary'].fillna(0)
        user_features['min_salary'] = user_features['min_salary'].fillna(0)
    
    # 职能多样性
    if 'jobtitle1' in df.columns:
        job_functions = df.groupby('user_id')['jobtitle1'].nunique().reset_index()
        job_functions.columns = ['user_id', 'unique_job_functions']
        user_features = user_features.merge(job_functions, on='user_id', how='left')
    
    # 6. 计算衍生特征
    print("  计算衍生特征...")
    # 使用强度（每天平均使用次数）
    user_features['usage_intensity'] = user_features['total_uses'] / user_features['days_active'].clip(lower=1)
    
    # 产品多样性指数
    user_features['product_diversity'] = user_features['product_count'] / 10  # 假设最多10种产品
    
    # 综合活跃度评分
    user_features['activity_score'] = (
        user_features['is_active'] + 
        user_features['is_searcher'] + 
        user_features['is_communicator'] + 
        user_features['is_viewer'] +
        user_features['is_new_job_user']
    ) / 5.0
    
    # 7. 处理缺失值
    print("  处理特征缺失值...")
    # 填充分类变量的缺失值
    categorical_cols = ['org_type', 'city_level', 'industry', 'company_size']
    for col in categorical_cols:
        if col in user_features.columns:
            user_features[col] = user_features[col].fillna('未知')
    
    # 填充数值变量的缺失值
    numerical_cols = ['days_active', 'unique_job_functions', 'avg_salary', 
                      'max_salary', 'min_salary', 'usage_intensity',
                      'product_diversity', 'activity_score']
    for col in numerical_cols:
        if col in user_features.columns:
            user_features[col] = user_features[col].fillna(0)
    
    # 8. 特征标准化（为聚类准备）
    print("  特征标准化...")
    # 选择要标准化的数值特征
    numeric_features_for_clustering = [
        'total_uses', 'product_count', 'days_active', 'equity_ratio',
        'recency', 'frequency', 'monetary', 'avg_salary',
        'unique_job_functions', 'usage_intensity', 'product_diversity',
        'activity_score'
    ]
    
    # 只保留实际存在的特征
    existing_features = [col for col in numeric_features_for_clustering if col in user_features.columns]
    
    # 复制原始数据
    user_features_raw = user_features.copy()
    
    # 标准化特征（为聚类分析准备）
    from sklearn.preprocessing import StandardScaler
    
    if existing_features:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(user_features[existing_features])
        
        # 将标准化后的特征添加到数据框
        for i, col in enumerate(existing_features):
            user_features[f'{col}_scaled'] = scaled_features[:, i]
    
    print(f"  特征创建完成: {len(user_features)} 个用户，{len(user_features.columns)} 个特征")
    print(f"  特征列: {list(user_features.columns)}")
    
    return user_features

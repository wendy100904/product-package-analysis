# 用户打标签 + 聚类画像输出
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_complete_labeling_system(user_df, original_df):
    """创建完整的用户标签系统（包含行业标签）"""
    print("\n创建用户标签系统...")
    
    labeled_users = user_df.copy()
    
    # 1. 基础聚类标签 - 根据你的9个聚类结果重新定义
    cluster_labels_map = {
        0: {'name': '低频付费型', 'description': '使用频率低，偏好付费产品，单一产品使用'},
        1: {'name': '中频权益型', 'description': '中频使用，偏好权益产品，常发布新职位'},
        2: {'name': '高频权益活跃型', 'description': '高活跃、高频使用，偏好权益产品'},
        3: {'name': '中高薪权益型', 'description': '中高薪资职位，偏好权益产品'},
        4: {'name': '低频付费单一型', 'description': '低频使用，单一付费产品'},
        5: {'name': '低活跃权益型', 'description': '低活跃度，偏好权益产品'},
        6: {'name': '高频新职位活跃型', 'description': '高频使用，常发布新职位，高活跃'},
        7: {'name': '超级活跃权益型', 'description': '超级活跃，高频使用，偏好权益产品'},
        8: {'name': '极低频付费型', 'description': '极低频率使用，单一付费产品'}
    }
    
    # 确保所有聚类都有对应的标签
    for cluster_id in sorted(labeled_users['cluster'].unique()):
        if cluster_id not in cluster_labels_map:
            # 为没有定义的聚类添加默认标签
            cluster_labels_map[cluster_id] = {
                'name': f'聚类_{cluster_id}',
                'description': f'第{cluster_id}类用户群体'
            }
    
    labeled_users['cluster_name'] = labeled_users['cluster'].map(
        lambda x: cluster_labels_map.get(x, {}).get('name', f'聚类_{x}')
    )
    labeled_users['cluster_desc'] = labeled_users['cluster'].map(
        lambda x: cluster_labels_map.get(x, {}).get('description', '')
    )
    
    # 2. 基于底层字段的业务标签（添加行业标签）
    def generate_business_labels(row):
        labels = []
        
        # 客户属性标签
        if row.get('org_type') == 'KA':
            labels.append('KA客户')
        elif row.get('org_type') == 'SME':
            labels.append('SME客户')
        
        # 城市级别标签
        if row.get('city_level') in ['一线城市', '新一线城市']:
            labels.append('高线城市')
        elif row.get('city_level') in ['二线城市', '三线城市']:
            labels.append('中线城市')
        elif row.get('city_level') in ['四线城市', '五线城市']:
            labels.append('低线城市')    
        
        # 行业标签
        industry = row.get('industry', '')
        if industry and industry != '未知' and industry != '其他行业' and industry != '其他':
            # 简化行业名称
            if len(industry) > 8:
                industry_short = industry[:6] + '...'
            else:
                industry_short = industry
            labels.append(f'{industry_short}行业')
        
        # 客户等级标签
        customer_level = row.get('customer_level', '')
        if customer_level and customer_level in ['A', 'B', 'C', 'D']:
            labels.append(f'{customer_level}级客户')
        
        # 企业规模标签
        company_size = str(row.get('company_size', ''))
        if row.get('company_size') in ['1-49人', '50-99人']: 
            labels.append('小型企业')
        elif row.get('company_size') in ['100-499人', '500-999人', '1000-2000人', '2000-5000人', '5000-10000人']:
            labels.append('中型企业')
        elif row.get('company_size') in ['10000人以上']: 
            labels.append('大型企业')
        
        # 行为活跃度标签  
        if row.get('is_active', 0) == 1 and row.get('days_active', 0) >= 14:
            labels.append('高活跃用户')
        elif row.get('is_active', 0) == 1 and row.get('days_active', 0) < 7:
            labels.append('低活跃用户')
        elif row.get('is_active', 0) == 1 and row.get('days_active', 0) >= 7 and row.get('days_active', 0) < 14:
            labels.append('中活跃用户')
        
        # 使用频率标签
        if row.get('total_uses', 0) > 100:
            labels.append('高频用户')
        elif row.get('total_uses', 0) > 10 and row.get('total_uses', 0) <= 100:
            labels.append('中频用户')
        elif row.get('total_uses', 0) <= 10:
            labels.append('低频用户')
        
        # 新发职位用户标签（已在活跃用户中体现，单独标记）
        if row.get('is_new_job_user', 0) == 1:
            labels.append('新发职位用户')
        
        # 产品使用标签
        if row.get('product_count', 0) >= 4:
            labels.append('多产品用户')
        elif row.get('product_count', 0) == 1:
            labels.append('单一产品用户')
        
        if row.get('equity_ratio', 0) > 0.7:
            labels.append('权益产品偏好')
        elif row.get('equity_ratio', 0) < 0.3:
            labels.append('付费产品偏好')
        
        # 薪资标签
        if row.get('avg_salary', 0) >= 40:
            labels.append('高薪职位用户')
        elif row.get('avg_salary', 0) >= 15:
            labels.append('中高薪职位用户')
        elif row.get('avg_salary', 0) > 0:
            labels.append('普通薪资职位用户')
        
        # RFM价值标签
        if row.get('recency', 365) <= 7:
            labels.append('近期活跃')
        elif row.get('recency', 365) > 30:
            labels.append('长期未活跃')
        
        if row.get('frequency', 0) > 20:
            labels.append('高频使用')
        elif row.get('frequency', 0) < 5:
            labels.append('低频使用')
        
        return ' | '.join(labels) if labels else '无标签'
    
    labeled_users['business_labels'] = labeled_users.apply(generate_business_labels, axis=1)
    
    # 3. 产品偏好标签（基于原始数据）
    def get_product_preference_labels(user_id):
        user_products = original_df[original_df['user_id'] == user_id]
        if len(user_products) == 0:
            return '无产品数据'
        
        product_counts = user_products['product_type'].value_counts()
        top_products = product_counts.head(3).index.tolist()
        
        equity_products = user_products[user_products['is_equity'] == '权益']
        equity_ratio = len(equity_products) / len(user_products)
        
        labels = []
        if top_products:
            labels.append(f"常用产品:{','.join(top_products[:2])}")
        
        if equity_ratio > 0.7:
            labels.append('重度权益用户')
        elif equity_ratio < 0.3:
            labels.append('轻度权益用户')
        
        return ' | '.join(labels)
    
    labeled_users['product_pref_labels'] = labeled_users['user_id'].apply(get_product_preference_labels)
    
    # 4. 职位特征标签（基于原始数据）
    def get_job_feature_labels(user_id):
        user_jobs = original_df[original_df['user_id'] == user_id]
        if len(user_jobs) == 0 or 'jobtitle1' not in user_jobs.columns:
            return '无职位信息'
        
        labels = []
        
        if not user_jobs['jobtitle1'].mode().empty:
            top_function = user_jobs['jobtitle1'].mode()[0]
            # 简化职能名称
            if len(top_function) > 8:
                func_short = top_function[:6] + '...'
            else:
                func_short = top_function
            labels.append(f"{func_short}领域")
        
        if 'salary_num' in user_jobs.columns:
            avg_salary = user_jobs['salary_num'].mean()
            if avg_salary >= 40:
                labels.append('高薪职位')
            elif avg_salary >= 15:
                labels.append('中高薪职位')
            elif avg_salary > 0:
                labels.append('普通薪资职位')
        
        return ' | '.join(labels) if labels else '职位特征未知'
    
    labeled_users['job_feature_labels'] = labeled_users['user_id'].apply(get_job_feature_labels)
    
    # 5. 合并完整标签
    labeled_users['full_label'] = (
        labeled_users['cluster_name'] + ' || ' +
        labeled_users['business_labels'] + ' || ' +
        '产品:' + labeled_users['product_pref_labels'] + ' || ' +
        '职位:' + labeled_users['job_feature_labels']
    )
    
    # 6. 为每个聚类生成典型标签摘要
    labeled_users = generate_cluster_label_summary(labeled_users)
    
    print(f"标签系统创建完成，共 {len(labeled_users)} 个用户")
    return labeled_users

# 4. 修改 generate_cluster_label_summary 函数以处理9个聚类


def generate_cluster_label_summary(labeled_users):
    """为每个聚类生成典型标签摘要"""
    print("\n为每个聚类生成典型标签摘要...")
    
    # 根据标签频率为每个聚类生成摘要
    for cluster_id in sorted(labeled_users['cluster'].unique()):
        cluster_data = labeled_users[labeled_users['cluster'] == cluster_id]
        cluster_name = cluster_data['cluster_name'].iloc[0]
        
        # 收集该聚类最典型的标签
        all_labels = []
        for labels in cluster_data['business_labels']:
            if labels != '无标签':
                all_labels.extend(labels.split(' | '))
        
        # 计算标签频率
        from collections import Counter
        label_counter = Counter(all_labels)
        
        # 取前5个最典型的标签
        top_labels = [label for label, _ in label_counter.most_common(5)]
        
        # 创建聚类标签摘要
        label_summary = ' | '.join(top_labels)
        
        # 添加到DataFrame
        labeled_users.loc[labeled_users['cluster'] == cluster_id, 'cluster_label_summary'] = label_summary
        
        # 打印每个聚类的标签摘要
        cluster_size = len(cluster_data)
        cluster_percentage = cluster_size / len(labeled_users) * 100
        print(f"聚类 {cluster_id} ({cluster_name}): {label_summary}")
        print(f"  用户数: {cluster_size} ({cluster_percentage:.1f}%)")
    
    return labeled_users

# 5. 修改 print_cluster_characteristics 函数以处理9个聚类


def print_cluster_characteristics(labeled_users, original_df, num_clusters=None):
    """详细输出每个聚类的特征（包含行业分布）"""
    
    # 如果未指定聚类数量，使用数据中的实际聚类数量
    if num_clusters is None:
        num_clusters = len(labeled_users['cluster'].unique())
    
    print("\n" + "="*100)
    print(f"{num_clusters}个类别用户特征详细分析")
    print("="*100)
    
    for cluster_id in sorted(labeled_users['cluster'].unique())[:num_clusters]:
        cluster_users = labeled_users[labeled_users['cluster'] == cluster_id]
        
        if len(cluster_users) == 0:
            print(f"\n聚类 {cluster_id}: 无用户")
            continue
        
        print(f"\n{'='*60}")
        print(f"聚类 {cluster_id}: {cluster_users['cluster_name'].iloc[0]}")
        print(f"描述: {cluster_users['cluster_desc'].iloc[0]}")
        print(f"用户数量: {len(cluster_users)} ({len(cluster_users)/len(labeled_users)*100:.1f}%)")
        print(f"典型标签摘要: {cluster_users['cluster_label_summary'].iloc[0]}")
        print(f"{'='*60}")
        
        # 获取聚类内用户的原始数据
        cluster_user_ids = cluster_users['user_id'].tolist()
        cluster_original_data = original_df[original_df['user_id'].isin(cluster_user_ids)]
        
        # 1. 客户属性特征（添加行业分布）
        print("\n1. 客户属性特征:")
        print(f"   • KA客户占比: {(cluster_users['org_type'] == 'KA').mean()*100:.1f}%")
        print(f"   • SME客户占比: {(cluster_users['org_type'] == 'SME').mean()*100:.1f}%")
        
        city_dist = cluster_users['city_level'].value_counts(normalize=True).head(3)
        print(f"   • 城市分布(前3):")
        for city, perc in city_dist.items():
            print(f"     - {city}: {perc*100:.1f}%")
        
        # 行业分布
        if 'industry' in cluster_users.columns:
            industry_dist = cluster_users['industry'].value_counts(normalize=True).head(5)
            print(f"   • 行业分布(前5):")
            for industry, perc in industry_dist.items():
                print(f"     - {industry}: {perc*100:.1f}%")
        
        # 2. 行为特征
        print("\n2. 行为特征:")
        print(f"   • 平均使用产品数: {cluster_users['product_count'].mean():.1f}")
        print(f"   • 平均使用次数: {cluster_users['total_uses'].mean():.1f}")
        print(f"   • 平均活跃天数: {cluster_users['days_active'].mean():.1f}")
        print(f"   • 权益产品使用率: {cluster_users['equity_ratio'].mean()*100:.1f}%")
        print(f"   • 活跃用户占比: {cluster_users['is_active'].mean()*100:.1f}%")
        print(f"   • 新发职位用户占比: {cluster_users['is_new_job_user'].mean()*100:.1f}%")
        
        # 3. 职位特征 - 使用薪资
        print("\n3. 职位特征:")
        print(f"   • 平均薪资: {cluster_users['avg_salary'].mean():.1f}k")
        print(f"   • 平均职能数量: {cluster_users['unique_job_functions'].mean():.1f}")
        
        # 4. RFM特征
        print("\n4. 价值特征(RFM):")
        print(f"   • 平均最近使用时间: {cluster_users['recency'].mean():.1f}天前")
        print(f"   • 平均使用频率: {cluster_users['frequency'].mean():.1f}次")
        
        # 5. 产品使用特征
        print("\n5. 产品使用特征:")
        if len(cluster_original_data) > 0:
            top_products = cluster_original_data['product_type'].value_counts().head(5)
            print(f"   • 热门产品(前5):")
            for product, count in top_products.items():
                usage_rate = count / len(cluster_original_data) * 100
                print(f"     - {product}: {usage_rate:.1f}%的记录")
            
            equity_ratio = (cluster_original_data['is_equity'] == '权益').mean() * 100
            print(f"   • 权益产品使用率: {equity_ratio:.1f}%")
        
        # 6. 典型标签组合
        print("\n6. 典型标签组合(前5):")
        top_label_combos = cluster_users['business_labels'].value_counts().head(5)
        for combo, count in top_label_combos.items():
            percentage = count / len(cluster_users) * 100
            print(f"   • {combo}: {percentage:.1f}%")
        
        # 7. 产品偏好标签
        print("\n7. 产品偏好标签分布:")
        product_labels = cluster_users['product_pref_labels'].value_counts().head(5)
        for label, count in product_labels.items():
            percentage = count / len(cluster_users) * 100
            print(f"   • {label}: {percentage:.1f}%")
        
        # 8. 职位特征标签
        print("\n8. 职位特征标签分布:")
        job_labels = cluster_users['job_feature_labels'].value_counts().head(5)
        for label, count in job_labels.items():
            percentage = count / len(cluster_users) * 100
            print(f"   • {label}: {percentage:.1f}%")
        
        print(f"\n{'='*60}")

# 6. 添加 export_cluster_reports 函数


def export_cluster_reports(labeled_users, original_df, num_clusters=None):
    """生成详细的聚类报告CSV"""
    
    reports = []
    
    # 如果未指定聚类数量，使用数据中的实际聚类数量
    if num_clusters is None:
        num_clusters = len(labeled_users['cluster'].unique())
    
    for cluster_id in sorted(labeled_users['cluster'].unique())[:num_clusters]:
        cluster_users = labeled_users[labeled_users['cluster'] == cluster_id]
        
        if len(cluster_users) == 0:
            continue
        
        # 获取聚类内用户的原始数据
        cluster_user_ids = cluster_users['user_id'].tolist()
        cluster_original_data = original_df[original_df['user_id'].isin(cluster_user_ids)]
        
        # 1. 聚类基本信息
        report = {
            'cluster_id': cluster_id,
            'cluster_name': cluster_users['cluster_name'].iloc[0],
            'cluster_description': cluster_users['cluster_desc'].iloc[0],
            'cluster_label_summary': cluster_users['cluster_label_summary'].iloc[0],
            'user_count': len(cluster_users),
            'user_percentage': len(cluster_users) / len(labeled_users) * 100,
        }
        
        # 2. 客户属性统计
        report['ka_ratio'] = (cluster_users['org_type'] == 'KA').mean() * 100
        report['sme_ratio'] = (cluster_users['org_type'] == 'SME').mean() * 100
        
        # 城市分布
        if len(cluster_users['city_level'].value_counts()) > 0:
            top_city = cluster_users['city_level'].value_counts().index[0]
            report['top_city'] = top_city
        else:
            report['top_city'] = '未知'
        
        # 行业分布
        if 'industry' in cluster_users.columns and len(cluster_users['industry'].value_counts()) > 0:
            top_industry = cluster_users['industry'].value_counts().index[0]
            report['top_industry'] = top_industry
        else:
            report['top_industry'] = '未知'
        
        # 3. 行为特征统计
        report['avg_product_count'] = cluster_users['product_count'].mean()
        report['avg_total_uses'] = cluster_users['total_uses'].mean()
        report['avg_equity_ratio'] = cluster_users['equity_ratio'].mean() * 100
        report['active_user_ratio'] = cluster_users['is_active'].mean() * 100
        report['new_job_user_ratio'] = cluster_users['is_new_job_user'].mean() * 100
        
        # 4. 职位特征统计
        report['avg_salary'] = cluster_users['avg_salary'].mean()
        report['avg_job_functions'] = cluster_users['unique_job_functions'].mean()
        
        # 5. 产品使用特征
        if len(cluster_original_data) > 0 and len(cluster_original_data['product_type'].value_counts()) > 0:
            top_product = cluster_original_data['product_type'].value_counts().index[0]
            report['top_product'] = top_product
            report['equity_usage_ratio'] = (cluster_original_data['is_equity'] == '权益').mean() * 100
        else:
            report['top_product'] = '未知'
            report['equity_usage_ratio'] = 0
        
        # 6. 典型标签
        if len(cluster_users['business_labels'].value_counts()) > 0:
            top_label = cluster_users['business_labels'].value_counts().index[0]
            report['top_business_label'] = top_label
        else:
            report['top_business_label'] = '无标签'
        
        reports.append(report)
    
    # 创建DataFrame并保存
    if reports:
        report_df = pd.DataFrame(reports)
        
        # 重新排列列顺序
        columns_order = [
            'cluster_id', 'cluster_name', 'cluster_description', 'cluster_label_summary',
            'user_count', 'user_percentage',
            'ka_ratio', 'sme_ratio', 'top_city', 'top_industry',
            'avg_product_count', 'avg_total_uses', 'avg_equity_ratio', 
            'active_user_ratio', 'new_job_user_ratio',
            'avg_salary', 'avg_job_functions',
            'top_product', 'equity_usage_ratio',
            'top_business_label'
        ]
        
        # 只保留存在的列
        existing_columns = [col for col in columns_order if col in report_df.columns]
        report_df = report_df[existing_columns]
        
        # 保存报告
        report_df.to_csv('cluster_characteristics_report.csv', index=False, encoding='utf-8-sig')
        print(f"\n聚类特征报告已保存到: cluster_characteristics_report.csv")
        
        return report_df
    else:
        print("没有可报告的聚类数据")
        return pd.DataFrame()

# 还需要添加 analyze_product_preference_by_cluster 函数


def analyze_product_preference_by_cluster(df, user_df):
    """分析每个聚类的产品偏好"""
    print("\n7. 按聚类分析产品偏好...")
    
    # 检查必要的列是否存在
    required_columns = ['user_id', 'cluster']
    missing_columns = [col for col in required_columns if col not in user_df.columns]
    
    if missing_columns:
        print(f"  警告: user_df缺少必要的列: {missing_columns}")
        return []
    
    # 检查是否有cluster_name列，如果没有则创建默认的
    if 'cluster_name' not in user_df.columns:
        print("  警告: user_df中没有cluster_name列，将创建默认名称")
        # 创建默认的聚类名称
        user_df['cluster_name'] = user_df['cluster'].apply(lambda x: f'聚类_{x}')
    
    # 合并聚类信息到原始数据
    df_with_cluster = df.merge(user_df[['user_id', 'cluster', 'cluster_name']], on='user_id', how='left')
    
    # 分析每个聚类的热门产品
    cluster_product_analysis = []
    
    for cluster_id in sorted(user_df['cluster'].unique()):
        cluster_data = df_with_cluster[df_with_cluster['cluster'] == cluster_id]
        cluster_users = user_df[user_df['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            print(f"  聚类 {cluster_id}: 无数据")
            continue
            
        # 产品使用统计
        product_usage = cluster_data['product_type'].value_counts().head(10)
        
        # 检查是否至少有一个产品
        if len(product_usage) == 0:
            print(f"  聚类 {cluster_id}: 无产品使用数据")
            continue
        
        # 权益产品使用情况 - 检查is_equity列是否存在
        equity_usage = []
        non_equity_usage = []
        
        if 'is_equity' in cluster_data.columns:
            equity_data = cluster_data[cluster_data['is_equity'] == '权益']
            if len(equity_data) > 0:
                equity_usage = equity_data['product_type'].value_counts().head(5).index.tolist()
            
            non_equity_data = cluster_data[cluster_data['is_equity'] == '非权益']
            if len(non_equity_data) > 0:
                non_equity_usage = non_equity_data['product_type'].value_counts().head(5).index.tolist()
        
        # 计算产品使用率
        total_users_in_cluster = len(cluster_users)
        product_usage_rate = {}
        
        for product, count in product_usage.items():
            # 计算使用该产品的用户数
            users_with_product = cluster_data[cluster_data['product_type'] == product]['user_id'].nunique()
            usage_rate = users_with_product / total_users_in_cluster * 100
            product_usage_rate[product] = usage_rate
        
        # 获取聚类名称
        if 'cluster_name' in cluster_users.columns and len(cluster_users) > 0:
            cluster_name = cluster_users['cluster_name'].iloc[0]
        else:
            cluster_name = f'聚类_{cluster_id}'
        
        cluster_info = {
            'cluster': cluster_id,
            'cluster_name': cluster_name,
            'cluster_size': total_users_in_cluster,
            'top_products': product_usage.index.tolist(),
            'top_products_count': product_usage.values.tolist(),
            'top_products_usage_rate': [product_usage_rate.get(p, 0) for p in product_usage.index.tolist()],
            'top_equity_products': equity_usage,
            'top_non_equity_products': non_equity_usage
        }
        
        cluster_product_analysis.append(cluster_info)
    
    # 可视化每个聚类的产品偏好
    num_clusters = len(cluster_product_analysis)
    if num_clusters > 0:
        cols = min(3, num_clusters)
        rows = (num_clusters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        
        # 如果只有一行，确保axes是数组
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, cluster_info in enumerate(cluster_product_analysis):
            if idx >= rows * cols:
                break
                
            row = idx // cols
            col = idx % cols
            
            if row < rows and col < cols:
                ax = axes[row, col]
                
                # 准备数据（取前5个产品）
                products = cluster_info['top_products'][:5]
                usage_rates = cluster_info['top_products_usage_rate'][:5]
                
                if len(products) > 0 and len(usage_rates) > 0:
                    # 绘制条形图
                    bars = ax.barh(range(len(products)), usage_rates, 
                                 color=plt.cm.Set3(np.arange(len(products))/len(products)))
                    ax.set_yticks(range(len(products)))
                    ax.set_yticklabels(products, fontsize=9)
                    ax.invert_yaxis()
                    ax.set_xlabel('用户使用率 (%)', fontsize=10)
                    
                    # 设置标题（可能包含换行）
                    title = f'{cluster_info["cluster_name"]}\n({cluster_info["cluster_size"]}用户)'
                    if len(title) > 30:  # 如果标题太长，减小字体
                        ax.set_title(title, fontsize=9)
                    else:
                        ax.set_title(title, fontsize=11)
                    
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # 在条形上添加数值
                    for i, (bar, rate) in enumerate(zip(bars, usage_rates)):
                        width = bar.get_width()
                        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                               f'{rate:.1f}%', ha='left', va='center', fontsize=8)
                else:
                    ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{cluster_info["cluster_name"]}\n({cluster_info["cluster_size"]}用户)', fontsize=11)
        
        # 隐藏多余的子图
        for idx in range(len(cluster_product_analysis), rows*cols):
            row = idx // cols
            col = idx % cols
            if row < rows and col < cols:
                axes[row, col].axis('off')
        
        plt.suptitle('各聚类产品偏好分析', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        
        # 打印详细分析结果
        print("\n各聚类产品偏好详情:")
        for cluster_info in cluster_product_analysis:
            print(f"\n{cluster_info['cluster_name']} (用户数: {cluster_info['cluster_size']}):")
            
            if len(cluster_info['top_products']) > 0:
                print("  最常使用的产品:")
                for i, (product, count, rate) in enumerate(zip(
                    cluster_info['top_products'][:5],
                    cluster_info['top_products_count'][:5],
                    cluster_info['top_products_usage_rate'][:5]
                )):
                    print(f"    {i+1}. {product}: {count}次使用，{rate:.1f}%的用户使用")
            else:
                print("  无产品使用数据")
            
            if cluster_info['top_equity_products']:
                print("  热门权益产品:")
                for i, product in enumerate(cluster_info['top_equity_products'][:3]):
                    print(f"    {i+1}. {product}")
            
            if cluster_info['top_non_equity_products']:
                print("  热门非权益产品:")
                for i, product in enumerate(cluster_info['top_non_equity_products'][:3]):
                    print(f"    {i+1}. {product}")
    
    else:
        print("  没有有效的聚类数据可用于产品偏好分析")
    
    return cluster_product_analysis

# 7. 修改主函数

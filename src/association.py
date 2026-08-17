# -*- coding: utf-8 -*-
"""产品关联分析模块：共现 / 支持度-置信度-提升度（Apriori 思想）+ 路径模式挖掘（脱敏版）。"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict


def analyze_product_associations_simple(df, user_df):
    """简化的产品关联分析"""
    print("\n6. 产品关联分析...")
    
    # 6.1 创建用户-产品矩阵
    print("创建用户-产品矩阵...")
    
    # 6.2 找出最常用的产品组合（共现分析）
    # 获取每个用户使用的产品列表
    user_products = df.groupby('user_id')['product_type'].apply(list).reset_index()
    
    # 找出频繁的产品对
    product_pairs = {}
    
    for products in user_products['product_type']:
        products = list(set(products))  # 去重
        if len(products) >= 2:
            # 生成所有产品对组合
            for i in range(len(products)):
                for j in range(i+1, len(products)):
                    pair = tuple(sorted([products[i], products[j]]))
                    product_pairs[pair] = product_pairs.get(pair, 0) + 1
    
    # 转换为DataFrame并排序
    pairs_df = pd.DataFrame([
        {'product1': pair[0], 'product2': pair[1], 'co_occurrence': count}
        for pair, count in product_pairs.items()
    ])
    
    if len(pairs_df) == 0:
        print("  未找到产品对数据")
        return pd.DataFrame()
    
    pairs_df = pairs_df.sort_values('co_occurrence', ascending=False)
    
    # 6.3 计算支持度、置信度和提升度
    total_users = len(user_df)
    
    # 计算每个产品的用户数
    product_user_counts = df.groupby('product_type')['user_id'].nunique()
    
    results = []
    for _, row in pairs_df.head(50).iterrows():  # 扩展到前50个产品对
        product1, product2 = row['product1'], row['product2']
        co_occurrence = row['co_occurrence']
        
        # 支持度 = 同时使用两个产品的用户数 / 总用户数
        support = co_occurrence / total_users
        
        # 置信度 (product1 -> product2) = 同时使用两个产品的用户数 / 使用product1的用户数
        if product1 in product_user_counts and product_user_counts[product1] > 0:
            confidence = co_occurrence / product_user_counts[product1]
        else:
            confidence = 0
        
        # 提升度 = 置信度 / (使用product2的用户数 / 总用户数)
        if product2 in product_user_counts and product_user_counts[product2] > 0:
            lift = confidence / (product_user_counts[product2] / total_users)
        else:
            lift = 0
        
        results.append({
            'product1': product1,
            'product2': product2,
            'co_occurrence': co_occurrence,
            'support': support,
            'confidence': confidence,
            'lift': lift
        })
    
    results_df = pd.DataFrame(results)
    
    # 6.4 可视化产品关联热力图
    plt.figure(figsize=(16, 12))
    
    # 创建热力图数据 - 使用所有产品
    all_products = sorted(df['product_type'].unique())
    heatmap_data = pd.DataFrame(index=all_products, columns=all_products)
    
    # 填充数据
    for _, row in results_df.iterrows():
        product1, product2 = row['product1'], row['product2']
        lift_value = row['lift']
        
        # 填充两个方向
        heatmap_data.loc[product1, product2] = lift_value
        heatmap_data.loc[product2, product1] = lift_value
    
    # 填充对角线为1（产品与自身的关联）
    for product in all_products:
        heatmap_data.loc[product, product] = 1.0
    
    # 填充缺失值为0（无关联）
    heatmap_data = heatmap_data.fillna(0).astype(float)
    
    # 绘制热力图
    mask = np.zeros_like(heatmap_data, dtype=bool)
    np.fill_diagonal(mask, True)  # 隐藏对角线
    
    # 创建调色板 - 使用离散颜色
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(heatmap_data, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap=cmap,
                center=1.0,  # 中心点为1
                vmin=0,      # 最小值为0
                vmax=3,      # 最大值为3
                cbar_kws={'label': '提升度', 'shrink': 0.8},
                square=True,
                linewidths=0.5,
                linecolor='gray')
    
    plt.title('产品关联分析热力图 (提升度)', fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # 6.5 输出提升度>1的关联产品组合
    print("\n" + "="*80)
    print("提升度>1的产品关联规则:")
    print("="*80)
    
    # 筛选提升度>1的规则
    high_lift_rules = results_df[
        (results_df['lift'] > 1.0) & 
        (results_df['confidence'] > 0)  # 确保置信度有效
    ].copy()
    
    if len(high_lift_rules) == 0:
        print("  未找到提升度>1的产品关联规则")
        return results_df
    
    # 排序：按提升度降序，再按置信度降序
    high_lift_rules = high_lift_rules.sort_values(['lift', 'confidence'], ascending=[False, False])
    
    # 输出格式化的关联规则
    print(f"\n发现 {len(high_lift_rules)} 条提升度>1的关联规则：")
    print("-" * 80)
    
    for idx, row in high_lift_rules.iterrows():
        product1, product2 = row['product1'], row['product2']
        
        print(f"\n规则 {idx+1}: {product1} → {product2}")
        print(f"  {'-'*40}")
        print(f"  共现用户数: {row['co_occurrence']:,}")
        print(f"  支持度:     {row['support']:.2%} ({row['co_occurrence']:,}/{total_users:,})")
        print(f"  置信度:     {row['confidence']:.2%}")
        print(f"  提升度:     {row['lift']:.2f}")
        
        # 添加业务解读
        if row['lift'] > 2.0:
            lift_interpret = "强关联"
        elif row['lift'] > 1.5:
            lift_interpret = "中等关联"
        else:
            lift_interpret = "弱关联"
        
        if row['confidence'] > 0.5:
            conf_interpret = "高转化"
        elif row['confidence'] > 0.3:
            conf_interpret = "中等转化"
        else:
            conf_interpret = "低转化"
        
        print(f"  解读:      {lift_interpret}，{conf_interpret}")
    
    # 6.6 按产品分组展示
    print("\n" + "="*80)
    print("按产品分组的关联规则汇总:")
    print("="*80)
    
    # 按product1分组
    for product in high_lift_rules['product1'].unique():
        product_rules = high_lift_rules[high_lift_rules['product1'] == product]
        
        if len(product_rules) > 0:
            print(f"\n使用 {product} 的用户，可能还会使用：")
            for _, rule in product_rules.iterrows():
                print(f"  • {rule['product2']}: 提升度={rule['lift']:.2f}, 置信度={rule['confidence']:.2%}")
    
    # 6.7 输出强关联规则（提升度>1.5且置信度>0.3）
    strong_rules = high_lift_rules[
        (high_lift_rules['lift'] > 1.5) & 
        (high_lift_rules['confidence'] > 0.3)
    ]
    
    if len(strong_rules) > 0:
        print("\n" + "="*80)
        print("强关联规则推荐（提升度>1.5且置信度>30%）：")
        print("="*80)
        
        for idx, row in strong_rules.iterrows():
            product1, product2 = row['product1'], row['product2']
            print(f"\n推荐 {product1} → {product2}:")
            print(f"  理由: 用户使用{product1}后，使用{product2}的可能性是普通用户的{row['lift']:.1f}倍")
            print(f"  预期转化率: {row['confidence']:.1%}")
            print(f"  影响用户数: {row['co_occurrence']:,}人")
            
            # 业务建议
            if row['lift'] > 2.0 and row['confidence'] > 0.4:
                print(f"  建议: 强推！考虑产品捆绑或自动推荐")
            elif row['lift'] > 1.5 and row['confidence'] > 0.3:
                print(f"  建议: 推荐！在产品界面添加引导")
    
    # 6.8 添加统计摘要
    print("\n" + "="*80)
    print("关联分析统计摘要:")
    print("="*80)
    print(f"总用户数: {total_users:,}")
    print(f"总产品数: {len(all_products)}")
    print(f"分析的产品对数量: {len(results_df)}")
    print(f"提升度>1的规则数: {len(high_lift_rules)}")
    print(f"提升度>1.5的规则数: {len(high_lift_rules[high_lift_rules['lift'] > 1.5])}")
    print(f"提升度>2.0的规则数: {len(high_lift_rules[high_lift_rules['lift'] > 2.0])}")
    
    # 计算平均指标
    if len(high_lift_rules) > 0:
        print(f"平均提升度: {high_lift_rules['lift'].mean():.2f}")
        print(f"平均置信度: {high_lift_rules['confidence'].mean():.2%}")
        print(f"平均支持度: {high_lift_rules['support'].mean():.2%}")
    
    return results_df



# 3. 调整聚类名称和数量（根据你的聚类结果）
# 修改 create_complete_labeling_system 函数中的聚类标签映射

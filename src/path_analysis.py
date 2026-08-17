# 行为路径挖掘：找长度>=3 的高频路径
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC',
                                   'Noto Sans CJK SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.show = lambda *a, **k: plt.savefig("report/figures/path_pattern.png",
                                       dpi=120, bbox_inches="tight")


def build_demo_sequences(csv_path="./data/sample_user_behavior.csv"):
    """从示例数据构造用户行为序列"""
    df = pd.read_csv(csv_path)
    df = df.sort_values(["user_id", "event_date"])
    seqs = []
    for uid, g in df.groupby("user_id"):
        seq = g["product_type"].tolist()
        seqs.append({"user_id": uid, "sequence": seq, "length": len(seq)})
    return seqs


def analyze_common_paths(user_sequences):
    """挖掘并可视化常见行为路径模式。"""

    def find_common_paths_user_based(sequences, min_length=3, max_length=5):
        """
        找出常见路径模式（基于用户维度）
        每个用户对每个路径模式只计数一次
        """
        # 用于存储每个路径模式覆盖的用户数
        path_user_counter = Counter()
        # 用于存储每个路径模式的总出现次数
        path_total_counter = Counter()
        # 用于记录每个用户已经统计过的路径，避免重复
        user_paths_record = defaultdict(set)

        for seq_info in sequences:
            seq = seq_info['sequence']
            user_id = seq_info['user_id']
            seq_len = len(seq)

            # 记录这个用户在当前序列中出现的所有路径
            paths_in_this_sequence = set()

            # 提取长度为min_length到max_length的连续子序列
            for length in range(min_length, min(max_length + 1, seq_len + 1)):
                for i in range(seq_len - length + 1):
                    subseq = tuple(seq[i:i+length])
                    paths_in_this_sequence.add(subseq)
                    # 总出现次数+1
                    path_total_counter[subseq] += 1

            # 对于这个用户新发现的路径，更新用户计数
            for path in paths_in_this_sequence:
                if path not in user_paths_record[user_id]:
                    user_paths_record[user_id].add(path)
                    path_user_counter[path] += 1

        return path_user_counter, path_total_counter, user_paths_record

    # 计算基于用户的路径模式
    user_based_paths, total_occurrences, user_paths_record = find_common_paths_user_based(user_sequences)

    # 总用户数（有长度≥3路径的用户）
    users_with_long_paths = len([s for s in user_sequences if s['length'] >= 3])
    print(f"有长度≥3路径的用户总数: {users_with_long_paths}")

    # 获取最常见的15个路径模式（按用户覆盖率排序）
    top_user_based_paths = user_based_paths.most_common(20)

    print("\n最常见的路径模式（长度≥3，按用户覆盖率）:")
    print("-" * 80)
    for path_tuple, user_count in top_user_based_paths[:15]:
        path_str = ' → '.join(path_tuple)
        user_percentage = (user_count / users_with_long_paths * 100) if users_with_long_paths > 0 else 0
        total_occ = total_occurrences[path_tuple]

        # 计算平均每个用户出现这个路径的次数
        avg_per_user = total_occ / user_count if user_count > 0 else 0

        print(f"  {path_str}:")
        print(f"    覆盖用户数: {user_count} ({user_percentage:.1f}%的用户)")
        print(f"    总出现次数: {total_occ}")
        print(f"    平均每用户出现: {avg_per_user:.2f}次")
        print()

    # 可视化：按用户覆盖率排序的Top路径
    plt.figure(figsize=(14, 8))

    # 准备数据
    top_paths_for_viz = top_user_based_paths[:10]
    path_labels = [' → '.join(p[0]) for p in top_paths_for_viz]
    user_counts = [p[1] for p in top_paths_for_viz]
    user_percentages = [count / users_with_long_paths * 100 for count in user_counts]

    # 创建水平条形图
    bars = plt.barh(range(len(path_labels)), user_percentages, color='lightseagreen')
    plt.yticks(range(len(path_labels)), path_labels)
    plt.xlabel('用户覆盖率 (%)')
    plt.title('Top 10 常见路径模式（按用户覆盖率）')
    plt.gca().invert_yaxis()

    # 添加数值标签
    for i, (bar, percentage, count) in enumerate(zip(bars, user_percentages, user_counts)):
        plt.text(percentage + 1, bar.get_y() + bar.get_height()/2, 
                 f'{percentage:.1f}% ({count}用户)', va='center')

    plt.tight_layout()
    plt.show()

    # 也可以按总出现次数排序查看
    print("\n最常见的路径模式（长度≥3，按总出现次数）:")
    print("-" * 80)
    top_by_occurrence = total_occurrences.most_common(15)
    for path_tuple, total_occ in top_by_occurrence[:15]:
        path_str = ' → '.join(path_tuple)
        user_count = user_based_paths[path_tuple]
        user_percentage = (user_count / users_with_long_paths * 100) if users_with_long_paths > 0 else 0

        print(f"  {path_str}:")
        print(f"    总出现次数: {total_occ}")
        print(f"    覆盖用户数: {user_count} ({user_percentage:.1f}%的用户)")
        print()
    
    # 额外：按路径长度统计
    print("\n按路径长度统计:")
    print("-" * 80)

    path_length_stats = defaultdict(lambda: {'paths': 0, 'users': set(), 'occurrences': 0})

    for path_tuple, user_count in user_based_paths.items():
        path_length = len(path_tuple)
        path_length_stats[path_length]['paths'] += 1
        path_length_stats[path_length]['occurrences'] += total_occurrences[path_tuple]
        path_length_stats[path_length]['users'].update(user_paths_record.keys())

    for length in sorted(path_length_stats.keys()):
        stats = path_length_stats[length]
        unique_users = len(stats['users'])
        print(f"  长度 {length} 的路径:")
        print(f"    不同路径数: {stats['paths']}")
        print(f"    总出现次数: {stats['occurrences']}")
        print(f"    涉及用户数: {unique_users} ({unique_users/users_with_long_paths*100:.1f}%)")
        print()    


if __name__ == "__main__":
    seqs = build_demo_sequences()
    analyze_common_paths(seqs)

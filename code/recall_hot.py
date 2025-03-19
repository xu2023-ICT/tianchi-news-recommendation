import pandas as pd
import numpy as np
import cupy as cp
from tqdm import tqdm
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

def sample_test_data(df_click, df_query, test_size):
    """
    更智能的测试数据采样
    
    Args:
        df_click: 点击数据
        df_query: 查询数据
        test_size: 目标测试用户数量
    """
    # 1. 确保用户有足够的历史点击数据
    user_click_counts = df_click.groupby('user_id').size()
    qualified_users = user_click_counts[user_click_counts >= 5].index  # 至少有5次点击
    
    # 2. 获取有效的查询用户（确保有target item）
    valid_query_users = df_query[df_query['click_article_id'] != -1]['user_id'].unique()
    
    # 3. 找到同时满足条件的用户
    valid_users = set(qualified_users) & set(valid_query_users)
    
    # 4. 采样用户
    if len(valid_users) < test_size:
        print(f"警告：符合条件的用户数({len(valid_users)})小于请求的测试规模({test_size})")
        sampled_users = list(valid_users)
    else:
        sampled_users = np.random.choice(list(valid_users), size=test_size, replace=False)
    
    # 5. 对每个用户保留完整的时间序列数据
    df_click_sampled = df_click[df_click['user_id'].isin(sampled_users)]
    df_query_sampled = df_query[df_query['user_id'].isin(sampled_users)]
    
    # 6. 确保每个用户都有查询数据
    final_users = set(df_query_sampled['user_id'].unique())
    df_click_sampled = df_click_sampled[df_click_sampled['user_id'].isin(final_users)]
    
    return df_click_sampled, df_query_sampled

def get_article_popularity_in_time_window_gpu(df_click, time_window=3600):
    """使用GPU加速计算文章在时间窗口内的热度"""
    df_click = df_click.sort_values('click_timestamp')
    timestamps = cp.array(df_click['click_timestamp'].values)
    article_ids = cp.array(df_click['click_article_id'].values)
    
    popularity_dict = {}
    for i in tqdm(range(len(timestamps))):
        curr_time = timestamps[i]
        window_start = curr_time - time_window
        mask = (timestamps >= window_start) & (timestamps < curr_time)
        window_articles = article_ids[mask]
        unique_articles, counts = cp.unique(window_articles, return_counts=True)
        popularity_dict[float(curr_time)] = dict(zip(unique_articles.get(), counts.get()))
    
    return popularity_dict

def recall_hot_articles(df_query, df_click, popularity_dict, top_k=30):
    """基于热度的文章召回"""
    # 获取用户的历史点击时间
    user_clicks = df_click.groupby('user_id').agg({
        'click_timestamp': list,
        'click_article_id': list
    }).to_dict('index')
    
    recall_results = []
    for user_id, target_item in tqdm(df_query.values):
        if user_id not in user_clicks:
            continue
        
        # 获取用户最后一次点击时间（排除目标文章的点击）
        user_times = user_clicks[user_id]['click_timestamp']
        if not user_times:
            continue
            
        last_click_time = max(user_times)
        
        # 获取该时间点的热门文章
        if last_click_time in popularity_dict:
            time_popularity = popularity_dict[last_click_time]
        else:
            available_times = np.array(list(popularity_dict.keys()))
            nearest_time = available_times[np.argmin(np.abs(available_times - last_click_time))]
            time_popularity = popularity_dict[nearest_time]
        
        # 排序并召回
        sorted_articles = sorted(time_popularity.items(), key=lambda x: x[1], reverse=True)
        top_articles = sorted_articles[:top_k]
        
        for article_id, pop_score in top_articles:
            recall_results.append({
                'user_id': int(user_id),
                'article_id': int(article_id),
                'sim_score': float(pop_score),
                'label': 1 if article_id == target_item else 0 if target_item != -1 else np.nan
            })
    
    return pd.DataFrame(recall_results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='基于热度的文章召回')
    parser.add_argument('--mode', default='test', choices=['valid', 'online', 'test'])
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--time_window', type=int, default=3600)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--min_clicks', type=int, default=5)
    args = parser.parse_args()
    
    # 读取数据
    print("读取数据...")
    if args.mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
    elif args.mode == 'test':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        
        print(f"采样测试数据...")
        df_click, df_query = sample_test_data(df_click, df_query, args.test_size)
        
        print(f"测试集统计:")
        print(f"用户数: {df_click['user_id'].nunique()}")
        print(f"点击数: {len(df_click)}")
        print(f"查询数: {len(df_query)}")
        print(f"平均每用户点击数: {len(df_click) / df_click['user_id'].nunique():.2f}")
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
    
    # 计算文章热度
    print("\n计算文章热度...")
    popularity_dict = get_article_popularity_in_time_window_gpu(df_click, time_window=args.time_window)
    
    # 召回热门文章
    print("\n执行文章召回...")
    df_recall = recall_hot_articles(df_query, df_click, popularity_dict, top_k=args.top_k)
    
    # 计算并输出指标
    if args.mode in ['valid', 'test']:
        valid_recalls = df_recall[df_recall['label'].notnull()]
        total_users = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hit_users = valid_recalls[valid_recalls['label'] == 1].user_id.nunique()
        
        print("\n======== 召回指标详情 ========")
        print(f"总用户数: {total_users}")
        print(f"命中用户数: {hit_users}")
        hit_rate = hit_users / total_users
        print(f"命中率: {hit_rate:.4f} ({hit_rate*100:.1f}%)")
        print(f"目标命中率: 67.9%")
        print(f"是否达标: {'✓' if hit_rate >= 0.679 else '✗'}")
        
        if valid_recalls[valid_recalls['label'] == 1].shape[0] > 0:
            mean_rank = valid_recalls[valid_recalls['label'] == 1]['sim_score'].rank(ascending=False).mean()
            print(f"平均召回位置: {mean_rank:.2f}")
        
        user_stats = valid_recalls.groupby('user_id').agg({
            'article_id': 'count',
            'label': 'sum'
        })
        print("\n-------- 用户级别统计 --------")
        print(f"平均每用户召回文章数: {user_stats['article_id'].mean():.2f}")
        print(f"平均每用户命中文章数: {user_stats['label'].mean():.2f}")
        print("============================\n")
    
    # 保存结果
    save_dir = f'../user_data/data/{"test" if args.mode == "test" else args.mode}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/recall_hot.pkl'
    df_recall.to_pickle(save_path)
    print(f'召回结果已保存至: {save_path}')
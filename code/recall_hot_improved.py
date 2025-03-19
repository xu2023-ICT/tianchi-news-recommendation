import pandas as pd
import numpy as np
import cupy as cp
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

def get_user_last_click(df_click, user_id, exclude_timestamp=None):
    """获取用户最后一次点击时间（排除待预测的点击）"""
    user_clicks = df_click[df_click['user_id'] == user_id]
    if exclude_timestamp is not None:
        user_clicks = user_clicks[user_clicks['click_timestamp'] != exclude_timestamp]
    if len(user_clicks) == 0:
        return None
    return user_clicks['click_timestamp'].max()

def get_time_window_articles(df_click, timestamp, window_size=7200):  # 增加到2小时
    """获取指定时间点之前时间窗口内的文章热度"""
    # 使用更大的时间窗口
    window_start = timestamp - window_size
    
    window_clicks = df_click[
        (df_click['click_timestamp'] >= window_start) &
        (df_click['click_timestamp'] < timestamp)
    ]
    
    if len(window_clicks) == 0:
        return {}
    
    # 改进的时间衰减权重计算
    time_diff = timestamp - window_clicks['click_timestamp']
    # 使用更平缓的对数衰减
    time_weights = 1 / (1 + np.log1p(time_diff / (window_size/4)))
    
    # 计算文章热度得分
    article_scores = {}
    for article_id in window_clicks['click_article_id'].unique():
        article_clicks = window_clicks[window_clicks['click_article_id'] == article_id]
        article_idx = article_clicks.index
        
        # 结合多个因素计算文章得分
        base_score = len(article_clicks)  # 基础点击量
        time_score = time_weights[article_idx].mean()  # 时间衰减
        
        # 计算用户多样性得分
        unique_users = len(article_clicks['user_id'].unique())
        diversity_score = unique_users / base_score  # 用户多样性
        
        # 综合评分
        article_scores[article_id] = (
            base_score * 
            time_score * 
            (1 + np.log1p(diversity_score))  # 用户多样性提升
        )
    
    return article_scores

def get_category_preferences(df_click, user_id, time_decay=True):
    """获取用户的分类偏好，支持时间衰减和长期兴趣"""
    user_clicks = df_click[df_click['user_id'] == user_id]
    if len(user_clicks) == 0:
        return {}
    
    # 分别计算长期和短期兴趣
    latest_time = user_clicks['click_timestamp'].max()
    time_diff = latest_time - user_clicks['click_timestamp']
    
    # 短期兴趣（最近24小时）
    short_term_mask = time_diff <= 24*3600
    short_term_clicks = user_clicks[short_term_mask]
    
    # 长期兴趣（所有历史）
    category_counts = user_clicks['category_id'].value_counts()
    total_clicks = category_counts.sum()
    long_term_prefs = (category_counts / total_clicks).to_dict()
    
    if len(short_term_clicks) > 0:
        # 计算短期兴趣
        short_term_counts = short_term_clicks['category_id'].value_counts()
        short_term_prefs = (short_term_counts / short_term_counts.sum()).to_dict()
        
        # 融合长期和短期兴趣（短期占比更大）
        final_prefs = {}
        all_categories = set(long_term_prefs.keys()) | set(short_term_prefs.keys())
        for cat in all_categories:
            short_term = short_term_prefs.get(cat, 0)
            long_term = long_term_prefs.get(cat, 0)
            final_prefs[cat] = 0.7 * short_term + 0.3 * long_term
    else:
        final_prefs = long_term_prefs
    
    return final_prefs

def adjust_scores_by_category(article_scores, df_article, user_preferences):
    """根据用户分类偏好调整文章得分"""
    adjusted_scores = {}
    for article_id, base_score in article_scores.items():
        category_id = df_article.loc[df_article['article_id'] == article_id, 'category_id'].iloc[0]
        category_weight = user_preferences.get(category_id, 0.1)
        
        # 使用非线性提升
        boost = np.sqrt(1 + category_weight)  # 平滑的非线性提升
        adjusted_scores[article_id] = base_score * boost
    
    return adjusted_scores

def recall_hot_articles(df_query, df_click, df_article, time_window=7200, top_k=50):  # 增加召回数量
    """改进的基于热度的文章召回"""
    article_to_category = df_article.set_index('article_id')['category_id'].to_dict()
    df_click = df_click.sort_values('click_timestamp')
    
    recall_results = []
    for user_id, target_item in tqdm(df_query.values):
        # 获取用户最后一次非待预测点击时间
        last_click_time = get_user_last_click(df_click, user_id, target_item)
        if last_click_time is None:
            continue
            
        # 获取时间窗口内的热门文章
        article_scores = get_time_window_articles(df_click, last_click_time, time_window)
        if not article_scores:
            continue
            
        # 获取用户已读文章
        user_history = set(df_click[df_click['user_id'] == user_id]['click_article_id'])
        
        # 过滤掉已读文章
        article_scores = {k: v for k, v in article_scores.items() if k not in user_history}
        
        # 获取用户分类偏好
        user_preferences = get_category_preferences(df_click, user_id, time_decay=True)
        
        # 根据用户分类偏好调整得分
        adjusted_scores = adjust_scores_by_category(article_scores, df_article, user_preferences)
        
        # 排序并选择top_k
        sorted_articles = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 创建召回结果
        recall_results.extend([{
            'user_id': int(user_id),
            'article_id': int(article_id),
            'sim_score': float(score),
            'label': 1 if article_id == target_item else 0 if target_item != -1 else np.nan
        } for article_id, score in sorted_articles])
    
    return pd.DataFrame(recall_results)

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
    qualified_users = user_click_counts[user_click_counts >= 5].index
    
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='改进的基于热度的文章召回')
    parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
    parser.add_argument('--time_window', type=int, default=3600)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--test_size', type=int, default=1000)  # 添加测试规模参数
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
    
    # 读取文章信息
    df_article = pd.read_csv('data/articles.csv')
    
    # 数据预处理
    print("预处理数据...")
    df_click = df_click.merge(
        df_article[['article_id', 'category_id']], 
        left_on='click_article_id',
        right_on='article_id',
        how='left'
    )
    
    # 执行召回
    print("执行文章召回...")
    df_recall = recall_hot_articles(
        df_query, 
        df_click, 
        df_article,
        time_window=args.time_window,
        top_k=args.top_k
    )
    
    # 排序
    df_recall = df_recall.sort_values(
        ['user_id', 'sim_score'], 
        ascending=[True, False]
    ).reset_index(drop=True)
    
    # 计算指标
    if args.mode in ['valid', 'test']:
        total_users = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        valid_recalls = df_recall[df_recall['label'].notnull()]
        hit_users = valid_recalls[valid_recalls['label'] == 1].user_id.nunique()
        hit_rate = hit_users / total_users
        
        print("\n======== 召回指标详情 ========")
        print(f"总用户数: {total_users}")
        print(f"命中用户数: {hit_users}")
        print(f"命中率: {hit_rate:.4f} ({hit_rate*100:.1f}%)")
        print(f"目标命中率: 67.9%")
        print(f"是否达标: {'✓' if hit_rate >= 0.679 else '✗'}")
        
        if valid_recalls[valid_recalls['label'] == 1].shape[0] > 0:
            mean_rank = valid_recalls[valid_recalls['label'] == 1].groupby('user_id')['sim_score'].rank(ascending=False).mean()
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
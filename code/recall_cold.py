import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import os
import pickle
from datetime import datetime
from collections import defaultdict
from utils import Logger, evaluate
warnings.filterwarnings('ignore')

def get_click_article_ids_set(df_click):
    """获取所有点击过的文章集合"""
    return set(df_click.click_article_id.values)

def get_article_info_dict(df_article):
    """获取文章属性字典"""
    max_min_scaler = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
    df_article['created_at_ts'] = df_article[['created_at_ts']].apply(max_min_scaler)
    
    return {
        'type': dict(zip(df_article['article_id'], df_article['category_id'])),
        'words': dict(zip(df_article['article_id'], df_article['words_count'])),
        'created_time': dict(zip(df_article['article_id'], df_article['created_at_ts']))
    }

def get_user_hist_info(df_click, df_article):
    """获取用户历史行为信息"""
    df = df_click.merge(df_article[['article_id', 'category_id', 'words_count', 'created_at_ts']], 
                       left_on='click_article_id', 
                       right_on='article_id', 
                       how='left')
    
    user_hist = defaultdict(lambda: {'types': set(), 'words_mean': 0, 'last_time': 0})
    
    for user_id, group in df.groupby('user_id'):
        user_hist[user_id]['types'] = set(group['category_id'])
        user_hist[user_id]['words_mean'] = group['words_count'].mean()
        user_hist[user_id]['last_time'] = group['created_at_ts'].max()
    
    return user_hist

def cold_start_recall(df_click, df_article, df_query, time_window=90, words_delta=200, emb_sim_path=None, top_k=100):
    """
    冷启动召回算法
    
    Args:
        df_click: 点击数据
        df_article: 文章数据
        df_query: 待推荐用户
        time_window: 时间窗口(天)
        words_delta: 文章字数差异容忍度
        emb_sim_path: 文章embedding相似度矩阵路径
        top_k: 召回数量
    """
    # 获取文章信息
    article_info = get_article_info_dict(df_article)
    user_hist = get_user_hist_info(df_click, df_article)
    click_articles = get_click_article_ids_set(df_click)
    
    # 加载embedding相似度矩阵
    if emb_sim_path and os.path.exists(emb_sim_path):
        with open(emb_sim_path, 'rb') as f:
            emb_sim_dict = pickle.load(f)
    else:
        emb_sim_dict = {}
    
    # 获取热门文章作为补充
    popular_items = df_click['click_article_id'].value_counts().head(1000).index.tolist()
    
    recall_results = defaultdict(list)
    
    for user_id in tqdm(df_query['user_id'].unique()):
        if user_id not in user_hist:
            # 对于完全冷启动的用户，直接返回热门文章
            recall_results[user_id] = [(item, 1.0/(i+1)) for i, item in enumerate(popular_items[:top_k])]
            continue
            
        user_info = user_hist[user_id]
        candidate_items = set()
        
        # 1. 基于用户历史兴趣的文章筛选
        for article_id in df_article['article_id']:
            if article_id in click_articles:
                continue
                
            # 类型匹配
            if article_info['type'][article_id] not in user_info['types']:
                continue
                
            # 字数匹配
            if abs(article_info['words'][article_id] - user_info['words_mean']) > words_delta:
                continue
                
            # 时间窗口匹配
            time_diff = abs(article_info['created_time'][article_id] - user_info['last_time'])
            if time_diff > time_window:
                continue
                
            candidate_items.add(article_id)
            
        # 2. 计算候选文章得分
        item_scores = []
        for item in candidate_items:
            score = 0.0
            
            # 主题相关性得分
            type_score = 1.0 if article_info['type'][item] in user_info['types'] else 0.0
            
            # 文章新鲜度得分
            time_score = 1.0 / (1 + abs(article_info['created_time'][item] - user_info['last_time']))
            
            # 文章字数相似度得分
            words_score = 1.0 / (1 + abs(article_info['words'][item] - user_info['words_mean'])/words_delta)
            
            # embedding相似度得分
            emb_score = 0.0
            if item in emb_sim_dict:
                emb_score = max([emb_sim_dict[item].get(h, 0.0) for h in user_hist[user_id]['types']])
            
            # 综合得分
            score = (type_score * 0.4 + time_score * 0.3 + words_score * 0.2 + emb_score * 0.1)
            item_scores.append((item, score))
        
        # 3. 排序并补充热门文章
        item_scores.sort(key=lambda x: x[1], reverse=True)
        if len(item_scores) < top_k:
            for item in popular_items:
                if item not in candidate_items:
                    item_scores.append((item, 0.0))
                if len(item_scores) >= top_k:
                    break
        
        recall_results[user_id] = item_scores[:top_k]
    
    return recall_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='冷启动召回')
    parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
    parser.add_argument('--time_window', type=int, default=90)
    parser.add_argument('--words_delta', type=int, default=200)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--logfile', default='cold_start.log')
    args = parser.parse_args()
    
    # 初始化日志
    os.makedirs('../user_data/log', exist_ok=True)
    log = Logger(f'../user_data/log/{args.logfile}').logger
    log.info(f'冷启动召回开始运行，模式: {args.mode}')
    
    # 读取数据
    log.info("读取数据...")
    if args.mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
    elif args.mode == 'test':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        
        # 随机选择一部分用户
        test_users = df_query['user_id'].sample(n=args.test_size, random_state=2024)
        df_query = df_query[df_query['user_id'].isin(test_users)]
        df_click = df_click[df_click['user_id'].isin(test_users)]
        
        log.info(f'测试模式: 选取{args.test_size}个用户')
        log.info(f'点击数据大小: {df_click.shape}')
        log.info(f'查询数据大小: {df_query.shape}')
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
    
    log.debug(f'点击数据样例:\n{df_click.head()}')
    
    # 读取文章信息
    df_article = pd.read_csv('data/articles.csv')
    log.debug(f'文章数据大小: {df_article.shape}')
    
    # 执行冷启动召回
    log.info(f"执行冷启动召回... time_window={args.time_window}, words_delta={args.words_delta}, top_k={args.top_k}")
    emb_sim_path = '../user_data/sim/offline/emb_i2i_sim.pkl'
    recall_results = cold_start_recall(
        df_click, 
        df_article,
        df_query,
        time_window=args.time_window,
        words_delta=args.words_delta,
        emb_sim_path=emb_sim_path,
        top_k=args.top_k
    )
    
    # 转换为DataFrame格式
    log.info("整理召回结果...")
    recall_df = []
    for user_id, items in recall_results.items():
        for item_id, score in items:
            recall_df.append({
                'user_id': int(user_id),
                'article_id': int(item_id),
                'sim_score': float(score),
                'label': 1 if item_id == df_query[df_query['user_id']==user_id]['click_article_id'].iloc[0] 
                         else 0 if df_query[df_query['user_id']==user_id]['click_article_id'].iloc[0] != -1 
                         else np.nan
            })
    
    df_recall = pd.DataFrame(recall_df)
    log.debug(f'召回结果大小: {df_recall.shape}')
    
    # 计算指标
    if args.mode in ['valid', 'test']:
        log.info('计算召回指标...')
        total_users = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        valid_recalls = df_recall[df_recall['label'].notnull()]
        hit_users = valid_recalls[valid_recalls['label'] == 1].user_id.nunique()
        hit_rate = hit_users / total_users
        
        log.info("\n======== 召回指标详情 ========")
        log.info(f"总用户数: {total_users}")
        log.info(f"命中用户数: {hit_users}")
        log.info(f"命中率: {hit_rate:.4f} ({hit_rate*100:.1f}%)")
        
        if valid_recalls[valid_recalls['label'] == 1].shape[0] > 0:
            mean_rank = valid_recalls[valid_recalls['label'] == 1].groupby('user_id')['sim_score'].rank(ascending=False).mean()
            log.info(f"平均召回位置: {mean_rank:.2f}")
        
        user_stats = valid_recalls.groupby('user_id').agg({
            'article_id': 'count',
            'label': 'sum'
        })
        log.info("\n-------- 用户级别统计 --------")
        log.info(f"平均每用户召回文章数: {user_stats['article_id'].mean():.2f}")
        log.info(f"平均每用户命中文章数: {user_stats['label'].mean():.2f}")
        log.info("============================")
    
    # 保存结果
    save_dir = f'../user_data/data/{"test" if args.mode == "test" else args.mode}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/recall_cold.pkl'
    df_recall.to_pickle(save_path)
    log.info(f'召回结果已保存至: {save_path}')

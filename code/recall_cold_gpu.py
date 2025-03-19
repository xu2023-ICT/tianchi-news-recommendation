import pandas as pd
import numpy as np
import cupy as cp
from tqdm import tqdm
import warnings
import os
import pickle
from datetime import datetime
from collections import defaultdict
from utils import Logger, evaluate
import concurrent.futures
import math
from numba import jit
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_adjusted_score(base_score: float, time_diff: float, category_weight: float) -> float:
    """使用Numba加速的得分计算"""
    time_decay = 1.0 / (1 + math.log1p(time_diff))
    return base_score * time_decay * (1 + math.sqrt(category_weight))

def to_gpu(data):
    """将数据转移到GPU"""
    if isinstance(data, np.ndarray):
        return cp.asarray(data)
    return data

def to_cpu(data):
    """将数据转移回CPU"""
    if isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return data

def preprocess_data(df_click: pd.DataFrame, df_article: pd.DataFrame) -> tuple[dict, dict]:
    """优化的数据预处理"""
    # 提前计算所有需要的特征
    scaler = MinMaxScaler()
    df_article['created_at_ts'] = scaler.fit_transform(df_article[['created_at_ts']])
    
    # 转换为GPU张量，预处理文章特征
    article_features = {
        'type': cp.asarray(df_article['category_id'].values, dtype=cp.int32),
        'words': cp.asarray(df_article['words_count'].values, dtype=cp.float32),
        'created_time': cp.asarray(df_article['created_at_ts'].values, dtype=cp.float32),
        'article_ids': cp.asarray(df_article['article_id'].values, dtype=cp.int32)
    }
    
    # 预计算用户特征
    user_features = {}
    for user_id in df_click['user_id'].unique():
        user_data = df_click[df_click['user_id'] == user_id]
        user_features[user_id] = {
            'types': cp.asarray([user_data['category_id'].iloc[-1]], dtype=cp.int32),  # 最后一次点击的类别
            'words_mean': cp.asarray([user_data['words_count'].mean()], dtype=cp.float32),
            'last_time': cp.asarray([user_data['created_at_ts'].max()], dtype=cp.float32)
        }
    
    return article_features, user_features

def get_article_info_dict(df_article):
    """获取文章属性字典并转换为GPU tensor"""
    max_min_scaler = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))
    df_article['created_at_ts'] = df_article[['created_at_ts']].apply(max_min_scaler)
    
    # 转换为numpy数组再转为cupy数组
    return {
        'type': cp.asarray(df_article['category_id'].values),
        'words': cp.asarray(df_article['words_count'].values),
        'created_time': cp.asarray(df_article['created_at_ts'].values),
        'article_ids': cp.asarray(df_article['article_id'].values)
    }

def batch_compute_similarity(user_features, article_features, words_delta):
    """批量计算用户和文章之间的相似度"""
    # 扩展维度以便广播
    user_types = user_features['types'][:, None]  # [n_users, 1]
    user_words = user_features['words_mean'][:, None]  # [n_users, 1]
    user_times = user_features['last_time'][:, None]  # [n_users, 1]
    
    # 文章特征
    article_types = article_features['type'][None, :]  # [1, n_articles]
    article_words = article_features['words'][None, :]  # [1, n_articles]
    article_times = article_features['created_time'][None, :]  # [1, n_articles]
    
    # 计算各种相似度得分
    type_score = (user_types == article_types).astype(cp.float32)
    words_score = 1.0 / (1 + cp.abs(user_words - article_words) / words_delta)
    time_score = 1.0 / (1 + cp.abs(user_times - article_times))
    
    # 综合得分
    final_score = (
        type_score * 0.4 + 
        time_score * 0.3 + 
        words_score * 0.3
    )
    
    return final_score

def generate_hard_negatives(df_click, df_article, user_id, positive_items, n_samples=5):
    """生成难负样本
    
    通过选择与正样本相似但用户未交互的文章作为难负样本
    """
    user_clicks = df_click[df_click['user_id'] == user_id]
    user_categories = set(user_clicks['category_id'])
    
    # 获取同类别但未交互的文章
    candidate_articles = df_article[
        (df_article['category_id'].isin(user_categories)) & 
        (~df_article['article_id'].isin(positive_items))
    ]
    
    if len(candidate_articles) < n_samples:
        # 补充随机负样本
        other_articles = df_article[~df_article['article_id'].isin(positive_items)]
        candidate_articles = pd.concat([candidate_articles, other_articles])
    
    return candidate_articles['article_id'].sample(n=min(n_samples, len(candidate_articles))).values

def batch_compute_similarity_with_negatives(user_features, article_features, negative_samples, words_delta):
    """带负采样的批量相似度计算"""
    # 原有的相似度计算
    sim_matrix = batch_compute_similarity(user_features, article_features, words_delta)
    
    # 对每个用户的负样本，降低其相似度得分
    for user_idx, neg_items in negative_samples.items():
        item_indices = [i for i, aid in enumerate(to_cpu(article_features['article_ids'])) if aid in neg_items]
        if item_indices:
            sim_matrix[user_idx, item_indices] *= 0.5  # 降低负样本的得分
    
    return sim_matrix

def process_user_batch(batch_data: dict) -> list[dict]:
    """并行处理用户批次"""
    user_ids = batch_data['user_ids']
    user_features_dict = batch_data['user_features']
    article_features = batch_data['article_features']
    words_delta = batch_data['words_delta']
    top_k = batch_data['top_k']
    
    # 为批次构建特征矩阵
    batch_size = len(user_ids)
    user_feature_arrays = {
        'types': cp.zeros((batch_size, 1), dtype=cp.int32),
        'words_mean': cp.zeros((batch_size, 1), dtype=cp.float32),
        'last_time': cp.zeros((batch_size, 1), dtype=cp.float32)
    }
    
    # 填充特征矩阵
    for idx, user_id in enumerate(user_ids):
        if user_id in user_features_dict:
            user_feat = user_features_dict[user_id]
            user_feature_arrays['types'][idx] = user_feat['types']
            user_feature_arrays['words_mean'][idx] = user_feat['words_mean']
            user_feature_arrays['last_time'][idx] = user_feat['last_time']
    
    # 在GPU上计算相似度矩阵
    sim_matrix = cp.zeros((batch_size, len(article_features['article_ids'])), dtype=cp.float32)
    
    # 分块计算以避免GPU内存溢出
    chunk_size = 1000
    for i in range(0, sim_matrix.shape[1], chunk_size):
        end_idx = min(i + chunk_size, sim_matrix.shape[1])
        sim_matrix[:, i:end_idx] = batch_compute_similarity(
            user_feature_arrays,
            {k: v[i:end_idx] for k, v in article_features.items()},
            words_delta
        )
    
    # 获取top_k结果
    results = []
    for idx, user_id in enumerate(user_ids):
        scores = cp.asnumpy(sim_matrix[idx])
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        
        for article_idx in top_indices:
            results.append({
                'user_id': int(user_id),
                'article_id': int(article_features['article_ids'][article_idx]),
                'sim_score': float(scores[article_idx])
            })
    
    return results

def cold_start_recall_gpu(df_click, df_article, df_query, batch_size=1000, time_window=90, 
                         words_delta=200, top_k=100, n_workers=4):
    """优化的GPU加速冷启动召回"""
    # 预处理数据
    article_features, user_features = preprocess_data(df_click, df_article)
    
    # 准备批处理数据
    user_batches = np.array_split(df_query['user_id'].unique(), 
                                 max(1, len(df_query) // batch_size))
    
    batch_data = []
    for user_batch in user_batches:
        batch_data.append({
            'user_ids': user_batch,
            'user_features': user_features,  # 传递整个用户特征字典
            'article_features': article_features,
            'words_delta': words_delta,
            'top_k': top_k
        })
    
    # 并行处理批次
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_user_batch, data) for data in batch_data]
        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(futures), 
                         desc="Processing batches"):
            results.extend(future.result())
    
    return pd.DataFrame(results)

def add_diversity_boost(recall_results, df_article, diversity_weight=0.2):
    """增加多样性提升"""
    for user_id in recall_results.keys():
        items = recall_results[user_id]
        if not items:
            continue
            
        # 统计已选择文章的类别分布
        category_counts = defaultdict(int)
        for item_id, _ in items:
            category = df_article[df_article['article_id'] == item_id]['category_id'].iloc[0]
            category_counts[category] += 1
            
        # 根据类别多样性调整得分
        adjusted_items = []
        for item_id, score in items:
            category = df_article[df_article['article_id'] == item_id]['category_id'].iloc[0]
            diversity_penalty = category_counts[category] / len(items)
            adjusted_score = score * (1 - diversity_weight * diversity_penalty)
            adjusted_items.append((item_id, adjusted_score))
            
        # 重新排序
        recall_results[user_id] = sorted(adjusted_items, key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GPU加速的冷启动召回')
    parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--time_window', type=int, default=90)
    parser.add_argument('--words_delta', type=int, default=200)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--logfile', default='cold_start_gpu.log')
    parser.add_argument('--n_negatives', type=int, default=5, help='每个用户的负样本数量')
    parser.add_argument('--diversity_weight', type=float, default=0.2, help='多样性权重')
    parser.add_argument('--n_workers', type=int, default=4, help='并行处理的工作线程数')
    parser.add_argument('--chunk_size', type=int, default=1000, help='GPU处理的分块大小')
    args = parser.parse_args()
    
    # 初始化日志
    os.makedirs('../user_data/log', exist_ok=True)
    log = Logger(f'../user_data/log/{args.logfile}').logger
    log.info(f'GPU加速冷启动召回开始运行，模式: {args.mode}')
    
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
        
    # 读取文章信息
    df_article = pd.read_csv('data/articles.csv')
    log.debug(f'文章数据大小: {df_article.shape}')
    
    # 数据预处理
    log.info("预处理数据...")
    df_click = df_click.merge(
        df_article[['article_id', 'category_id', 'words_count', 'created_at_ts']], 
        left_on='click_article_id',
        right_on='article_id',
        how='left'
    )
    
    # 执行改进的GPU加速召回
    log.info(f"执行GPU加速冷启动召回... batch_size={args.batch_size}, time_window={args.time_window}, words_delta={args.words_delta}, top_k={args.top_k}, n_negatives={args.n_negatives}")
    recall_results = cold_start_recall_gpu(
        df_click, 
        df_article,
        df_query,
        batch_size=args.batch_size,
        time_window=args.time_window,
        words_delta=args.words_delta,
        top_k=args.top_k,
        n_workers=args.n_workers
    )
    
    # 增加多样性
    add_diversity_boost(recall_results, df_article, args.diversity_weight)
    
    # 转换为DataFrame格式
    log.info("整理召回结果...")
    recall_df = []
    for user_id, items in recall_results.items():
        target_item = df_query[df_query['user_id']==user_id]['click_article_id'].iloc[0]
        for item_id, score in items:
            recall_df.append({
                'user_id': int(user_id),
                'article_id': int(item_id),
                'sim_score': float(score),
                'label': 1 if item_id == target_item else 0 if target_item != -1 else np.nan
            })
    
    df_recall = pd.DataFrame(recall_df)
    log.debug(f'召回结果大小: {df_recall.shape}')
    
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
        
        log.info("\n======== 召回指标详情 ========")
        log.info(f"总用户数: {total_users}")
        log.info(f"命中用户数: {hit_users}")
        log.info(f"命中率: {hit_rate:.4f} ({hit_rate*100:.1f}%)")
        log.info(f"目标命中率: 67.9%")
        log.info(f"是否达标: {'✓' if hit_rate >= 0.679 else '✗'}")
        
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


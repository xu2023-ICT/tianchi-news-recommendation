import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from utils import Logger

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings('ignore')

seed = 2020

# 命令行参数
parser = argparse.ArgumentParser(description='排序特征')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_feature.log')
parser.add_argument('--test_size', type=int, default=1000, help='测试模式下的样本数')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
test_size = args.test_size

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'排序特征，mode: {mode}')


def func_if_sum(x):
    """计算用户历史交互物品与目标物品的相似度加权和"""
    user_id = x['user_id']
    article_id = x['article_id']

    # 添加错误处理
    if user_id not in user_item_dict:
        return 0
    
    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += item_sim[i][article_id] * (0.7**loc)
        except Exception as e:
            continue
    return sim_sum


def func_if_last(x):
    """计算用户最后一次交互物品与目标物品的相似度"""
    user_id = x['user_id']
    article_id = x['article_id']

    # 添加错误处理
    if user_id not in user_item_dict:
        return 0
        
    try:
        last_item = user_item_dict[user_id][-1]
        return item_sim[last_item][article_id]
    except Exception as e:
        return 0


def func_binetwork_sim_last(x):
    """计算用户最后一次交互物品与目标物品的二部网络相似度"""
    user_id = x['user_id']
    article_id = x['article_id']

    # 添加错误处理
    if user_id not in user_item_dict:
        return 0
        
    try:
        last_item = user_item_dict[user_id][-1]
        return binetwork_sim[last_item][article_id]
    except Exception as e:
        return 0


def consine_distance(vector1, vector2):
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray:
        return -1
    distance = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return distance


def func_w2w_sum(x, num):
    """计算用户最近num次交互物品与目标物品的word2vec相似度和"""
    user_id = x['user_id']
    article_id = x['article_id']

    # 添加错误处理
    if user_id not in user_item_dict:
        return 0
        
    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1][:num]

    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += consine_distance(article_vec_map[article_id],
                                      article_vec_map[i])
        except Exception as e:
            continue
    return sim_sum


def func_w2w_last_sim(x):
    """计算用户最后一次交互物品与目标物品的word2vec相似度"""
    user_id = x['user_id']
    article_id = x['article_id']

    # 添加错误处理
    if user_id not in user_item_dict:
        return 0
        
    try:
        last_item = user_item_dict[user_id][-1]
        return consine_distance(article_vec_map[article_id],
                              article_vec_map[last_item])
    except Exception as e:
        return 0


def func_hot_score(x):
    """计算文章热度分数"""
    article_id = x['article_id']
    timestamp = x['click_timestamp'] if 'click_timestamp' in x else x['created_at_ts']
    
    score = 0
    try:
        # 获取最近时间窗口的热度分数
        window_scores = article_hot_dict.get(article_id, {})
        if window_scores:
            # 找到最接近的时间窗口
            closest_time = min(window_scores.keys(), 
                             key=lambda t: abs(t - timestamp))
            score = window_scores[closest_time]
    except Exception as e:
        pass
    return score

def func_time_decay_hot(x, alpha=0.8, max_hours=720):  # 最大考虑30天
    """
    计算时间衰减的热度分数，添加溢出保护
    
    Args:
        x: 数据行
        alpha: 衰减系数 (0.8表示每小时衰减20%)
        max_hours: 最大考虑的小时数
    """
    try:
        score = x['article_hot_score']
        time_diff = x['created_at_ts'] - x['user_click_last_article_click_time']
        
        # 限制时间差范围
        hours_diff = min(max_hours, abs(time_diff) / 3600)
        
        # 对于负向时间差（文章比点击更早），使用不同的处理
        if time_diff < 0:
            decay = 1.0  # 或者使用较小的衰减
        else:
            decay = alpha ** hours_diff
            
        return float(score * decay)  # 确保返回float类型
    except Exception as e:
        return 0.0


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
    elif mode == 'test':
        # 测试模式：读取测试数据
        df_feature = pd.read_pickle('../user_data/data/test/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        
        # 只处理test recall中的用户数据
        test_users = df_feature['user_id'].unique()
        df_click = df_click[df_click['user_id'].isin(test_users)]
        
        log.info(f'测试模式：处理{len(test_users)}个用户')
        log.info(f'df_feature shape: {df_feature.shape}')
        log.info(f'df_click shape: {df_click.shape}')
    else:
        df_feature = pd.read_pickle('../user_data/data/online/recall.pkl')
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')

    # 文章特征
    log.debug(f'df_feature.shape: {df_feature.shape}')

    df_article = pd.read_csv('../data/articles.csv')
    df_article['created_at_ts'] = df_article['created_at_ts'] / 1000
    df_article['created_at_ts'] = df_article['created_at_ts'].astype('int')
    df_feature = df_feature.merge(df_article, how='left')
    df_feature['created_at_datetime'] = pd.to_datetime(
        df_feature['created_at_ts'], unit='s')

    log.debug(f'df_article.head(): {df_article.head()}')
    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 历史记录相关特征
    df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)
    df_click.rename(columns={'click_article_id': 'article_id'}, inplace=True)
    df_click = df_click.merge(df_article, how='left')

    df_click['click_timestamp'] = df_click['click_timestamp'] / 1000
    df_click['click_datetime'] = pd.to_datetime(df_click['click_timestamp'],
                                                unit='s',
                                                errors='coerce')
    df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour

    # 用户点击文章的创建时间差的平均值
    df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby(
        ['user_id'])['created_at_ts'].diff()
    df_temp = df_click.groupby([ 'user_id'])['user_id_click_article_created_at_ts_diff'].mean().reset_index()
    df_temp.columns = [ 'user_id', 'user_id_click_article_created_at_ts_diff_mean' ]
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 用户点击文章的时间差的平均值
    df_click['user_id_click_diff'] = df_click.groupby(
        ['user_id'])['click_timestamp'].diff()
    df_temp = df_click.groupby(['user_id'])['user_id_click_diff'].mean().reset_index()
    df_temp.columns = ['user_id', 'user_id_click_diff_mean']
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_click['click_timestamp_created_at_ts_diff'] = df_click[ 'click_timestamp'] - df_click['created_at_ts']

    # 点击文章的创建时间差的统计值
    df_temp = df_click.groupby(['user_id'])['click_timestamp_created_at_ts_diff'].agg(['mean','std']).reset_index()
    df_temp.columns = [ 'user_id', 'user_click_timestamp_created_at_ts_diff_mean', 'user_click_timestamp_created_at_ts_diff_std' ]
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_datetime_hour 统计值
    df_temp = df_click.groupby(['user_id'])['click_datetime_hour'].agg(
        user_click_datetime_hour_std='std'
    ).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 words_count 统计值
    df_temp = df_click.groupby(['user_id']).agg(
        user_clicked_article_words_count_mean=('words_count', 'mean'),
        user_click_last_article_words_count=('words_count', lambda x: x.iloc[-1])
    ).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 created_at_ts 统计值 
    df_temp = df_click.groupby('user_id').agg(
        user_click_last_article_created_time=('created_at_ts', lambda x: x.iloc[-1]),
        user_clicked_article_created_time_max=('created_at_ts', 'max')
    ).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 点击的新闻的 click_timestamp 统计值
    df_temp = df_click.groupby('user_id').agg(
        user_click_last_article_click_time=('click_timestamp', lambda x: x.iloc[-1]),
        user_clicked_article_click_time_mean=('click_timestamp', 'mean')
    ).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    df_feature['user_last_click_created_at_ts_diff'] = df_feature[ 'created_at_ts'] - df_feature['user_click_last_article_created_time']
    df_feature['user_last_click_timestamp_diff'] = df_feature[ 'created_at_ts'] - df_feature['user_click_last_article_click_time']
    df_feature['user_last_click_words_count_diff'] = df_feature[ 'words_count'] - df_feature['user_click_last_article_words_count']

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 计数统计
    for f in [['user_id'], ['article_id'], ['user_id', 'category_id']]:
        df_temp = df_click.groupby(f).size().reset_index()
        df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]

        df_feature = df_feature.merge(df_temp, how='left')

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    # 召回相关特征
    ## itemcf 相关
    user_item_ = df_click.groupby('user_id')['article_id'].agg(
        list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['article_id']))

    # 添加日志输出，帮助调试
    log.debug(f'用户数: {len(user_item_dict)}')
    log.debug(f'df_feature中的用户数: {df_feature["user_id"].nunique()}')
    log.debug(f'用户交集数: {len(set(user_item_dict.keys()) & set(df_feature["user_id"].unique()))}')

    if mode == 'valid':
        f = open('../user_data/sim/offline/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    elif mode == 'test':
        f = open('../user_data/sim/test/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/itemcf_sim.pkl', 'rb')
        item_sim = pickle.load(f)
        f.close()

    # 用户历史点击物品与待预测物品相似度
    df_feature['user_clicked_article_itemcf_sim_sum'] = df_feature.apply(func_if_sum, axis=1)
    df_feature['user_last_click_article_itemcf_sim'] = df_feature.apply(func_if_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## binetwork 相关
    if mode == 'valid':
        f = open('../user_data/sim/offline/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()
    elif mode == 'test':
        f = open('../user_data/sim/test/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/sim/online/binetwork_sim.pkl', 'rb')
        binetwork_sim = pickle.load(f)
        f.close()

    df_feature['user_last_click_article_binetwork_sim'] = df_feature.apply(func_binetwork_sim_last, axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    ## w2v 相关
    if mode == 'valid':
        f = open('../user_data/data/offline/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()
    elif mode == 'test':
        f = open('../user_data/data/test/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()
    else:
        f = open('../user_data/data/online/article_w2v.pkl', 'rb')
        article_vec_map = pickle.load(f)
        f.close()

    df_feature['user_last_click_article_w2v_sim'] = df_feature.apply(func_w2w_last_sim, axis=1)
    df_feature['user_click_article_w2w_sim_sum_2'] = df_feature.apply(lambda x: func_w2w_sum(x, 2), axis=1)

    log.debug(f'df_feature.shape: {df_feature.shape}')
    log.debug(f'df_feature.columns: {df_feature.columns.tolist()}')

    """
    # 加载热度召回的相似度文件
    if mode == 'valid':
        f = open('../user_data/data/offline/article_hot_dict.pkl', 'rb')
    elif mode == 'test':
        f = open('../user_data/data/test/article_hot_dict.pkl', 'rb')
    else:
        f = open('../user_data/data/online/article_hot_dict.pkl', 'rb')
    article_hot_dict = pickle.load(f)
    f.close()

    # 添加热度相关特征
    df_feature['article_hot_score'] = df_feature.apply(func_hot_score, axis=1)
    df_feature['article_time_decay_hot_score'] = df_feature.apply(func_time_decay_hot, axis=1)

    # 计算用户历史点击文章的平均热度
    df_temp = df_click.copy()
    df_temp['article_hot_score'] = df_temp.apply(func_hot_score, axis=1)
    df_temp = df_temp.groupby('user_id')['article_hot_score'].agg({
        'user_hist_articles_mean_hot': 'mean',
        'user_hist_articles_max_hot': 'max',
        'user_last_article_hot': lambda x: x.iloc[-1]
    }).reset_index()
    df_feature = df_feature.merge(df_temp, how='left')

    # 添加相对热度特征
    df_feature['article_hot_score_diff_mean'] = df_feature['article_hot_score'] - df_feature['user_hist_articles_mean_hot']
    df_feature['article_hot_score_diff_last'] = df_feature['article_hot_score'] - df_feature['user_last_article_hot']
    
    log.debug(f'添加热度特征后 df_feature.shape: {df_feature.shape}')
    log.debug(f'热度特征列: {[col for col in df_feature.columns if "hot" in col]}') 
    """

    # 保存特征文件
    if mode == 'valid':
        df_feature.to_pickle('../user_data/data/offline/feature.pkl')
    elif mode == 'test':
        os.makedirs('../user_data/data/test', exist_ok=True)
        df_feature.to_pickle('../user_data/data/test/feature.pkl')
    else:
        df_feature.to_pickle('../user_data/data/online/feature.pkl')
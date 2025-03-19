import argparse
import math
import os
import pickle
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from utils import Logger, evaluate

# 保持原有的参数解析代码
parser = argparse.ArgumentParser(description='binetwork 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')
args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'binetwork 召回，mode: {mode}')

def cal_sim(df):
    """计算物品相似度"""
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        list).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    item_user_ = df.groupby('click_article_id')['user_id'].agg(
        list).reset_index()
    item_user_dict = dict(
        zip(item_user_['click_article_id'], item_user_['user_id']))

    sim_dict = {}

    for item, users in tqdm(item_user_dict.items()):
        sim_dict.setdefault(item, {})

        for user in users:
            tmp_len = len(user_item_dict[user])
            for relate_item in user_item_dict[user]:
                sim_dict[item].setdefault(relate_item, 0)
                sim_dict[item][relate_item] += 1 / \
                    (math.log(len(users)+1) * math.log(tmp_len+1))

    return sim_dict, user_item_dict

def recall_for_user(args):
    """单个用户的召回处理函数"""
    user_id, item_id, item_sim, user_item_dict = args
    
    if user_id not in user_item_dict:
        return None
        
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1][:1]

    for item in interacted_items:
        for relate_item, wij in sorted(item_sim[item].items(),
                                       key=lambda d: d[1],
                                       reverse=True)[0:100]:
            if relate_item not in interacted_items:
                rank.setdefault(relate_item, 0)
                rank[relate_item] += wij

    sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
    
    if not sim_items:
        return None
        
    result_df = pd.DataFrame({
        'user_id': [user_id] * len(sim_items),
        'article_id': [x[0] for x in sim_items],
        'sim_score': [x[1] for x in sim_items],
        'label': [1 if x[0] == item_id else 0 for x in sim_items] if item_id != -1 
                else [np.nan] * len(sim_items)
    })
    
    return result_df

def process_user_batch(user_batch_df, item_sim, user_item_dict):
    """处理一批用户的召回"""
    args_list = [(row['user_id'], row['click_article_id'], item_sim, user_item_dict) 
                 for _, row in user_batch_df.iterrows()]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(recall_for_user, args_list), 
                          total=len(args_list)))
    
    # 过滤掉None结果并合并DataFrame
    results = [df for df in results if df is not None]
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

if __name__ == '__main__':
    # 数据加载
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        sim_pkl_file = '../user_data/sim/offline/binetwork_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
        sim_pkl_file = '../user_data/sim/online/binetwork_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    # 计算物品相似度
    item_sim, user_item_dict = cal_sim(df_click)
    
    # 保存相似度矩阵
    os.makedirs(os.path.dirname(sim_pkl_file), exist_ok=True)
    with open(sim_pkl_file, 'wb') as f:
        pickle.dump(item_sim, f)

    # 将用户分批处理
    batch_size = 100  # 可以根据实际情况调整批次大小
    user_batches = np.array_split(df_query, len(df_query) // batch_size + 1)
    
    # 处理每一批用户
    results = []
    for batch_df in tqdm(user_batches, desc="Processing user batches"):
        batch_result = process_user_batch(batch_df, item_sim, user_item_dict)
        if not batch_result.empty:
            results.append(batch_result)
    
    # 合并所有结果
    df_final = pd.concat(results, ignore_index=True)
    
    # 排序和保存结果
    df_final = df_final.sort_values(['user_id', 'sim_score'],
                                   ascending=[True, False]).reset_index(drop=True)
    
    # 计算召回指标
    if mode == 'valid':
        log.info('计算召回指标')
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        metrics = evaluate(df_final[df_final['label'].notnull()], total)
        log.debug(f'binetwork metrics: {metrics}')
    
    # 保存召回结果
    output_path = '../user_data/data/offline' if mode == 'valid' else '../user_data/data/online'
    output_file = f'{output_path}/recall_binetwork.pkl'
    os.makedirs(output_path, exist_ok=True)
    df_final.to_pickle(output_file)
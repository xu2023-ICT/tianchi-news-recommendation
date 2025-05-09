import argparse
import math
import os
import pickle
import random
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')
seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_w2v.log')
parser.add_argument('--test_size', type=int, default=1000, help='测试模式下的样本数')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
test_size = args.test_size

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'w2v 召回,mode: {mode}')

#使用 Word2Vec 算法学习用户行为(如点击物品)对应的词向量。

# 使用 Annoy 建立物品的索引,基于向量进行相似度计算。

# article_vec_map = word2vec(df_click, 'user_id', 'click_article_id', model_path)
def word2vec(df_, f1, f2, model_path):
    '''
    输入的 df_ 数据框按用户(f1)分组,获取每个用户的点击历史(f2)。

    通过 Word2Vec 算法训练词向量(通过设置 sg=1,使用 Skip-Gram 模型)。

    如果已有训练好的模型(w2v.m),则加载模型；否则训练新的模型并保存。
    '''
    df = df_.copy()
    # 按照用户id分组分组，然后只要click_article_id部分，将其命名为{}_{}_list'.format(f1, f2)，并转换为列表
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})
    # 获取所有用户的点击历史
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    # 删除'{}_{}_list'.format(f1, f2)列
    del tmp['{}_{}_list'.format(f1, f2)]

    # words是所有的交互历史，下面循环就是将sentences所有元素变成字符串，然后用一个words整体存起来
    words = []
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    # 存在就读取，不存在就创建
    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        model = Word2Vec(size=256,          # gensim 3.x 用 size
                         window=3,
                         min_count=1,
                         sg=1, # 使用skip-Gram模型（如果是0，则使用CBWO）
                         hs=0, #不使用层次softmax
                         seed=seed, 
                         negative=5, # 负采样的负样本数
                         workers=10 # 线程数
                         ) 
        model.build_vocab(sentences) # 构建词汇表
        model.train(sentences, total_examples=len(sentences), epochs=1)
        model.save(f'{model_path}/w2v.m')

    # 构建物品ID到词向量的映射
    article_vec_map = {}
    for word in set(words):
        if word in model.wv:
            article_vec_map[int(word)] = model.wv[word]

    return article_vec_map

# df_data = recall(df_query, article_vec_map, article_index, user_item_dict)
def recall(df_query, article_vec_map, article_index, user_item_dict):
    '''
    该函数通过用户的点击历史(user_item_dict)和物品的嵌入向量(article_vec_map)计算相似物品。

    对于每个用户，获取最近点击的物品，并计算与该物品相似的其他物品(最多返回 100 个)。

    计算相似度分数并将结果存储到 DataFrame 中。
    '''
    # 存储每个用户的推荐结果
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)

        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id] # 获取点击物品列表
        interacted_items = interacted_items[-1:] # 只取最近一次点击的物品

        # 遍历用户最近点击的物品
        for item in interacted_items:
            # 如果不在物品的词向量，就跳过
            if item not in article_vec_map:
                continue

            # 获取物品的词向量
            article_vec = article_vec_map[item]

            # 使用Annoy索引获取与该物品最相似的100个物品
            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 100, include_distances=True)
            # 计算相似度分数,这里使用了Annoy的计算公式
            sim_scores = [2 - distance for distance in distances]

            # 将相似物品添加到rank字典中
            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij # 累加相似度分数 ???这里为什么要累加，

            '''
            这里的rank确实毫无必要，因为只是取一个物品
            但是如果取三个物品，rank就有相加的必要了
            '''

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    return pd.concat(data_list, sort=False)


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/data/offline', exist_ok=True)
        os.makedirs('../user_data/model/offline', exist_ok=True)

        w2v_file = '../user_data/data/offline/article_w2v.pkl'
        model_path = '../user_data/model/offline'
    elif mode == 'test':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        test_users = df_query['user_id'].sample(n=test_size, random_state=2024)
        df_query = df_query[df_query['user_id'].isin(test_users)]
        df_click = df_click[df_click['user_id'].isin(test_users)]

        os.makedirs('../user_data/data/test', exist_ok=True)
        os.makedirs('../user_data/model/test', exist_ok=True)
        w2v_file = '../user_data/data/test/article_w2v.pkl'
        model_path = '../user_data/model/test'

        log.info(f'测试模式：选取{test_size}个用户')
        log.info(f'df_click shape: {df_click.shape}')
        log.info(f'df_query shape: {df_query.shape}')
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/data/online', exist_ok=True)
        os.makedirs('../user_data/model/online', exist_ok=True)

        w2v_file = '../user_data/data/online/article_w2v.pkl'
        model_path = '../user_data/model/online'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id', model_path)
    with open(w2v_file, 'wb') as f:
        pickle.dump(article_vec_map, f)

    article_index = AnnoyIndex(256, 'angular')
    article_index.set_seed(2020)

    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)

    article_index.build(100)

    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))

    df_data = recall(df_query, article_vec_map, article_index, user_item_dict)

    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True, False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    if mode == 'valid':
        log.info(f'计算召回指标')
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)
        log.debug(
            f'w2v: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_w2v.pkl')
    elif mode == 'test':
        df_data.to_pickle('../user_data/data/test/recall_w2v.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_w2v.pkl')

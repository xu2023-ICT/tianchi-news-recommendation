import argparse
import math
import os
import pickle
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_itemcf.log')
parser.add_argument('--test_size', type=int, default=1000, help='测试模式下的样本数')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
test_size = args.test_size

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'itemcf 召回, mode: {mode}')


def cal_sim(df_click):
    '''
    df_click是data处理的用户id点击的物品id
    计算了物品相似度矩阵sim_dict
    返回user_item_dict是用户和点击的文章的索引
    '''
    # user_item_同一id的点击文章click整理整一个list
    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    # 构建成一个字典记录用户点的文章id
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # 记录点击物品的数量
    item_cnt = defaultdict(int)
    # 记录相似度矩阵 相似度矩阵根据每个人浏览过的物品，累加出一个值，最后在除以根号下两个物品出现的次数(抑制热门物品)
    sim_dict = {}

    for user, items in tqdm(user_item_dict.items()):
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            #如果没有item这一项，就设置为{}，有了就不动
            sim_dict.setdefault(item, {})

            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                # 将sim_dict[item]这一项再嵌套一个字典
                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重
                # 考虑文章的正向顺序点击和反向顺序点击
                # 正向1 反向0.7
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))

                sim_dict[item][relate_item] += loc_weight  / math.log(1 + len(items))

    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            # 抑制热门物品
            sim_dict[item][relate_item] = cij / math.sqrt(item_cnt[item] * item_cnt[relate_item])

    return sim_dict, user_item_dict



def recall(df_query, item_sim, user_item_dict):
    '''
    df_query是部分用户id与最后一次点击的文章id索引
    item_sim是相似度矩阵
    user_item_dict是用户和点击文章的索引

    recall() 就是根据用户最近点的文章
    从相似度矩阵里找出最像的 top100 文章
    并打上是否命中的标签
    为下一步排序模块准备训练/测试数据。
    '''
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {}

        # 如果该用户因为去除最后一条点击行为而没有点击行为了
        if user_id not in user_item_dict:
            continue
        
        # interacted_items是用户遍历过的物品并取最新交互的两个
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:2]

        # 遍历用户最近点击的每个物品
        for loc, item in enumerate(interacted_items):
            # 对该物品的相似物品按照相似度排序，取前200个
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1], # 按照相似度d[1]排序
                                           reverse=True)[0:200]:
                # 如果该物品不是用户已经点击过的物品，才进行加分，避免推荐该用户已经点击的物品
                if relate_item not in interacted_items:
                    # 如果这个物品没有出现在rank中，初始化为0
                    rank.setdefault(relate_item, 0)
                    # 累加物品的推荐得分，权重衰减系数根据位置loc进行调整
                    rank[relate_item] += wij * (0.7**loc)

        # 得到的rank是根据用户最近点击的两个物品，然后根据相似度矩阵，推荐了两百个最相似的，并对点击的物品根据loc进行衰减，下面取前一百个物品

        # 将物品按照得分降序排序，取前一百个物品
        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]
        # 获取推荐物品的id和相似度分数
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        # 创建一个DataFrame用于保存当前用户推荐的结果
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        # 如果是线上测试模式（item_id == -1），标签为空（np.nan）
        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            # 线下验证时，给推荐物品打标签，将推荐的物品且用户真实点击的物品设置为1
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        # 保存需要的列，并确保数据类型正确
        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        # 将当前用户推荐结果添加到data_list
        data_list.append(df_temp)
    # 将所有用户的推荐结果合并成一个 DataFrame 返回
    return pd.concat(data_list, sort=False)

if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/itemcf_sim.pkl'
    elif mode == 'test':
        # 测试模式：读取部分数据
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        
        # 随机选择一部分用户
        test_users = df_query['user_id'].sample(n=test_size, random_state=2024)
        df_query = df_query[df_query['user_id'].isin(test_users)]
        df_click = df_click[df_click['user_id'].isin(test_users)]
        
        os.makedirs('../user_data/sim/test', exist_ok=True)
        sim_pkl_file = '../user_data/sim/test/itemcf_sim.pkl'
        
        log.info(f'测试模式：选取{test_size}个用户')
        log.info(f'df_click shape: {df_click.shape}')
        log.info(f'df_query shape: {df_query.shape}')
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/itemcf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()

    # 直接处理所有数据
    df_data = recall(df_query, item_sim, user_item_dict)

    # 对结果进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True, False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)
        log.debug(
            f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_itemcf.pkl')
    elif mode == 'test':
        os.makedirs('../user_data/data/test', exist_ok=True)
        df_data.to_pickle('../user_data/data/test/recall_itemcf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_itemcf.pkl')

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

parser = argparse.ArgumentParser(description='排序特征')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_feature.log')
parser.add_argument('--test_size', type=int, default=1000, help='测试模式下的样本数')

args = parser.parse_args()
mode = args.mode
logfile = args.logfile
test_size = args.test_size

os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'排序特征，mode: {mode}')


def func_if_sum(x):
    user_id = x['user_id']
    article_id = x['article_id']
    if user_id not in user_item_dict:
        return 0
    interacted_items = user_item_dict[user_id][::-1]
    sim_sum = 0
    for loc, i in enumerate(interacted_items):
        try:
            sim_sum += item_sim[i][article_id] * (0.7**loc)
        except:
            continue
    return sim_sum


def func_if_last(x):
    user_id = x['user_id']
    article_id = x['article_id']
    if user_id not in user_item_dict:
        return 0
    try:
        last_item = user_item_dict[user_id][-1]
        return item_sim[last_item][article_id]
    except:
        return 0


def func_binetwork_sim_last(x):
    user_id = x['user_id']
    article_id = x['article_id']
    if user_id not in user_item_dict:
        return 0
    try:
        last_item = user_item_dict[user_id][-1]
        return binetwork_sim[last_item][article_id]
    except:
        return 0


def consine_distance(vector1, vector2):
    if type(vector1) != np.ndarray or type(vector2) != np.ndarray:
        return -1
    distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return distance


def func_w2w_sum(x, num):
    user_id = x['user_id']
    article_id = x['article_id']
    if user_id not in user_item_dict:
        return 0
    interacted_items = user_item_dict[user_id][::-1][:num]
    sim_sum = 0
    for i in interacted_items:
        try:
            sim_sum += consine_distance(article_vec_map[article_id], article_vec_map[i])
        except:
            continue
    return sim_sum


def func_w2w_last_sim(x):
    user_id = x['user_id']
    article_id = x['article_id']
    if user_id not in user_item_dict:
        return 0
    try:
        last_item = user_item_dict[user_id][-1]
        return consine_distance(article_vec_map[article_id], article_vec_map[last_item])
    except:
        return 0


if mode == 'valid':
    df_feature = pd.read_pickle('../user_data/data/offline/recall.pkl')
    df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
elif mode == 'test':
    df_feature = pd.read_pickle('../user_data/data/test/recall.pkl')
    df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
    test_users = df_feature['user_id'].unique()
    df_click = df_click[df_click['user_id'].isin(test_users)]
else:
    df_feature = pd.read_pickle('../user_data/data/online/recall.pkl')
    df_click = pd.read_pickle('../user_data/data/online/click.pkl')

log.debug(f'df_feature.shape: {df_feature.shape}')

df_article = pd.read_csv('../data/articles.csv')
df_article['created_at_ts'] = df_article['created_at_ts'] // 1000
df_feature = df_feature.merge(df_article, how='left')
df_feature['created_at_datetime'] = pd.to_datetime(df_feature['created_at_ts'], unit='s')

df_click.sort_values(['user_id', 'click_timestamp'], inplace=True)
df_click.rename(columns={'click_article_id': 'article_id'}, inplace=True)
df_click = df_click.merge(df_article, how='left')
df_click['click_timestamp'] = df_click['click_timestamp'] // 1000
df_click['click_datetime'] = pd.to_datetime(df_click['click_timestamp'], unit='s', errors='coerce')
df_click['click_datetime_hour'] = df_click['click_datetime'].dt.hour

df_click['user_id_click_article_created_at_ts_diff'] = df_click.groupby(['user_id'])['created_at_ts'].diff()
df_temp = df_click.groupby(['user_id'])['user_id_click_article_created_at_ts_diff'].mean().reset_index()
df_temp.columns = ['user_id', 'user_id_click_article_created_at_ts_diff_mean']
df_feature = df_feature.merge(df_temp, how='left')

df_click['user_id_click_diff'] = df_click.groupby(['user_id'])['click_timestamp'].diff()
df_temp = df_click.groupby(['user_id'])['user_id_click_diff'].mean().reset_index()
df_temp.columns = ['user_id', 'user_id_click_diff_mean']
df_feature = df_feature.merge(df_temp, how='left')

df_click['click_timestamp_created_at_ts_diff'] = df_click['click_timestamp'] - df_click['created_at_ts']
df_temp = df_click.groupby(['user_id'])['click_timestamp_created_at_ts_diff'].agg(['mean', 'std']).reset_index()
df_temp.columns = ['user_id', 'user_click_timestamp_created_at_ts_diff_mean', 'user_click_timestamp_created_at_ts_diff_std']
df_feature = df_feature.merge(df_temp, how='left')

df_temp = df_click.groupby(['user_id'])['click_datetime_hour'].std().reset_index()
df_temp.columns = ['user_id', 'user_click_datetime_hour_std']
df_feature = df_feature.merge(df_temp, how='left')

# 修改点：words_count统计项
df_temp1 = df_click.groupby('user_id')['words_count'].mean().reset_index()
df_temp1.columns = ['user_id', 'user_clicked_article_words_count_mean']
df_temp2 = df_click.groupby('user_id')['words_count'].apply(lambda x: x.iloc[-1]).reset_index()
df_temp2.columns = ['user_id', 'user_click_last_article_words_count']
df_temp = pd.merge(df_temp1, df_temp2, on='user_id', how='left')
df_feature = df_feature.merge(df_temp, on='user_id', how='left')

# 修改点：created_at_ts统计项
df_temp1 = df_click.groupby('user_id')['created_at_ts'].apply(lambda x: x.iloc[-1]).reset_index()
df_temp1.columns = ['user_id', 'user_click_last_article_created_time']
df_temp2 = df_click.groupby('user_id')['created_at_ts'].max().reset_index()
df_temp2.columns = ['user_id', 'user_clicked_article_created_time_max']
df_temp = pd.merge(df_temp1, df_temp2, on='user_id', how='left')
df_feature = df_feature.merge(df_temp, on='user_id', how='left')

# 修改点：click_timestamp统计项
df_temp1 = df_click.groupby('user_id')['click_timestamp'].apply(lambda x: x.iloc[-1]).reset_index()
df_temp1.columns = ['user_id', 'user_click_last_article_click_time']
df_temp2 = df_click.groupby('user_id')['click_timestamp'].mean().reset_index()
df_temp2.columns = ['user_id', 'user_clicked_article_click_time_mean']
df_temp = pd.merge(df_temp1, df_temp2, on='user_id', how='left')
df_feature = df_feature.merge(df_temp, on='user_id', how='left')

df_feature['user_last_click_created_at_ts_diff'] = df_feature['created_at_ts'] - df_feature['user_click_last_article_created_time']
df_feature['user_last_click_timestamp_diff'] = df_feature['created_at_ts'] - df_feature['user_click_last_article_click_time']
df_feature['user_last_click_words_count_diff'] = df_feature['words_count'] - df_feature['user_click_last_article_words_count']

for f in [['user_id'], ['article_id'], ['user_id', 'category_id']]:
    df_temp = df_click.groupby(f).size().reset_index()
    df_temp.columns = f + ['{}_cnt'.format('_'.join(f))]
    df_feature = df_feature.merge(df_temp, how='left')

user_item_ = df_click.groupby('user_id')['article_id'].agg(list).reset_index()
user_item_dict = dict(zip(user_item_['user_id'], user_item_['article_id']))

if mode == 'valid':
    item_sim = pickle.load(open('../user_data/sim/offline/itemcf_sim.pkl', 'rb'))
elif mode == 'test':
    item_sim = pickle.load(open('../user_data/sim/test/itemcf_sim.pkl', 'rb'))
else:
    item_sim = pickle.load(open('../user_data/sim/online/itemcf_sim.pkl', 'rb'))

df_feature['user_clicked_article_itemcf_sim_sum'] = df_feature.apply(func_if_sum, axis=1)
df_feature['user_last_click_article_itemcf_sim'] = df_feature.apply(func_if_last, axis=1)

if mode == 'valid':
    binetwork_sim = pickle.load(open('../user_data/sim/offline/binetwork_sim.pkl', 'rb'))
elif mode == 'test':
    binetwork_sim = pickle.load(open('../user_data/sim/test/binetwork_sim.pkl', 'rb'))
else:
    binetwork_sim = pickle.load(open('../user_data/sim/online/binetwork_sim.pkl', 'rb'))

df_feature['user_last_click_article_binetwork_sim'] = df_feature.apply(func_binetwork_sim_last, axis=1)

if mode == 'valid':
    article_vec_map = pickle.load(open('../user_data/data/offline/article_w2v.pkl', 'rb'))
elif mode == 'test':
    article_vec_map = pickle.load(open('../user_data/data/test/article_w2v.pkl', 'rb'))
else:
    article_vec_map = pickle.load(open('../user_data/data/online/article_w2v.pkl', 'rb'))

df_feature['user_last_click_article_w2v_sim'] = df_feature.apply(func_w2w_last_sim, axis=1)
df_feature['user_click_article_w2w_sim_sum_2'] = df_feature.apply(lambda x: func_w2w_sum(x, 2), axis=1)

if mode == 'valid':
    df_feature.to_pickle('../user_data/data/offline/feature.pkl')
elif mode == 'test':
    os.makedirs('../user_data/data/test', exist_ok=True)
    df_feature.to_pickle('../user_data/data/test/feature.pkl')
else:
    df_feature.to_pickle('../user_data/data/online/feature.pkl')

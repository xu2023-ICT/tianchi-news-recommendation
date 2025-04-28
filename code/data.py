import argparse
import os
import random
from random import sample

import pandas as pd
from tqdm import tqdm

from utils import Logger

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='数据处理')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'数据处理，mode: {mode}')


def data_offline(df_train_click, df_test_click):
    '''
    将df_train_click中五万个用户最后一条id作为测试集
    返回df_query是最后一条点击
    返回df_click除五万个用户之外的所有用户点击,并且按照用户id和时间戳排序返回
    '''
    train_users = df_train_click['user_id'].values.tolist() #先转numpy 再转list
    val_users = sample(train_users, 50000) #随机抽取五万用户作为验证集，每个用户取最后一条文章id
    log.debug(f'val_users num: {len(set(val_users))}') 

    # 训练集用户 抽出行为数据最后一条作为线下验证集 
    click_list = [] #由两部分组成，一部分是train_click（train_click是在val_user中除了最后一条之外所有的交互，另一部分是不在val_uesrs
    valid_query_list = [] #在val_users中，只包含用户的最后一条交互信息

    groups = df_train_click.groupby(['user_id'])
    for user_id, group in tqdm(groups):
        if user_id in val_users:
            valid_query = group.tail(1)
            valid_query_list.append(
                valid_query[['user_id', 'click_article_id']]) #取每个用户id中最后一个点击的文章id

            train_click = group.head(group.shape[0] - 1)# 取出用户最后一条之外的所有信息
            click_list.append(train_click) # 
        else:
            click_list.append(group)

    # 将click_list valid_query_list重新拼接
    df_train_click = pd.concat(click_list, sort=False)
    df_valid_query = pd.concat(valid_query_list, sort=False)


    test_users = df_test_click['user_id'].unique() 
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])
    #测试集用户也是只取最后一条记录
    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])
    
    df_query = pd.concat([df_valid_query, df_test_query],
                         sort=False).reset_index(drop=True)
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../user_data/data/offline', exist_ok=True)

    df_click.to_pickle('../user_data/data/offline/click.pkl')
    df_query.to_pickle('../user_data/data/offline/query.pkl')


#用于上传数据时所需要的数据处理
def data_online(df_train_click, df_test_click):
    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    # 这里添加表示不知道用户最后会点击什么，-1表示不知道
    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = df_test_query
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../data/online', exist_ok=True)

    df_click.to_pickle('../user_data/data/online/click.pkl')
    df_query.to_pickle('../user_data/data/online/query.pkl')


if __name__ == '__main__':
    # 输出当前路径
    df_train_click = pd.read_csv('../data/train_click_log.csv')
    # df_train_click_b = pd.read_csv('data/testB_click_log.csv')
    # df_train_click = pd.concat([df_train_click, df_train_click_b], axis=0)
    df_test_click = pd.read_csv('../data/testA_click_log.csv')

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)

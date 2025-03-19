# main.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import lightgbm as lgb
from collections import defaultdict
from tqdm import tqdm

class NewsRecommender:
    def __init__(self, mode='valid'):
        self.mode = mode
        self.recall_methods = ['itemcf', 'w2v', 'binetwork'] 
        self.recall_weights = {'itemcf': 1, 'binetwork': 1, 'w2v': 0.1}
        
    def load_data(self):
        """加载基础数据"""
        if self.mode == 'valid':
            base_path = '../user_data/data/offline/'
        else:
            base_path = '../user_data/data/online/'
            
        self.df_click = pd.read_pickle(f'{base_path}click.pkl')
        self.df_query = pd.read_pickle(f'{base_path}query.pkl')
        
    def process_user_features(self):
        """处理用户特征"""
        # 用户点击历史统计特征
        user_history = self.df_click.groupby('user_id').agg({
            'click_article_id': ['count', 'nunique'],
            'click_timestamp': ['min', 'max']
        }).reset_index()
        
        user_history.columns = ['user_id', 'total_clicks', 'unique_clicks', 
                              'first_click_time', 'last_click_time']
        
        # 用户活跃时间特征
        user_history['user_active_days'] = (user_history['last_click_time'] - 
                                          user_history['first_click_time']) / (24 * 3600)
                                          
        # 用户平均点击频率
        user_history['avg_clicks_per_day'] = user_history['total_clicks'] / \
                                            user_history['user_active_days'].clip(1)
                                            
        return user_history
        
    def process_item_features(self):
        """处理物品特征"""
        # 文章基础统计特征
        item_info = self.df_click.groupby('click_article_id').agg({
            'user_id': ['count', 'nunique'],
            'click_timestamp': ['min', 'max']  
        }).reset_index()
        
        item_info.columns = ['article_id', 'total_clicks', 'unique_users',
                           'first_click_time', 'last_click_time']
        
        # 文章热度衰减
        now_time = self.df_click['click_timestamp'].max()
        item_info['item_popularity_decay'] = np.exp(-(now_time - item_info['last_click_time']) / (24 * 3600))
        
        return item_info
        
    def merge_recall_results(self):
        """合并多路召回结果"""
        recall_list = []
        
        for method in self.recall_methods:
            recall_result = pd.read_pickle(f'../user_data/data/{"offline" if self.mode=="valid" else "online"}/recall_{method}.pkl')
            
            # 对每路召回结果进行归一化
            user_scores = recall_result.groupby('user_id')['sim_score']
            recall_result['sim_score'] = user_scores.transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
            
            # 加权
            recall_result['sim_score'] *= self.recall_weights[method]
            recall_list.append(recall_result)
            
        # 合并所有召回结果
        recall_final = pd.concat(recall_list, sort=False)
        recall_final = recall_final.groupby(['user_id', 'article_id']).agg({
            'sim_score': 'sum',
            'label': 'first'
        }).reset_index()
        
        return recall_final
        
    def generate_features(self, recall_df):
        """特征工程"""
        # 合并用户特征
        user_features = self.process_user_features()
        recall_df = recall_df.merge(user_features, on='user_id', how='left')
        
        # 合并物品特征
        item_features = self.process_item_features()
        recall_df = recall_df.merge(item_features, on='article_id', how='left')
        
        # 交叉特征
        recall_df['user_item_click_ratio'] = recall_df['user_total_clicks'] / \
                                            recall_df['item_total_clicks']
                                            
        # 时间特征
        recall_df['user_item_time_diff'] = recall_df['last_click_time_y'] - \
                                          recall_df['last_click_time_x']
                                          
        return recall_df
        
    def train_ranker(self, train_data):
        """训练排序模型"""
        feature_cols = ['sim_score', 'total_clicks_x', 'unique_clicks',
                       'user_active_days', 'avg_clicks_per_day',
                       'total_clicks_y', 'unique_users', 'item_popularity_decay',
                       'user_item_click_ratio', 'user_item_time_diff']
                       
        ranker = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            num_leaves=32,
            learning_rate=0.1,
            n_estimators=100,
            importance_type='gain'
        )
        
        # 准备排序数据
        group_sizes = train_data.groupby('user_id').size().values
        
        ranker.fit(
            train_data[feature_cols],
            train_data['label'],
            group=group_sizes,
            verbose=50
        )
        
        return ranker, feature_cols
        
    def run(self):
        """运行完整的推荐流程"""
        # 1. 加载数据
        self.load_data()
        
        # 2. 合并召回结果
        recall_df = self.merge_recall_results()
        
        # 3. 特征工程
        recall_df = self.generate_features(recall_df)
        
        # 4. 训练排序模型
        if self.mode == 'valid':
            train_data = recall_df[recall_df['label'].notnull()]
            ranker, feature_cols = self.train_ranker(train_data)
            
            # 预测并评估
            recall_df['pred_score'] = ranker.predict(recall_df[feature_cols])
            evaluate_result = evaluate(recall_df[recall_df['label'].notnull()],
                                    self.df_query[self.df_query['click_article_id'] != -1].user_id.nunique())
            print("Evaluation results:", evaluate_result)
            
        else:
            # 线上预测
            recall_df['pred_score'] = ranker.predict(recall_df[feature_cols])
            recall_df = recall_df.sort_values(['user_id', 'pred_score'], ascending=[True, False])
            
            # 生成提交结果
            submission = gen_sub(recall_df)
            submission.to_csv('submission.csv', index=False)
            
if __name__ == "__main__":
    # 验证集训练
    recommender = NewsRecommender(mode='valid')
    recommender.run()
    
    # 测试集预测
    recommender = NewsRecommender(mode='online')
    recommender.run()
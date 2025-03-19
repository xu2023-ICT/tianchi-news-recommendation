import argparse
import math
import os
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from utils import Logger, evaluate
parser = argparse.ArgumentParser(description='Hybrid CPU-GPU binetwork recall')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test.log')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--test_size', type=int, default=1000, help='测试模式下的样本数')
args = parser.parse_args()

mode = args.mode
logfile = args.logfile
device = torch.device(args.device)

# Initialize logger
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'Hybrid CPU-GPU binetwork recall, mode: {mode}, device: {device}')

def prepare_user_item_data(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    """
    Prepare user-item interaction data structures for efficient processing
    """
    # Create user and item mappings
    user_item_dict = df.groupby('user_id')['click_article_id'].agg(list).to_dict()
    item_user_dict = df.groupby('click_article_id')['user_id'].agg(list).to_dict()
    
    # Calculate user activity lengths for normalization
    user_activity_len = {
        user: len(items) for user, items in user_item_dict.items()
    }
    
    return user_item_dict, item_user_dict, user_activity_len

def calculate_item_similarity_batch(
    target_items: List[int],
    item_user_dict: Dict[int, List[int]],
    user_item_dict: Dict[int, List[int]],
    user_activity_len: Dict[int, int],
    batch_size: int = 512
) -> Dict[int, Dict[int, float]]:
    """
    Calculate similarity for a batch of target items using GPU acceleration
    """
    sim_dict = defaultdict(dict)
    
    # Process target items in smaller batches
    for i in range(0, len(target_items), batch_size):
        batch_items = target_items[i:i + batch_size]
        
        # Collect all users who interacted with the batch items
        batch_users = set()
        for item in batch_items:
            if item in item_user_dict:
                batch_users.update(item_user_dict[item])
        
        if not batch_users:
            continue
            
        # Create tensors for batch processing
        user_list = list(batch_users)
        user_to_idx = {user: idx for idx, user in enumerate(user_list)}
        
        # Create interaction matrix for batch items
        batch_interactions = torch.zeros(
            (len(batch_items), len(user_list)), 
            device=device
        )
        
        # Fill interaction matrix
        for batch_idx, item in enumerate(batch_items):
            if item in item_user_dict:
                for user in item_user_dict[item]:
                    batch_interactions[batch_idx, user_to_idx[user]] = 1.0
        
        # Calculate normalization factors
        user_weights = torch.tensor(
            [1.0 / math.log(user_activity_len[user] + 1) for user in user_list],
            device=device
        )
        item_weights = torch.tensor(
            [1.0 / math.log(len(item_user_dict.get(item, [])) + 1) for item in batch_items],
            device=device
        ).reshape(-1, 1)
        
        # Calculate similarities for all items that these users interacted with
        all_related_items = set()
        for user in batch_users:
            all_related_items.update(user_item_dict[user])
        
        # Process related items in chunks to save memory
        chunk_size = batch_size * 2
        related_items_list = list(all_related_items)
        
        for j in range(0, len(related_items_list), chunk_size):
            chunk_items = related_items_list[j:j + chunk_size]
            
            # Create interaction matrix for chunk items
            chunk_interactions = torch.zeros(
                (len(chunk_items), len(user_list)), 
                device=device
            )
            
            # Fill chunk interaction matrix
            for chunk_idx, rel_item in enumerate(chunk_items):
                if rel_item in item_user_dict:
                    for user in item_user_dict[rel_item]:
                        if user in user_to_idx:
                            chunk_interactions[chunk_idx, user_to_idx[user]] = 1.0
            
            # Calculate similarity scores
            sim_scores = torch.mm(batch_interactions, chunk_interactions.t())
            sim_scores = sim_scores * item_weights
            sim_scores = sim_scores * torch.tensor(
                [1.0 / math.log(len(item_user_dict.get(item, [])) + 1) for item in chunk_items],
                device=device
            )
            
            # Store non-zero similarities
            sim_scores = sim_scores.cpu().numpy()
            for batch_idx, item in enumerate(batch_items):
                for chunk_idx, rel_item in enumerate(chunk_items):
                    score = sim_scores[batch_idx, chunk_idx]
                    if score > 0:
                        sim_dict[item][rel_item] = score
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    return sim_dict

def recall_items(
    user_id: int,
    target_item: int,
    user_item_dict: Dict[int, List[int]],
    sim_dict: Dict[int, Dict[int, float]],
    top_k: int = 50
) -> List[Dict]:
    """
    Generate recall items for a single user
    """
    if user_id not in user_item_dict:
        return []
    
    user_items = user_item_dict[user_id]
    if not user_items:
        return []
    
    # Get most recent item
    recent_item = user_items[-1]
    if recent_item not in sim_dict:
        return []
    
    # Get similar items
    similar_items = sim_dict[recent_item]
    similar_items = {
        item: score 
        for item, score in similar_items.items() 
        if item not in user_items
    }
    
    # Sort and get top items
    top_items = sorted(
        similar_items.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    return [
        {
            'user_id': user_id,
            'article_id': item_id,
            'sim_score': score,
            'label': 1 if item_id == target_item else 0 if target_item != -1 else np.nan
        }
        for item_id, score in top_items
    ]

if __name__ == '__main__':
    # Load data
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        sim_pkl_file = '../user_data/sim/offline/binetwork_sim.pkl'
    elif mode == 'test':
        # 测试模式：读取部分数据
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        
        # 随机选择一部分用户
        test_users = df_query['user_id'].sample(n=args.test_size, random_state=2024)
        df_query = df_query[df_query['user_id'].isin(test_users)]
        df_click = df_click[df_click['user_id'].isin(test_users)]
        
        # 设置测试模式的路径
        os.makedirs('../user_data/sim/test', exist_ok=True)
        os.makedirs('../user_data/data/test', exist_ok=True)
        sim_pkl_file = '../user_data/sim/test/binetwork_sim.pkl'
        
        log.info(f'测试模式：选取{args.test_size}个用户')
        log.info(f'df_click shape: {df_click.shape}')
        log.info(f'df_query shape: {df_query.shape}')
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
        sim_pkl_file = '../user_data/sim/online/binetwork_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    
    # Prepare data structures
    user_item_dict, item_user_dict, user_activity_len = prepare_user_item_data(df_click)
    
    # Calculate similarities using batched processing
    unique_items = list(item_user_dict.keys())
    log.info(f'Processing {len(unique_items)} unique items')
    
    sim_dict = {}
    for i in tqdm(range(0, len(unique_items), args.batch_size), desc="Computing similarities"):
        batch_items = unique_items[i:i + args.batch_size]
        batch_sim = calculate_item_similarity_batch(
            batch_items,
            item_user_dict,
            user_item_dict,
            user_activity_len,
            args.batch_size
        )
        sim_dict.update(batch_sim)
        
        # Optional: Save intermediate results
        if (i + args.batch_size) % (args.batch_size * 10) == 0:
            temp_file = sim_pkl_file + f'.temp_{i}'
            with open(temp_file, 'wb') as f:
                pickle.dump(sim_dict, f)
    
    # Save final similarity dictionary
    os.makedirs(os.path.dirname(sim_pkl_file), exist_ok=True)
    with open(sim_pkl_file, 'wb') as f:
        pickle.dump(sim_dict, f)
    
    # Generate recommendations for all users
    results = []
    for _, row in tqdm(df_query.iterrows(), total=len(df_query), desc="Generating recommendations"):
        user_results = recall_items(
            row['user_id'],
            row['click_article_id'],
            user_item_dict,
            sim_dict
        )
        results.extend(user_results)
    
    # Create final DataFrame
    df_final = pd.DataFrame(results)
    
    if not df_final.empty:
        df_final = df_final.sort_values(
            ['user_id', 'sim_score'],
            ascending=[True, False]
        ).reset_index(drop=True)
        
        # Calculate metrics if in validation mode
        if mode == 'valid':
            log.info('Calculating recall metrics')
            total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
            metrics = evaluate(df_final[df_final['label'].notnull()], total)
            log.debug(f'Hybrid CPU-GPU binetwork metrics: {metrics}')
    
        # Save results
        if mode == 'valid':
            output_file = '../user_data/data/offline/recall_binetwork.pkl'
        elif mode == 'test':
            output_file = '../user_data/data/test/recall_binetwork.pkl'
        else:
            output_file = '../user_data/data/online/recall_binetwork.pkl'
            
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_final.to_pickle(output_file)
        log.info(f'Results saved to {output_file}')
    else:
        log.warning("No recommendations generated!")
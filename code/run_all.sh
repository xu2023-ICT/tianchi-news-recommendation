#!/bin/bash

# 设置环境（可选，比如切换 conda 环境）
# source activate tianchi

timestamp=$(date +"%Y%m%d_%H%M%S")

echo "开始运行各个模块..."

python data.py --mode valid --logfile log_data_$timestamp.log
python recall_itemcf.py --mode valid --logfile log_itemcf_$timestamp.log
python recall_binetworkGpu.py --mode valid --logfile log_binetwork_$timestamp.log
python recall_w2v.py --mode valid --logfile log_w2v_$timestamp.log
python recall.py --mode valid --logfile log_recall_merge_$timestamp.log
python rank_feature_fixed.py --mode valid --logfile log_feature_$timestamp.log
python rank_lgb.py --mode valid --logfile log_lgb_$timestamp.log

echo "全部运行完成！"

# 处理数据,这块用py3.7以上版本好像有点问题，建议用py3.7
python data.py --mode valid --logfile 

# itemcf 召回
python recall_itemcf.py --mode valid --logfile 

# binetwork 召回
python recall_binetworkGpu.py --mode valid --logfile 

# w2v 召回
python recall_w2v.py --mode valid --logfile 

# 召回合并
python recall.py --mode valid --logfile 

# 排序特征
python rank_feature.py --mode valid --logfile 

# lgb 模型训练
python rank_lgb.py --mode valid --logfile 

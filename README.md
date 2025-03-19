零基础入门推荐系统 - 新闻推荐 Top2  

比赛地址: https://tianchi.aliyun.com/competition/entrance/531842/introduction

# 解决方案
采用3种召回方式：itemcf 召回，binetwork 召回和基于 word2vec 的 i2i 召回。合并去重并删除没有召回到真实商品的用户数据后，利用特征工程+ LGB 二分类模型进行排序。

# 复现步骤
操作系统：ubuntu 16.04  
```
pip install requirements.txt
cd code
bash test.sh
```

# update
将test.sh的代码依次运行即可，注意，我在运行data.py时必须把python版本改成3.7，如果跑不了记得conda整个3.7的环境

电脑性能还可以的可以跑gpu版本，binework的会快不少，4060ti可以20分钟内跑完，整体比较费时间的就是rank和这个

最后的排序由于之前老是一步报错就得重新来，直接改成了notebook，自己修改的时候也可以跟着来。

然后本来还想试试冷启动和热度召回的，但是效果很不好，就没继续做了。

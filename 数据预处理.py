import pandas as pd

# 读取数据
df_1_15 = pd.read_csv("df_1_15.csv")
df_16_30 = pd.read_csv("df_16_30.csv")

# 读取单机app.csv文件，提取其中的单机app的app_id
df_single_app = pd.read_csv("常用20类app中的单机app.csv")
single_app_ids = df_single_app["app_id"].unique()

#持续时间大于等于10并且满足至少一个上传流量大于0、下载流量大于0、或者 app_id 在 single_app_ids 列表中的行会被保留，
#即如果持续时间小于10，或者持续时间大于等于10，但是上行流量和下行流量都是0，那么就认为用户只是点了一下进去，没有使用
#但是如果是单机app的话，这样的数据是合理的
df_1_15 = df_1_15[(df_1_15['duration'] >= 10) & ((df_1_15['up_flow'] > 0) | (df_1_15['down_flow'] > 0) | (df_1_15['app_id'].isin(single_app_ids)))]
df_16_30 = df_16_30[(df_16_30['duration'] >= 10) & ((df_16_30['up_flow'] > 0) | (df_16_30['down_flow'] > 0) | (df_16_30['app_id'].isin(single_app_ids)))]

# 重置索引并删除不需要的索引列
df_1_15.reset_index(drop=True, inplace=True)
df_16_30.reset_index(drop=True, inplace=True)

# 对齐用户，去掉只在一个数据集中出现的用户
users = set(df_1_15['user_id']).intersection(set(df_16_30['user_id']))
df_1_15 = df_1_15[df_1_15['user_id'].isin(users)]
df_16_30 = df_16_30[df_16_30['user_id'].isin(users)]

#将user_id设置为行名
#df_1_15.set_index('user_id', inplace=True)
#df_16_30.set_index('user_id', inplace=True)

df_1_15.to_csv('df_1_15.csv', index=False) # 保存1~15天的数据
df_16_30.to_csv('df_16_30.csv', index=False) # 保存16~30天的数据

print(df_1_15)
print(df_16_30)
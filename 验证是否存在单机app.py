import pandas as pd

# 读取数据
df_1_15 = pd.read_csv("df_1_15.csv")
df_16_30 = pd.read_csv("df_16_30.csv")

# 删除不需要的列
df_1_15.drop(['start_datetime', 'end_datetime', 'user_id', 'app_type'], axis=1, inplace=True)
df_16_30.drop(['start_datetime', 'end_datetime', 'user_id', 'app_type'], axis=1, inplace=True)

# 合并df_1_11和df_12_21，并根据app_id进行分组并求和
df_merged = pd.concat([df_1_15, df_16_30]).groupby('app_id').sum()

# 输出合并后的结果
print(df_merged)

# 统计down_flow和up_flow同时为0的行数
num_rows_with_zero_flow = len(df_merged[(df_merged['down_flow'] == 0) & (df_merged['up_flow'] == 0)])

# 输出down_flow和up_flow同时为0的app_id
app_ids_with_zero_flow = df_merged[(df_merged['down_flow'] == 0) & (df_merged['up_flow'] == 0)].index.tolist()

# 将down_flow和up_flow同时为0的app_id保存成数据框并以'单机app.csv'的文件名保存
df_app_ids_with_zero_flow = pd.DataFrame(app_ids_with_zero_flow, columns=['app_id'])
df_app_ids_with_zero_flow.to_csv('常用20类app中的单机app.csv', index=False)

# 打印结果
print("常用20类app中down_flow和up_flow同时为0的行数:", num_rows_with_zero_flow)
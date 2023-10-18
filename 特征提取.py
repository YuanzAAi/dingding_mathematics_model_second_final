import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 读取数据
df_1_15 = pd.read_csv("df_1_15.csv")
df_16_30 = pd.read_csv("df_16_30.csv")

# 使用对数变换进行处理，使其分布更加平滑
df_1_15["duration"] = np.log1p(df_1_15["duration"])
df_1_15["up_flow"] = np.log1p(df_1_15["up_flow"])
df_1_15["down_flow"] = np.log1p(df_1_15["down_flow"])

df_16_30["duration"] = np.log1p(df_16_30["duration"])
df_16_30["up_flow"] = np.log1p(df_16_30["up_flow"])
df_16_30["down_flow"] = np.log1p(df_16_30["down_flow"])


# 使用归一化进行处理，使其值在0到1之间
scaler = MinMaxScaler()
df_1_15["duration"] = scaler.fit_transform(df_1_15[["duration"]].values.reshape(-1, 1))
df_1_15["up_flow"] = scaler.fit_transform(df_1_15["up_flow"].values.reshape(-1, 1))
df_1_15["down_flow"] = scaler.fit_transform(df_1_15["down_flow"].values.reshape(-1, 1))

df_16_30["duration"] = scaler.fit_transform(df_16_30[["duration"]].values.reshape(-1, 1))
df_16_30["up_flow"] = scaler.fit_transform(df_16_30["up_flow"].values.reshape(-1, 1))
df_16_30["down_flow"] = scaler.fit_transform(df_16_30["down_flow"].values.reshape(-1, 1))

# 定义一个函数，将日期类型数据分解为正弦和余弦分量
def date_to_sin_cos(df, col):
    # 将日期类型数据转换为datetime格式
    df[col] = pd.to_datetime(df[col])
    # 提取日、时等变量
    df['day'] = df[col].dt.day
    df['hour'] = df[col].dt.hour
    # 提取秒和分钟变量
    df['second'] = df[col].dt.second
    df['minute'] = df[col].dt.minute
    # 计算每个变量对应的周期长度
    day_cycle = 2 * np.pi / 24
    hour_cycle = 2 * np.pi / 60
    # 计算每个变量对应的周期长度
    second_cycle = 2 * np.pi / 60
    minute_cycle = 2 * np.pi / 60
    # 计算每个变量对应的正弦和余弦分量
    df['day_sin'] = np.sin(day_cycle * df['day'])
    df['day_cos'] = np.cos(day_cycle * df['day'])
    df['hour_sin'] = np.sin(hour_cycle * df['hour'])
    df['hour_cos'] = np.cos(hour_cycle * df['hour'])
    # 计算每个变量对应的正弦和余弦分量
    df['second_sin'] = np.sin(second_cycle * df['second'])
    df['second_cos'] = np.cos(second_cycle * df['second'])
    df['minute_sin'] = np.sin(minute_cycle * df['minute'])
    df['minute_cos'] = np.cos(minute_cycle * df['minute'])
    # 删除原始的日期类型数据列
    df.drop(col, axis=1, inplace=True)
    return df

# 对df_1_15和df_16_30中的start_datetime和end_datetime进行处理
df_1_15 = date_to_sin_cos(df_1_15, 'start_datetime')
df_1_15 = date_to_sin_cos(df_1_15, 'end_datetime')
df_16_30 = date_to_sin_cos(df_16_30, 'start_datetime')
df_16_30 = date_to_sin_cos(df_16_30, 'end_datetime')

# 读取app_class.csv文件
app_class = pd.read_csv('app_class.csv')

# 将数据集中的app_id替换为对应的app_class
df_1_15 = df_1_15.merge(app_class, on='app_id', how='left')
df_1_15.drop('app_id', axis=1, inplace=True)
df_1_15.rename(columns={'app_class': 'app_id'}, inplace=True)

df_16_30 = df_16_30.merge(app_class, on='app_id', how='left')
df_16_30.drop('app_id', axis=1, inplace=True)
df_16_30.rename(columns={'app_class': 'app_id'}, inplace=True)

# 对英文字母a-t进行编码做成分类标签
le = LabelEncoder()
le.fit(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'])
df_1_15['app_id'] = le.transform(df_1_15['app_id'])
df_16_30['app_id'] = le.transform(df_16_30['app_id'])


group_cols = ['user_id', 'app_id', 'app_type']


# 按照用户id和app_id分组，进行特征提取的函数
def feature_extract(df, group_cols):
    # 根据分组列进行分组
    grouped = df.groupby(group_cols)

    # 统计按用户id，app_id，app_type分组的总使用次数（app_id出现了几次就使用了几次）
    use_count = grouped.size().reset_index(name='use_count')

    # 统计使用时长的总和，平均每次的使用时长，使用时长的最大值，标准差和偏度
    use_duration_sum = grouped['duration'].sum().reset_index(name='use_duration_sum')
    use_duration_mean = grouped['duration'].mean().reset_index(name='use_duration_mean')
    use_duration_max = grouped['duration'].max().reset_index(name='use_duration_max')
    use_duration_std = grouped['duration'].std().reset_index(name='use_duration_std')
    use_duration_skew = grouped['duration'].skew().reset_index(name='use_duration_skew')

    # 统计上行流量和下行流量的总和，平均每次的使用量，最大值，标准差和偏度
    up_flow_sum = grouped['up_flow'].sum().reset_index(name='up_flow_sum')
    up_flow_mean = grouped['up_flow'].mean().reset_index(name='up_flow_mean')
    up_flow_max = grouped['up_flow'].max().reset_index(name='up_flow_max')
    up_flow_std = grouped['up_flow'].std().reset_index(name='up_flow_std')
    up_flow_skew = grouped['up_flow'].skew().reset_index(name='up_flow_skew')

    down_flow_sum = grouped['down_flow'].sum().reset_index(name='down_flow_sum')
    down_flow_mean = grouped['down_flow'].mean().reset_index(name='down_flow_mean')
    down_flow_max = grouped['down_flow'].max().reset_index(name='down_flow_max')
    down_flow_std = grouped['down_flow'].std().reset_index(name='down_flow_std')
    down_flow_skew = grouped['down_flow'].skew().reset_index(name='down_flow_skew')

    # 统计有效日均时长，即每天至少使用一次的情况下，每天的平均使用时长
    # 首先，根据用户id和app_id进行分组，计算每一天的使用时长总和
    daily_duration = df.groupby([*group_cols, 'day'])['duration'].sum().reset_index()
    # 然后，根据用户id和app_id进行分组，计算有效日均时长
    daily_duration_mean = daily_duration.groupby(group_cols)['duration'].mean().reset_index(
        name='daily_duration_mean')

    # 首先，根据用户id和app_id进行分组，提取每一次使用的日期
    use_date = df.groupby([*group_cols, 'day']).size().reset_index()[[*group_cols, 'day']]

    # 统计使用频率（使用天数占比）
    # 然后，根据用户id和app_id进行分组，计算总共的天数
    total_days_count = df.groupby(group_cols)['day'].nunique().reset_index(name='total_days_count')


    # 统计日均次数、日均时长、日均流量
    # 首先，根据用户id和app_id进行分组，计算每一天的使用次数、使用时长、上行流量、下行流量
    daily_stats = df.groupby([*group_cols, 'day']).agg(
        {'duration': 'sum', 'up_flow': 'sum', 'down_flow': 'sum', 'user_id': 'count'}).rename(
        columns={'user_id': 'use_count'}).reset_index()
    # 然后，根据用户id和app_id进行分组，计算日均次数、日均时长、日均流量
    daily_stats_mean = daily_stats.groupby(group_cols).mean().rename(
        columns={'use_count': 'daily_use_count', 'duration': 'daily_use_duration', 'up_flow': 'daily_up_flow',
                 'down_flow': 'daily_down_flow'}).reset_index()
    # 将所有的特征合并为一个数据框
    features = pd.merge(use_count, use_duration_sum, on=group_cols)
    features = pd.merge(features, use_duration_mean, on=group_cols)
    features = pd.merge(features, use_duration_max, on=group_cols)
    features = pd.merge(features, use_duration_std, on=group_cols)
    features = pd.merge(features, use_duration_skew, on=group_cols)
    features = pd.merge(features, up_flow_sum, on=group_cols)
    features = pd.merge(features, up_flow_mean, on=group_cols)
    features = pd.merge(features, up_flow_max, on=group_cols)
    features = pd.merge(features, up_flow_std, on=group_cols)
    features = pd.merge(features, up_flow_skew, on=group_cols)
    features = pd.merge(features, down_flow_sum, on=group_cols)
    features = pd.merge(features, down_flow_mean, on=group_cols)
    features = pd.merge(features, down_flow_max, on=group_cols)
    features = pd.merge(features, down_flow_std, on=group_cols)
    features = pd.merge(features, down_flow_skew, on=group_cols)
    features = pd.merge(features, daily_duration_mean, on=group_cols)
    features = pd.merge(features, daily_stats_mean, on=group_cols)
    features = features.drop(['day'], axis=1)
    # 输出结果
    return features

# 使用feature_extract这个函数进行特征提取
df_1_15 = feature_extract(df_1_15, group_cols)
df_16_30 = feature_extract(df_16_30, group_cols)

# 输出结果
df_1_15.to_csv('按用户和使用app分组求和后并提取特征的df_1_15.csv', index=False)
df_16_30.to_csv('按用户和使用app分组求和后并提取特征的df_16_30.csv', index=False)

# 定义欠采样器
rus = RandomUnderSampler(sampling_strategy='not minority', random_state=0)

# 对训练集进行欠采样
X_train, y_train = rus.fit_resample(df_1_15.drop(['app_id'], axis=1), df_1_15[['app_id']])
X_test, y_test = rus.fit_resample(df_16_30.drop(['app_id'], axis=1), df_16_30[['app_id']])

# 将特征和标签合并为一个数据框
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)


# 重置索引并删除不需要的索引列
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# 输出结果
train.to_csv('欠采样并提取特征后的df_1_15.csv', index=False)
test.to_csv('欠采样并提取特征后的df_16_30.csv', index=False)


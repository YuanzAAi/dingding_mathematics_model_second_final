import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
plt.rcParams["font.family"]="STSong"

# 读取数据
df_1_15 = pd.read_csv("df_1_15.csv")
df_16_30 = pd.read_csv("df_16_30.csv")

# 将df_1_15和df_16_30按用户id，app_id和app_type进行合并，并且去除start_datetime和end_datetime这两列
df = pd.concat([df_1_15, df_16_30], ignore_index=True)
df = df.drop(['start_datetime', 'end_datetime'], axis=1)
del df_1_15,df_16_30

# 首先进行描述性统计，将app_id，duration、up_flow和down_flow的均值，中位数，最大值，最小值，方差、标准差，上四分位和下四分位计算出来，储存在代码框里
desc_stats = df.groupby('app_id').agg({'duration': ['mean', 'median', 'max', 'min', 'var', 'std', 'quantile'], 'up_flow': ['mean', 'median', 'max', 'min', 'var', 'std', 'quantile'], 'down_flow': ['mean', 'median', 'max', 'min', 'var', 'std', 'quantile']})
# 然后输出为描述性统计.csv
desc_stats.to_csv('源数据描述性统计.csv')

# 根据user_id和app_id和app_type进行分组
df = df.groupby(['user_id', 'app_id', 'app_type']).sum().reset_index()
print(df)

# 然后画出合并的数据的箱线图
plt.figure(figsize=(12, 4))
sns.boxplot(data=df[["duration", "up_flow", "down_flow"]])
plt.title("源数据箱线图")
plt.savefig('源数据箱线图.png',dpi=300)
plt.show()

# 然后用duration、up_flow和down_flow画出分布图，用一个1行3列的子图表示，图片要dpi=300保存
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(data=df, x='duration')
plt.xlabel('使用时长（秒）')
plt.ylabel('频数')
plt.title('源数据使用时长分布图')

plt.subplot(1, 3, 2)
sns.histplot(data=df, x='up_flow')
plt.xlabel('上行流量')
plt.ylabel('频数')
plt.title('源数据上行流量分布图')

plt.subplot(1, 3, 3)
sns.histplot(data=df, x='down_flow')
plt.xlabel('下行流量')
plt.ylabel('频数')
plt.title('源数据下行流量分布图')

plt.tight_layout()
plt.savefig('源数据直方分布图.png', dpi=300)
plt.show()

# 再画出duration、up_flow和down_flow的核密度图，看看数据是否接近正态分布，图片要dpi=300保存
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.kdeplot(data=df, x='duration')
plt.xlabel('使用时长（秒）')
plt.ylabel('密度')
plt.title('源数据使用时长核密度图')

plt.subplot(1, 3, 2)
sns.kdeplot(data=df, x='up_flow')
plt.xlabel('上行流量')
plt.ylabel('密度')
plt.title('源数据上行流量核密度图')

plt.subplot(1, 3, 3)
sns.kdeplot(data=df, x='down_flow')
plt.xlabel('下行流量')
plt.ylabel('密度')
plt.title('源数据下行流量核密度图')

plt.tight_layout()
plt.savefig('源数据核密度图.png', dpi=300)
plt.show()

# 使用对数变换进行处理，使其分布更加平滑
df["duration"] = np.log1p(df["duration"])
df["up_flow"] = np.log1p(df["up_flow"])
df["down_flow"] = np.log1p(df["down_flow"])

# 使用归一化进行处理，使其值在0到1之间
scaler = MinMaxScaler()
df["duration"] = scaler.fit_transform(df[["duration"]].values.reshape(-1, 1))
df["up_flow"] = scaler.fit_transform(df["up_flow"].values.reshape(-1, 1))
df["down_flow"] = scaler.fit_transform(df["down_flow"].values.reshape(-1, 1))

#处理后再次画图

plt.figure(figsize=(12, 4))
sns.boxplot(data=df[["duration", "up_flow", "down_flow"]])
plt.title("处理后箱线图")
plt.savefig('对数化归一化后箱线图.png',dpi=300)
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(data=df, x='duration')
plt.xlabel('使用时长（秒）')
plt.ylabel('频数')
plt.title('处理后使用时长分布图')

plt.subplot(1, 3, 2)
sns.histplot(data=df, x='up_flow')
plt.xlabel('上行流量')
plt.ylabel('频数')
plt.title('处理后上行流量分布图')

plt.subplot(1, 3, 3)
sns.histplot(data=df, x='down_flow')
plt.xlabel('下行流量')
plt.ylabel('频数')
plt.title('处理后下行流量分布图')

plt.tight_layout()
plt.savefig('对数化归一化直方分布图.png', dpi=300)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.kdeplot(data=df, x='duration')
plt.xlabel('使用时长（秒）')
plt.ylabel('密度')
plt.title('处理后使用时长核密度图')

plt.subplot(1, 3, 2)
sns.kdeplot(data=df, x='up_flow')
plt.xlabel('上行流量')
plt.ylabel('密度')
plt.title('处理后上行流量核密度图')

plt.subplot(1, 3, 3)
sns.kdeplot(data=df, x='down_flow')
plt.xlabel('下行流量')
plt.ylabel('密度')
plt.title('处理后下行流量核密度图')

plt.tight_layout()
plt.savefig('对数化归一化后核密度图.png', dpi=300)
plt.show()
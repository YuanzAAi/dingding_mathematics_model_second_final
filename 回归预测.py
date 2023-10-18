# 导入相关库
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import  RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import  Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import  SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import  DecisionTreeRegressor
from xgboost import XGBRegressor
import tensorflow as tf
plt.rcParams["font.family"]="STSong"

# 读取训练集和测试集
train = pd.read_csv('欠采样并提取特征后的df_1_15.csv')
test = pd.read_csv('欠采样并提取特征后的df_16_30.csv')
train0 = pd.read_csv('无对数化归一化欠采样并提取特征后的df_1_15.csv')
test0 = pd.read_csv('无对数化归一化欠采样并提取特征后的df_16_30.csv')

# 选择特征和标签
X_train = train0[['daily_down_flow', 'up_flow_max', 'down_flow_max', 'use_duration_max', 'down_flow_mean', 'up_flow_mean']]
y_train = train0['daily_use_duration'] # 使用时长作为标签
X_test = test0[['daily_down_flow', 'up_flow_max', 'down_flow_max', 'use_duration_max', 'down_flow_mean', 'up_flow_mean']]
y_test = test0['daily_use_duration'] # 使用时长作为标签

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2023)

# 模型，并手动调整参数的值(要使用哪个模型就把注释号去掉就可)
#model = SVR(C=10, kernel='rbf', gamma='scale')
#model = MLPRegressor(hidden_layer_sizes=20, activation='relu', max_iter=10000, early_stopping=True)
#model = DecisionTreeRegressor(max_depth=3, min_samples_split=5, criterion='squared_error', random_state=317)
#model = RandomForestRegressor(n_estimators=300, max_depth=5,criterion='squared_error', min_samples_split=6, random_state=317)
#model = Ridge(alpha=100)
#model = KNeighborsRegressor(n_neighbors=8,leaf_size=20,algorithm='kd_tree')
#model = AdaBoostRegressor(n_estimators=300, learning_rate=0.1)
#model = GradientBoostingRegressor(n_estimators=300, criterion='friedman_mse', learning_rate=0.1, max_depth=3)
#model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, seed=317, gamma=0.25)
#model = LGBMRegressor(n_estimators=300, learning_rate=0.1, max_depth=3)
model = CatBoostRegressor(n_estimators=300, learning_rate=0.1, max_depth=3)
'''BP神经网络，有两个隐藏层，激活函数为relu'''
# 创建回归问题的神经网络模型
#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(6, activation='relu', input_shape=(6,)),
#    tf.keras.layers.Dense(64, activation='relu'),
#    tf.keras.layers.Dense(1, activation='relu')])

# 编译模型，使用均方误差损失函数和回归指标
#model.compile(loss='mean_squared_error', metrics=['mean_squared_error'])
# 训练模型，设置批次大小为10，迭代次数为1000
#model.fit(X_train, y_train, batch_size=128, epochs=300)
'''要使用这个模型就把注释号去掉就可'''

# 使用指定的模型在训练集上进行训练
model.fit(X_train, y_train)


# 使用指定的模型在测试集上进行预测
y_pred = model.predict(X_test)

# 计算NMSE和RMSE
r2 = r2_score(y_test, y_pred) # 计算R平方
y_test = y_test.ravel()
y_pred = y_pred.ravel()
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
nmse = np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)



# 将结果转换为数据框，并设置列名即模型名、NMSE、RMSE
result = pd.DataFrame([[model.__class__.__name__, nmse, rmse, r2]], columns=['Model', 'NMSE', 'RMSE', 'R2']) # 增加R平方

# 保存为：各模型回归的评价结果.csv，使用追加模式
result.to_csv('各模型回归的评价结果.csv', index=False, mode='a', header=False) # 这里使用mode='a'表示追加模式，不会覆盖之前的结果

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from surprise import Dataset, Reader, SVD, accuracy
plt.rcParams["font.family"]="STSong"

# 读取训练集和测试集
train = pd.read_csv('按用户和使用app分组求和后并提取特征的df_1_15.csv')
test = pd.read_csv('按用户和使用app分组求和后并提取特征的df_16_30.csv')

# 对user_id和app_id进行编码
le_user = LabelEncoder()
le_app = LabelEncoder()
train['user_id'] = le_user.fit_transform(train['user_id'])
train['app_id'] = le_app.fit_transform(train['app_id'])
test['user_id'] = le_user.transform(test['user_id'])
test['app_id'] = le_app.transform(test['app_id'])

# 对数值特征进行归一化
scaler = MinMaxScaler()
train[['daily_down_flow','up_flow_max','down_flow_max','use_duration_max', 'down_flow_mean', 'up_flow_mean']] = scaler.fit_transform(train[['daily_down_flow','up_flow_max','down_flow_max','use_duration_max', 'down_flow_mean', 'up_flow_mean']])
test[['daily_down_flow','up_flow_max','down_flow_max','use_duration_max', 'down_flow_mean', 'up_flow_mean']] = scaler.transform(test[['daily_down_flow','up_flow_max','down_flow_max','use_duration_max', 'down_flow_mean', 'up_flow_mean']])

# 创建评分范围的Reader对象
reader = Reader(rating_scale=(0, 1))

# 加载数据并划分为训练集和测试集
data_train = Dataset.load_from_df(train[['user_id', 'app_id', 'use_duration_max']], reader)
data_test = Dataset.load_from_df(test[['user_id', 'app_id', 'use_duration_max']], reader)
trainset = data_train.build_full_trainset()
testset = data_test.build_full_trainset().build_testset()

# 创建SVD模型并拟合
svd = SVD()
svd.fit(trainset)

# 在测试集上进行预测
predictions = svd.test(testset)

# 评估模型的均方根误差和平均绝对误差
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# 构建推荐系统模型
recommendations = {}

# 获取所有的用户id和物品id
user_ids = train['user_id'].unique()
item_ids = train['app_id'].unique()

# 遍历每个用户
for user_id in user_ids:
    max_rating = 0
    max_item = None

    # 遍历每个物品
    for item_id in item_ids:
        prediction = svd.predict(user_id, item_id)

        # 如果评分大于最高评分，更新最高评分和推荐物品
        if prediction.est > max_rating:
            max_rating = prediction.est
            max_item = item_id

    # 将推荐物品存入字典中
    recommendations[user_id] = max_item

# 使用可视化展示推荐结果
users = list(recommendations.keys())
items = list(recommendations.values())
ratings = [svd.predict(user, item).est for user, item in recommendations.items()]

plt.figure(figsize=(10, 6))
plt.bar(users, ratings, color=plt.cm.jet(items))
plt.xlabel('User ID')
plt.ylabel('Predicted Rating')
plt.title('推荐结果')
plt.savefig('推荐得分.png', dpi=300)
plt.show()


# 定义一个函数，接受一个用户id作为参数，返回推荐的app_id和预测评分
def recommend(user_id):
    # 检查用户id是否在训练集中，如果不在，返回提示信息
    if user_id not in list(user_ids):
        return '抱歉，该用户id不存在，请输入一个有效的用户id。'

    # 初始化一个空列表，用来存储每个物品的预测评分
    ratings = []

    # 遍历每个物品
    for item_id in item_ids:
        # 预测评分
        prediction = svd.predict(user_id, item_id)
        # 将预测评分和物品id添加到列表中
        ratings.append((prediction.est, item_id))

    # 对列表按照预测评分降序排序
    ratings.sort(reverse=True)

    # 取出前五个物品的预测评分和id
    top_five = ratings[:5]

    # 返回推荐物品和预测评分
    return f'为用户{user_id}推荐的app_id和预测评分是：\n' + '\n'.join(
        [f'app_id: {item[1]}, 预测评分: {item[0]:.2f}' for item in top_five])


# 输入一个用户id，调用函数，打印结果
user_id = int(input('请输入一个用户id：'))
result = recommend(user_id)
print(result)

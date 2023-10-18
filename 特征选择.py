# 导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # 导入随机森林分类器
from xgboost import XGBClassifier # 导入XGBoost分类器
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier # 导入AdaBoost,GradientBoosting分类器
from sklearn.tree import DecisionTreeClassifier # 导入决策树分类器
plt.rcParams["font.family"]="STSong"

# 读取训练集和测试集
train = pd.read_csv('欠采样并提取特征后的df_1_15.csv')
test = pd.read_csv('欠采样并提取特征后的df_16_30.csv')
train0 = pd.read_csv('无对数化归一化欠采样并提取特征后的df_1_15.csv')
test0 = pd.read_csv('无对数化归一化欠采样并提取特征后的df_16_30.csv')

# 绘制各个特征之间的相关性热力图
plt.figure(figsize=(18, 18))
sns.heatmap(train.corr(), cmap='coolwarm',annot=True)
plt.savefig('特征相关性热力图.png', dpi=300)
plt.show()

# 计算每个特征的shap值，然后进行可视化
X_train = train.drop(['user_id', 'app_id'], axis=1).fillna(0)
y_train = train['app_id']
X_train0 = train0.drop(['user_id', 'app_id'], axis=1).fillna(0)
y_train0 = train0['app_id']

# 使用逻辑回归模型进行分类预测
model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)

# 使用shap库计算shap值
model_callable = lambda x: model.predict_proba(x)[:, 1]
masker = shap.maskers.Independent(X_train)
explainer = shap.Explainer(model_callable, masker=masker)
shap_values = explainer(X_train)

# 绘制所有特征的shap值条形图
shap.plots.bar(shap_values,show = False)
plt.title('部分特征shap图')
plt.gcf().set_size_inches(18,14)
plt.savefig('shap值图.png', dpi=300)
plt.show()

shap.plots.beeswarm(shap_values,show = False)
plt.title('特征区分情况')
plt.gcf().set_size_inches(18,14)
plt.savefig('区分情况.png', dpi=300)
plt.show()

# 定义一个函数，用于获取树模型的特征重要性，并返回一个排序后的数据框
def get_feature_importance(model, X_train):
    # 获取特征名和重要性值
    feature_names = X_train.columns
    feature_importance = model.feature_importances_
    # 创建一个数据框，存储特征名和重要性值，并按照重要性值降序排序
    df_feature_importance = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    df_feature_importance = df_feature_importance.sort_values(by='importance', ascending=False)
    # 返回排序后的数据框
    return df_feature_importance

# 定义一个函数，用于绘制多柱图，显示不同树模型的特征重要性排行前5的特征
def plot_feature_importance(df_rf, df_xgb, df_ada, df_gb, df_dt):
    # 设置画布大小
    plt.figure(figsize=(15, 10))
    # 设置柱状图的宽度
    bar_width = 0.15
    # 设置柱状图的位置
    x1 = np.arange(5)
    x2 = x1 + bar_width
    x3 = x2 + bar_width
    x4 = x3 + bar_width
    x5 = x4 + bar_width
    # 绘制随机森林的特征重要性柱状图
    plt.bar(x1, df_rf['importance'][:5], width=bar_width, label='Random Forest')
    # 绘制XGBoost的特征重要性柱状图
    plt.bar(x2, df_xgb['importance'][:5], width=bar_width, label='XGBoost')
    # 绘制AdaBoost的特征重要性柱状图
    plt.bar(x3, df_ada['importance'][:5], width=bar_width, label='AdaBoost')
    # 绘制GBDT的特征重要性柱状图
    plt.bar(x4, df_gb['importance'][:5], width=bar_width, label='GBDT')
    # 绘制决策树的特征重要性柱状图
    plt.bar(x5, df_dt['importance'][:5], width=bar_width, label='Decision Tree')
    # 设置x轴的刻度和标签
    plt.xticks(x3, df_rf['feature'][:5])
    # 设置y轴的标签
    plt.ylabel('Feature Importance')
    # 设置标题
    plt.title('Feature Importance Comparison of Tree Models')
    # 显示图例
    plt.legend()
    # 保存图片到本地
    plt.savefig('树模型的特征重要性.png', dpi=300)
    # 显示图片
    plt.show()

# 使用随机森林分类器训练模型，并获取特征重要性数据框
rf = RandomForestClassifier()
rf.fit(X_train0, y_train0)
df_rf = get_feature_importance(rf, X_train0)

# 使用XGBoost分类器训练模型，并获取特征重要性数据框
xgb = XGBClassifier()
xgb.fit(X_train0, y_train0)
df_xgb = get_feature_importance(xgb, X_train0)

# 使用AdaBoost分类器训练模型，并获取特征重要性数据框
ada = AdaBoostClassifier()
ada.fit(X_train0, y_train0)
df_ada = get_feature_importance(ada, X_train0)

# 使用GBDT分类器训练模型，并获取特征重要性数据框
gb = GradientBoostingClassifier()
gb.fit(X_train0, y_train0)
df_gb = get_feature_importance(gb, X_train0)

# 使用决策树分类器训练模型，并获取特征重要性数据框
dt = DecisionTreeClassifier()
dt.fit(X_train0, y_train0)
df_dt = get_feature_importance(dt, X_train0)


# 调用绘图函数，显示不同树模型的特征重要性排行前5的特征
plot_feature_importance(df_rf, df_xgb, df_ada, df_gb, df_dt)
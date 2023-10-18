import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams["font.family"]="STSong"

result = pd.read_csv('各模型回归的评价结果.csv')

# 定义一个函数，用于绘制多柱图，显示不同分类器的评价指标
def plot_evaluation_metrics(df_results):
    # 设置画布大小
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    # 设置柱状图的宽度
    bar_width = 0.15
    # 设置柱状图的位置
    x5 = np.arange(1)
    x6 = x5 + bar_width
    x7 = x6 + bar_width
    x8 = x7 + bar_width
    x9 = x8 + bar_width
    x10 = x9 + bar_width
    x11 = x10 + bar_width
    # 画图
    plt.bar(x5, df_results.iloc[4, 2], width=bar_width, label='RF')
    plt.bar(x6, df_results.iloc[5, 2], width=bar_width, label='DT')
    plt.bar(x7, df_results.iloc[6, 2], width=bar_width, label='AdaBoost')
    plt.bar(x8, df_results.iloc[7, 2], width=bar_width, label='GBDT')
    plt.bar(x9, df_results.iloc[8, 2], width=bar_width, label='XGBBoost')
    plt.bar(x10, df_results.iloc[9, 2], width=bar_width, label='LightGBM')
    plt.bar(x11, df_results.iloc[10, 2], width=bar_width, label='CatBoost')
    # 设置x轴的刻度和标签
    plt.xticks([x5[0] + bar_width / 2], [df_results.columns[2]])
    # 设置y轴的标签
    plt.ylabel('分数')
    # 设置标题
    plt.title('RMSE(树和boost模型)')
    # 显示图例
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    x5 = np.arange(2)
    x6 = x5 + bar_width
    x7 = x6 + bar_width
    x8 = x7 + bar_width
    x9 = x8 + bar_width
    x10 = x9 + bar_width
    x11 = x10 + bar_width
    # 画图
    plt.bar(x5, df_results.iloc[4, [1, 3]], width=bar_width, label='RF')
    plt.bar(x6, df_results.iloc[5, [1, 3]], width=bar_width, label='DT')
    plt.bar(x7, df_results.iloc[6, [1, 3]], width=bar_width, label='AdaBoost')
    plt.bar(x8, df_results.iloc[7, [1, 3]], width=bar_width, label='GBDT')
    plt.bar(x9, df_results.iloc[8, [1, 3]], width=bar_width, label='XGBBoost')
    plt.bar(x10, df_results.iloc[9, [1, 3]], width=bar_width, label='LightGBM')
    plt.bar(x11, df_results.iloc[10, [1, 3]], width=bar_width, label='CatBoost')
    # 设置x轴的刻度和标签
    plt.xticks(x5 + bar_width / 2, df_results.columns[[1, 3]])
    # 设置y轴的标签
    plt.ylabel('分数')
    # 设置标题
    plt.title('NMSE, R^2(树和boost模型)')
    # 显示图例
    plt.legend(loc='upper right')

    # 保存图片到本地
    plt.savefig('各模型评价指标(树和boost模型).png', dpi=300)
    # 显示图片
    plt.show()


# 调用函数
plot_evaluation_metrics(result)
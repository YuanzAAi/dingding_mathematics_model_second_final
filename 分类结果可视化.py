import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams["font.family"]="STSong"

result = pd.read_csv('各模型分类的评价结果.csv')

# 定义一个函数，用于绘制多柱图，显示不同分类器的评价指标
def plot_evaluation_metrics(df_results):
    # 设置画布大小
    plt.figure(figsize=(15, 10))
    # 设置柱状图的宽度
    bar_width = 0.15
    # 设置柱状图的位置
    x1 = np.arange(4)
    x2 = x1 + bar_width
    x3 = x2 + bar_width
    x4 = x3 + bar_width
    x5 = x4 + bar_width
    x6 = x5 + bar_width
    x7 = x6 + bar_width
    x8 = x7 + bar_width
    x9 = x8 + bar_width
    x10 = x9 + bar_width
    x11 = x10 + bar_width
    # 画图
    plt.bar(x1, df_results.iloc[0, 1:], width=bar_width, label='MLP')
    plt.bar(x2, df_results.iloc[1, 1:], width=bar_width, label='DecisionTree')
    plt.bar(x3, df_results.iloc[2, 1:], width=bar_width, label='RandomForest')
    plt.bar(x4, df_results.iloc[3, 1:], width=bar_width, label='Logistic')
    plt.bar(x5, df_results.iloc[4, 1:], width=bar_width, label='AdaBoost')
    plt.bar(x6, df_results.iloc[5, 1:], width=bar_width, label='GBDT')
    plt.bar(x7, df_results.iloc[6, 1:], width=bar_width, label='LightGBM')
    plt.bar(x8, df_results.iloc[7, 1:], width=bar_width, label='XGBBoost')
    plt.bar(x9, df_results.iloc[8, 1:], width=bar_width, label='CatBoost')
    plt.bar(x10, df_results.iloc[9, 1:], width=bar_width, label='SVC')
    plt.bar(x11, df_results.iloc[10, 1:], width=bar_width, label='softmax')
    # 设置x轴的刻度和标签，使用评价指标的名称
    plt.xticks(x5 + bar_width / 2, df_results.columns[1:])
    # 设置y轴的标签，使用分数作为单位
    plt.ylabel('分数')
    # 设置标题，使用分类器性能比较作为标题
    plt.title('各模型评价指标')
    # 显示图例，使用分类器的名称作为图例，并设置位置在右上角
    plt.legend(loc='upper right')
    # 保存图片到本地，使用evaluation_metrics_comparison.png作为文件名，并设置分辨率为300dpi
    plt.savefig('各模型评价指标.png', dpi=300)
    # 显示图片
    plt.show()

plot_evaluation_metrics(result)
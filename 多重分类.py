# 导入相关库
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from keras.utils import to_categorical
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import tensorflow as tf
plt.rcParams["font.family"]="STSong"

# 读取训练集和测试集
train = pd.read_csv('按用户和使用app分组求和后并提取特征的df_1_15.csv')
test = pd.read_csv('按用户和使用app分组求和后并提取特征的df_16_30.csv')
train0 = pd.read_csv('无对数化归一化欠采样并提取特征后的df_1_15.csv')
test0 = pd.read_csv('无对数化归一化欠采样并提取特征后的df_16_30.csv')

# 选择特征和标签
X_train = train[['daily_down_flow','up_flow_max','down_flow_max' , 'use_duration_max', 'down_flow_mean',  'up_flow_mean',]]
y_train = train['app_id']
X_test = test[['daily_down_flow','up_flow_max','down_flow_max', 'use_duration_max', 'down_flow_mean', 'up_flow_mean']]
y_test = test['app_id']



# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2023)

# 模型，并手动调整参数的值(要用哪个模型的时候就把注释给去掉)
model = SVC(C=10, kernel='rbf', gamma='scale', probability=True)
#model = MLPClassifier(hidden_layer_sizes=20,activation='relu',max_iter=10000,early_stopping=True)
#model = DecisionTreeClassifier(max_depth=3,min_samples_split=5,criterion='gini',random_state = 317)
#model = RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_split=6,random_state = 317)
#model = LogisticRegression(C=0.1, solver='saga', multi_class='multinomial',max_iter=1000000,tol=1e-6,penalty='elasticnet',l1_ratio=1,verbose=1)
#model = AdaBoostClassifier(n_estimators=300,learning_rate=0.1)
#model = GradientBoostingClassifier(n_estimators=300,criterion='friedman_mse',learning_rate=0.1,max_depth=3)
#model = XGBClassifier(n_estimators=300,learning_rate=0.1,max_depth=6,seed = 317,gamma=0.25)
#model = LGBMClassifier(n_estimators=300,learning_rate=0.1,max_depth=3)
#model = CatBoostClassifier(n_estimators=300,learning_rate=0.1,max_depth=3)

'''多重分类神经网络模型'''
# 将标签数据转换为独热编码形式
#y_train = to_categorical(y_train, num_classes=20)
#y_test = to_categorical(y_test, num_classes=20)
#model = tf.keras.Sequential([
    #tf.keras.layers.Dense(6, activation='relu', input_shape=(6,)),
    #tf.keras.layers.Dense(10, activation='relu'),
    #tf.keras.layers.Dense(20, activation='softmax')  # 20重分类，softmax
#])

# 编译模型，设置损失函数为多分类交叉熵，评价指标为准确率
#model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型，设置批次大小为10，迭代次数为1000
#model.fit(X_train, y_train, batch_size=10, epochs=300)



# 使用指定的模型在训练集上进行训练
model.fit(X_train, y_train)

# 使用指定的模型在测试集上进行预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 计算准确率、召回率、F1值和AUC值
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro' )
auc = roc_auc_score(y_test, y_prob, average='macro', multi_class='ovo')

# 将结果转换为数据框，并设置列名即模型名、准确率、召回率、F1值和AUC值
result = pd.DataFrame([[model.__class__.__name__, acc, rec, f1, auc]], columns=['Model', 'Accuracy', 'Recall', 'F1-score', 'AUC'])

# 保存为：各模型分类的评价结果.csv，使用追加模式
result.to_csv('各模型分类的评价结果.csv', index=False,mode='a',header=False) # 这里使用mode='a'表示追加模式，不会覆盖之前的结果


# 画出带交叉验证的ROC曲线
plt.figure(figsize=(10, 8))
for i in range(5):
    # 划分训练集和验证集
    X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(X_train, y_train, test_size=0.2)
    # 使用最优参数的分类器在验证集上进行预测
    model.fit(X_train_cv, y_train_cv)
    y_prob_cv = model.predict_proba(X_val_cv)
    # 计算多分类问题的AUC值，使用ovo模式
    auc_cv = roc_auc_score(y_val_cv, y_prob_cv, average='macro', multi_class='ovo')
    # 绘制ROC曲线，使用ovo模式
    n_classes = len(np.unique(y_val_cv))
    fpr_cv = dict()
    tpr_cv = dict()
    for j in range(n_classes):
        fpr_cv[j], tpr_cv[j], _ = roc_curve(y_val_cv == j, y_prob_cv[:, j])
        plt.plot(fpr_cv[j], tpr_cv[j], label=f'Class {j}, AUC={auc_cv:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('ROC 曲线包含5折交叉')
plt.legend()
plt.savefig('BP的ROC曲线.png', dpi=300)
plt.show()

# 画出混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
plt.imshow(cm, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('混淆矩阵')
plt.colorbar()
plt.savefig('BP的分类混淆矩阵.png', dpi=300)
plt.show()

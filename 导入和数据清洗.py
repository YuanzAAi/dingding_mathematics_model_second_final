# 导入所需的库
import pandas as pd

# 定义一个函数，用于读取用户监测数据，并转换为数据框
def read_data(file):
    # 读取txt文件，分隔符为逗号，列名为表1的列名
    df = pd.read_csv(file, sep=',', header=None, names=['user_id', 'app_id', 'app_type', 'start_day', 'start_time', 'end_day', 'end_time', 'duration', 'up_flow', 'down_flow'])
    # 根据app_id进行筛选，借助辅助表格app_class.csv，只保留app_id在辅助表格中出现的数据
    app_class = pd.read_csv('app_class.csv') # 读取辅助表格
    df = df[df['app_id'].isin(app_class['app_id'])] # 筛选数据
    # app_type转换成0和1，即sys为0，usr为1
    df['app_type'] = df['app_type'].map({'sys': 0, 'usr': 1})
    # 将start_day和end_day中为0的值加1，以便转换为datetime类型
    df['start_day'] = df['start_day'].replace(0, 1)
    df['end_day'] = df['end_day'].replace(0, 1)
    # 进行异常值检测，将start_day和end_day中为负数的值视为异常值，并将其剔除
    abnormal = df[(df['start_day'] < 0) | (df['end_day'] < 0)] # 提取异常值
    abnormal.to_csv('异常值.csv', index=False, mode='a',header=False) # 将异常值追加到异常值.csv
    df = df[(df['start_day'] > 0) & (df['end_day'] > 0)] # 剔除异常值
    # 进行重复值检测，将数据中整行相同的重复值找出来，删除重复的行，保留一个即可
    duplicate = df[df.duplicated()] # 提取重复值
    duplicate.to_csv('重复值.csv', index=False, mode='a',header=False) # 将重复值追加到重复值.csv
    df = df.drop_duplicates() # 删除重复的行
    # 进行缺失值处理，使用上一行的元素来替代缺失值
    missing = df[df.isnull().any(axis=1)] # 提取缺失值
    missing.to_csv('缺失值.csv', index=False, mode='a',header=False) # 将缺失值追加到缺失值.csv
    df = df.fillna(method='ffill') # 使用上一行的元素来替代缺失值
    # start_day和start_time弄在一起，将其转换为datetime类型
    df['start_datetime'] = pd.to_datetime(year_month + df['start_day'].astype(str) + ' ' + df['start_time'], format='%Y%m%j %H:%M:%S')
    # end_day和end_time也弄在一起，将其转换为datetime类型
    df['end_datetime'] = pd.to_datetime(year_month + df['end_day'].astype(str) + ' ' + df['end_time'], format='%Y%m%j %H:%M:%S')
    print(df['start_datetime'])
    # 删除不需要的列
    df.drop(['start_day', 'start_time', 'end_day', 'end_time'], axis=1, inplace=True)
    # 按照user_id，app_id，app_type和处理过的变成datetime类型的start_datetime和end_datetime进行分组并求和
    df = df.groupby(['user_id', 'app_id', 'app_type', 'start_datetime', 'end_datetime']).sum().reset_index()
    return df

# 定义一个空列表，用于存放每次循环的数据框
data_list = []
# 定义一个参数n，用于控制每次循环读取的文件数量
n = 15
# 定义一个参数year_month，用于指定年份和月份
year_month = '202307'
# 用循环读取用户监测数据，文件名为day01.txt到day30.txt
for i in range(1, 31, n):
    # 定义一个空列表，用于存放每次循环内部的数据框
    data_sublist = []
    # 循环读取n个文件，并将其添加到子列表中
    for j in range(i, i+n):
        file = f'day{j:02d}.txt' # 文件名
        data = read_data(file) # 调用函数读取数据并转换为数据框
        print('文件',j,'已完成')
        data_sublist.append(data) # 将数据框添加到子列表中
    # 将子列表中的数据框合并为一个数据框，并将其添加到总列表中
    data_list.append(pd.concat(data_sublist, ignore_index=True))
    # 删除子列表和数据框，释放内存
    del data_sublist, data

# 将总列表中的数据框合并为一个大的数据框
data_all = pd.concat(data_list, ignore_index=True)

# 将数据分为两部分来进行，分别是1~15天的和16~30天的
data_1_15 = data_all[data_all['start_datetime'].dt.day <= 15] # 筛选出1~15天的数据
data_16_30 = data_all[data_all['start_datetime'].dt.day > 15] # 筛选出16~30天的数据

# 将处理完和筛选完的数据框保存，分别命名为df_1_15和df_16_30
data_1_15.to_csv('df_1_15.csv', index=False) # 保存1~15天的数据
data_16_30.to_csv('df_16_30.csv', index=False) # 保存16~30天的数据

# 打印输出信息
print('数据预处理完成，已经保存为df_1_15.csv和df_16_30.csv，异常值、重复值和缺失值也已经保存为异常值.csv、重复值.csv和缺失值.csv')

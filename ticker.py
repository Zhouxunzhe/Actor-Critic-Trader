import os
import pandas as pd

# 读取tickers.txt中的所有ticker
file = open('env/data/SH_2024/tickers.txt', 'r')
tickers = [line.strip() for line in file.readlines()]
file.close()

datas = []
for i in range(len(tickers)):
    file_path = f'env/data/SH_2024/ticker_{tickers[i]}.csv'
    data = pd.read_csv(file_path)
    datas.append(data)

# 获取最小的行数
min_rows = min(df.shape[0] for df in datas)
datas = [df.tail(min_rows) for df in datas]

# 保存截取后的数据
for i in range(len(tickers)):
    output_file_path = f'env/data/SH_2024/{tickers[i]}.csv'
    datas[i].to_csv(output_file_path, index=False)

print("All files have been processed successfully.")

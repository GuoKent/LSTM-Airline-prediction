import matplotlib.pyplot as plt
import pandas as pd

# 读取数据,取消标题
data = pd.read_csv('./data.csv', usecols=[1], header=None)

# 画图
plt.plot(data, 'b', label='real')
plt.legend(loc='best')
plt.show()

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random

# 这里修改网络结构
input_len = 12  # 使用前12个月的数据预测下一个月
hidden_size = 4
output_size = 1
num_layer = 2


# 建立LSTM,imput_size是输入数据长度，hidden_size是隐层维度，num_layer是LSTM层数
# dropout用于防止过拟合,以一定概率使神经元失效,bidirectional表示是否开启双向lstm
class LSTM(nn.Module):
    def __init__(self, input_size=input_len, hidden_size=hidden_size, output_size=output_size, num_layer=num_layer,
                 dropout=0.2, bidirectional=False):
        super(LSTM, self).__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, bidirectional=bidirectional)
        if self.bidirectional:
            self.layer2 = nn.Linear(hidden_size * 2, output_size)  # 双向LSTM特征维度要乘2
        else:
            self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.deterministic = True


# 设置随机数种子,使每次运行结果一样
setup_seed(1)

if __name__ == '__main__':
    lstm = LSTM(input_len, hidden_size, output_size, num_layer)

    # 读取数据,取消标题
    data = pd.read_csv('./data.csv', usecols=[1], header=None)

    # 数据预处理
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(data)  # 归一化
    dataset = dataset.astype('float32')

    # 设置X,Y数据集。以look_back=12为准，取第1到第12个为数组，形成data_X,取第13个作为预测值，形成data_Y，完成训练集的提取。
    def create_dataset(dataset, look_back=input_len):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
            # dataY.append(data[i + look_back])
        return np.array(dataX), np.array(dataY)


    # 创建好输入输出
    data_X, data_Y = create_dataset(dataset)
    # print(data_X)
    train_X = data_X
    train_Y = data_Y

    prex = np.array(dataset[-input_len:])  # 用于预测144月数据的输入

    # 设置LSTM能识别的数据类型，形成tran_X的一维两个参数的数组，train_Y的一维一个参数的数组
    train_X = train_X.reshape(-1, 1, input_len)
    train_Y = train_Y.reshape(-1, 1, 1)
    prex = prex.reshape(-1, 1, input_len)

    # 转化为tensor类型
    train_x = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_Y)
    prex = torch.from_numpy(prex)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

    # 开始训练
    for epoch in range(1000):
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        # 前向传播
        out = lstm(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:  # 每 100 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(epoch + 1, loss.item()))

    # 模型预测
    lstm = lstm.eval()  # 转换成测试模式

    data_X = data_X.reshape(-1, 1, input_len)
    data_X = torch.from_numpy(data_X)
    var_data = Variable(data_X)
    pred = lstm(var_data)  # 预测结果

    # 预测144月的结果
    prex = prex.reshape(-1, 1, input_len)
    prex = Variable(prex)
    pred144 = lstm(prex)

    # 将归一化的数据还原
    pred = pred.detach().numpy()
    pred = pred.reshape(pred.shape[0], -1)
    pred = scaler.inverse_transform(pred)

    pred144 = pred144.detach().numpy()
    pred144 = pred144.reshape(pred144.shape[0], -1)
    pred144 = scaler.inverse_transform(pred144)

    print('第144月的预测结果为:', pred144.item())

    # 保存模型
    torch.save(lstm.state_dict(), 'LSTM.pkl')

    # 画图
    month = np.arange(143)  # 原始数据从0开始
    num = np.arange(input_len, 143)  # 预测结果从第12个月开始
    plt.plot(month, data, 'b', label='real')
    plt.plot(num, pred, 'r', label='prediction')
    plt.legend(loc='best')
    plt.show()

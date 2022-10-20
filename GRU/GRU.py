import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random

input_len = 12  # 使用前12个月的数据预测下一个月
hidden_size = 4
num_layers = 1


# 定义GRU
# imput_size是输入数据长度,hidden_size是隐层维度,num_layer是GRU层数
class GRU(nn.Module):
    def __init__(self, input_size=input_len, hidden_size=hidden_size, output_size=1, num_layers=num_layers, dropout=0.2):
        super(GRU, self).__init__()
        self.layer1 = nn.GRU(input_size, hidden_size, num_layers)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.dropout = dropout

    def forward(self, x):
        x, h_n = self.layer1(x)  # output:(batch_size, seq_len, hidden_size)，h0可以直接None
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


# 设置随机数种子
setup_seed(30)

if __name__ == '__main__':
    gru = GRU(input_len, hidden_size, 1, num_layers)

    # 读取数据,取消标题
    data = pd.read_csv('./data.csv', usecols=[1], header=None)

    # 数据预处理
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(data)  # 归一化
    dataset = dataset.astype('float32')

    # 设置X,Y数据集.
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
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)

    # 开始训练
    for epoch in range(1000):
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        # 前向传播
        out = gru(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:  # 每 100 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(epoch + 1, loss.item()))

    # 模型预测
    gru = gru.eval()  # 转换成测试模式

    data_X = data_X.reshape(-1, 1, input_len)
    data_X = torch.from_numpy(data_X)
    var_data = Variable(data_X)
    pred = gru(var_data)  # 预测结果

    # 预测144月的结果
    prex = prex.reshape(-1, 1, input_len)
    prex = Variable(prex)
    pred144 = gru(prex)

    # 将归一化的数据还原
    pred = pred.detach().numpy()
    pred = pred.reshape(pred.shape[0], -1)
    pred = scaler.inverse_transform(pred)

    pred144 = pred144.detach().numpy()
    pred144 = pred144.reshape(pred144.shape[0], -1)
    pred144 = scaler.inverse_transform(pred144)

    print('第144月的预测结果为:', pred144.item())

    # 保存模型
    torch.save(gru.state_dict(), 'GRU.pkl')

    # 画图
    month = np.arange(143)  # 原始数据从0开始
    num = np.arange(input_len, 143)  # 预测结果从第12个月开始
    plt.plot(month, data, 'b', label='real')
    plt.plot(num, pred, 'r', label='prediction')
    plt.legend(loc='best')
    plt.show()

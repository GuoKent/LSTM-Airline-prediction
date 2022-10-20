LSTM.py  是LSTM网络搭建和训练的文件，只需修改num_layer(14行)和bidirectional(21行)两个参数即可。num_layer是LSTM网络                 层数，bidirectional表示是否为双向LSTM网络，True为双向，False为单向。其余参数无需修改，运行后会输出第144月                    的预测结果

GRU.py  是GRU网络搭建和训练的文件，无需修改参数，直接运行即可。运行后会输出第144月的预测结果
data_analize.py  是数据分析的代码，将数据在坐标图中表示出来。

依赖库：
torch=1.11.0+cu113
torchvision=0.12.0+cu113
matplotlib=2.2.3

LSTM.py  数据路径修改在第56行
GRU.py  数据路径修改在第48行
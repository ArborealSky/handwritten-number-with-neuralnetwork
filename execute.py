import mnist_loader
import network
import random
import numpy as np
import figure
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

# 读取数据
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
random.seed(123)  # 伪随机数，种子123,注释掉的话每次运行结果都不一样
np.random.seed(123)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print("开始训练，请等待...")

# 样例
# 建立784*30*10的神经网络，后面一个列表指定每层的激活函数是什么，目前只有两个，最后一个参数指定代价函数

net = network.Network([784, 30, 10], ['sigmoid', 'sigmoid', 'softmax'], cost=network.CrossEntropyCost)
net.large_weight_initializer()  # 比较不好初始化方法，注释掉的话会默认选择方差为1，均值为0，初始权值较小的正态分布初始化
epoch = 30
net.SGD(training_data, epoch, 1, 0.5, 5.0, 
        evaluation_data=test_data, 
        monitor_evaluation_cost=True,  # 因为绘图的时候需要用到两类数据集的代价和准确率，所以需要监控
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)  # 小批量梯度下降法，包含迭代测试结果的输出
net.save('model.json') # 保存模型

figure.main('result.json', epoch, 50000)  # 画图


# net = network.load('model.json') # 加载模型
# net.test_model() # 测试模型

print("训练结束！")



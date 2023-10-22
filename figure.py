import json
import random

import mnist_loader
import network

import matplotlib.pyplot as plt
import numpy as np


def main(filename, num_epochs,
        training_set_size=1000):

    # run_network(filename, num_epochs, training_set_size, lmbda) 跑测试样例
    make_plots(filename, num_epochs, training_set_size)
                    
def run_network(filename, num_epochs, training_set_size=1000, lmbda=0.0):


    random.seed(123)  # 伪随机数，种子123
    np.random.seed(123)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10],['sigmoid', 'sigmoid', 'softmax'] ,cost=network.CrossEntropyCost())
    net.large_weight_initializer()

    test_cost, test_accuracy, training_cost, training_accuracy = net.SGD(
                list(training_data)[:training_set_size], num_epochs, 10, 0.5,
                evaluation_data=test_data, lmbda = lmbda,
                monitor_evaluation_cost=True, 
                monitor_evaluation_accuracy=True, 
                monitor_training_cost=True, 
                monitor_training_accuracy=True)
    
    f = open(filename, "w")
    json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)  # 
    f.close()

def make_plots(filename, num_epochs, 
            training_set_size=1000):

    f = open(filename, "r")
    test_cost, test_accuracy, training_cost, training_accuracy = json.load(f)
    f.close()

    plot_cost_overlay(training_cost, test_cost, num_epochs)
    plot_overlay(test_accuracy, training_accuracy, num_epochs, training_set_size)

def plot_cost_overlay(training_cost, test_cost, num_epochs):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 设置横坐标轴范围为 xmin 到 num_epochs
    ax.set_xlim([0, num_epochs])


    # 绘制测试成本曲线，确保测试成本列表的长度与 num_epochs 相等
    ax.plot(np.arange(0, num_epochs), 
            [cost/10000 for cost in test_cost],
            color='#FFA933',
            label="test data cost")


    # 绘制训练成本曲线，确保训练成本列表的长度与 num_epochs 相等
    ax.plot(np.arange(0, num_epochs), 
            [cost/10000 for cost in training_cost],
            color='#2A6EA6',
            label="training data cost")
    
    ax.grid(True)
    ax.set_xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.show()


# 绘制测试准确率和训练准确率曲线
def plot_overlay(test_accuracy, training_accuracy, num_epochs, 
                training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, num_epochs), 
            [accuracy/100.0 for accuracy in test_accuracy], 
            color='#FFA933',
            label="test data accuracy")
    
    ax.plot(np.arange(0, num_epochs), 
            [accuracy*100.0/training_set_size 
            for accuracy in training_accuracy], 
            color='#2A6EA6',
            label="training data accuracy")
    
    ax.grid(True)
    ax.set_xlim([0, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([10, 100])
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    filename = input("输入保存结果的文件名称(建议result.json): ")
    num_epochs = int(input(
        "输入想要运行的epoch数: "))
    training_set_size = int(input(
        "输入训练集的采用的数据规模(建议1000): "))
    lmbda = float(input(
        "输入正则化参数,lambda(建议5.0): "))
    main(filename, num_epochs, training_set_size, lmbda)

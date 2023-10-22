import matplotlib.pyplot as plt
import numpy as np

def start():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(-3, 3, 0.1)

    ax.plot(x, sigmoid(x), color='#1F77B4', label="sigmoid")
    ax.plot(x, tanh(x), color='#ff7f0e', label="tanh")
    ax.plot(x, relu(x), color='#2ca02c', label="relu")
    ax.plot(x, softplus(x), color='#ffbb78', label="softplus")
    ax.plot(x, ELU(x), color='#9467bd', label="ELU")
    
    ax.grid(True, linewidth=0.5)
    ax.set_xlim([-3, 3])
    ax.set_xlabel('x')
    ax.set_ylim([-3, 3])
    ax.set_ylabel('activation function')
    plt.legend(loc="lower right")
    plt.show()

def sigmoid(x):
    # 直接返回sigmoid函数
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    e_x = np.exp(x)
    e_minusx = np.exp(-x)
    return (e_x - e_minusx) / (e_x + e_minusx)

def relu(x):
    return np.maximum(0, x)

def softplus(x):
    return np.log(1 + np.exp(x))

def ELU(x, alpha=4.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

if __name__ == "__main__":
    start()
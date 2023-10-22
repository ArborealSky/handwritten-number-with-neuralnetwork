import random        # 导入random模块
import numpy as np   # 导入numpy模块
import json          # 导入json模块
import sys           # 导入sys模块
import mnist_loader # 导入mnist_loader模块


# 定义损失函数

# 1.均方误差
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        
        return (a-y) * sigmoid_prime(z)


# 2.交叉熵
class CrossEntropyCost(object):

    @staticmethod
    def fn(p, y):

        return -np.sum(y * np.log(p))


    @staticmethod  # 此处激活函数的值z,没有用到，但为了保证函数接口统一必须写
    def delta(z, a, y):

        return (a-y)


# 定义神经网络类
class Network(object):

    def __init__(self, sizes, activate_function, cost=CrossEntropyCost):

        self.num_layers = len(sizes)                 # 神经网络的层数
        self.sizes = sizes                           # 每层神经元的个数
        self.activate_function = activate_function   # 选择每层激活函数
        self.default_weight_initializer()            # 默认权值初始化方法
        self.cost=cost                               # 选择损失函数

    # 默认权值初始化方法，均值为0，方差为1的正态分布，整体权重w除以根号下输入神经元个数，数值较小
    def default_weight_initializer(self):

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # 不好的权值初始化方法，均值为0，方差为1的正态分布，数值较大
    def large_weight_initializer(self):

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # 前向传播
    def feedforward(self, a):

        for b, w ,activate_function in zip(self.biases, self.weights, self.activate_function):
            if activate_function == 'sigmoid':
                a = sigmoid(np.dot(w, a)+b)
            elif activate_function == 'softmax':
                a = softmax(np.dot(w, a)+b)
            
        return a

    # 随机梯度下降法
    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)


        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        # 迭代训练,每次迭代都会打乱训练集,mini_batch_size为小批量梯度下降法的批量大小
        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+mini_batch_size]for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print('*'*40)
            print("Epoch %s 训练完成" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("训练集损失: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("训练集准确率: {:.4f}".format(accuracy*1.0 / n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("测试集损失: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("测试集准确率: {:.4f}".format(accuracy * 1.0 /n_data))

            print('*'*40 +'\n')

        # 保存每次epoch的结果
        f = open('result.json', "w")
        json.dump([evaluation_cost, evaluation_accuracy, training_cost, training_accuracy], f)
        f.close()
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    # 更新权值和偏置,eta为学习率，lmbda为正则化参数，n为训练集大小
    def update_mini_batch(self, mini_batch, eta, lmbda, n):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw  # 此处采用L2正则化
                        for w, nw in zip(self.weights, nabla_w)]


        # self.weights = [w-(eta/len(mini_batch))*nw-eta*lmbda*np.sign(w)/len(mini_batch)
        #                 for w, nw in zip(self.weights, nabla_w)]  # 此处采用L1正则化

        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    # 反向传播算法求梯度
    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # 前向传播
        activation = x
        activations = [x] 
        zs = [] 

        for b, w , activate_function in zip(self.biases, self.weights, self.activate_function):
            z = np.dot(w, activation)+b
            zs.append(z)

            if activate_function == 'sigmoid':
                activation = sigmoid(z)
            elif activate_function == 'softmax':
                activation = softmax(z)
                
            activations.append(activation)
        
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 反向传播，从倒数第二层开始
        for l in range(2, self.num_layers):
            z = zs[-l]
            if activate_function == 'sigmoid':
                sp = sigmoid_prime(z)
            elif activate_function == 'softmax':
                sp = softmax_prime(z)

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return (nabla_b, nabla_w)

    # 计算准确率
    def accuracy(self, data, convert=False):

        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))  # np.argmax返回最大值的索引,即返回概率最大的那个数字
                        for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    # 计算损失函数
    def total_cost(self, data, lmbda, convert=False):
        
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)

            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)  # 此处采用L2正则化
            # cost += (lmbda/len(data))*sum(np.sum(np.abs(w)) for w in self.weights)  # 此处采用L1正则化
        return cost

    # 保存神经网络参数到json文件
    def save(self, filename):
            
                data = {"sizes": self.sizes,
                        "weights": [w.tolist() for w in self.weights],
                        "biases": [b.tolist() for b in self.biases],
                        "cost": str(self.cost.__name__)}
                f = open(filename, "w")
                json.dump(data, f)
                f.close()

    def test_model():
        net = load('model.json')
        test_data = mnist_loader.load_data_wrapper()[2]
        print("测试集准确率: {:.4f}".format(net.accuracy(test_data)*1.0/len(test_data)))
    
# 从文件中加载神经网络参数
def load(filename):
    
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])  # 得到损失函数的名字
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net



# 将标签转换为向量,例如：5->[0,0,0,0,0,1,0,0,0,0]
def vectorized_result(j):

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# 激活函数

# sigmoid函数及其导数
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


# softmax函数及其导数
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # 防止溢出
    return exp_z / exp_z.sum(axis=0, keepdims=True)

def softmax_prime(z):
    return softmax(z)*(1-softmax(z))

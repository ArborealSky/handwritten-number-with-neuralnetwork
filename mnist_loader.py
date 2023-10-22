# 导入第三方库
import pickle       # pickle模块实现了基本的数据序列和反序列化
import gzip         # gzip模块为对数据进行读写的压缩和解压缩
import numpy as np  # numpy是一个运行速度非常快的数学库，主要用于数组计算

# 加载数据集
def load_data():
    
    f = gzip.open('mnist.pkl.gz', 'rb')  # 读取数据集,rb表示以二进制格式打开文件
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")  # 将数据集解压缩，并用pickle模块将解压后的数据集保存到对应的变量中
    f.close()
    return (training_data, validation_data, test_data)

# 将数据集进行重新排列
def load_data_wrapper():
    
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

# 将标签转换为向量,例如：5->[0,0,0,0,0,1,0,0,0,0]
def vectorized_result(j):
    
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

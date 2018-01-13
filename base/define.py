import numpy as np

# 激活函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

class Neuron(object):
    # 神经元
    def __init__(self, feature_size, active_func=sigmoid):
        self.feature_size = feature_size
        self.active_func = active_func
        # 每个输入的权重
        self.weights = np.random.rand(self.feature_size)
        self.output = 0
        self.output_before_act = 0
        # 后向传播的时候记录误差
        self.error = 0

    # 经过激活函数之后的输出
    def calc_output(self, input):
        self.input = input
        self.output_before_act = np.dot(self.input, self.weights)
        self.output = self.active_func(self.output_before_act)
        return self.output

    # 根据误差更新当前的 weights
    def update_weights(self, pre_output, learn_rate):
        for i in range(0, self.feature_size):
            delta = self.error * sigmoid_derivative(self.output_before_act)
            #print ("delta: %f, learn_rate: %f, pre_output[i]: %f" % (delta, learn_rate, pre_output[i]))
            self.weights[i] += delta * learn_rate * pre_output[i]

class Layer(object):
    def __init__(self, neuron_num):
        # 当前层的神经元
        self.neuron_list = []
        # 当前层神经元个数
        self.neuron_num = neuron_num

    def calc_output(self, pre_output):
        for neuron in self.neuron_list:
            neuron.calc_output(pre_output)

    def layer_output(self):
        output = np.ones(self.neuron_num)
        for i in range(0, self.neuron_num):
            output[i] = self.neuron_list[i].output
        return output

    def add_neuron(self, neuron):
        self.neuron_list.append(neuron)

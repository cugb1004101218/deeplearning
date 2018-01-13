import sys
import logging
import math
import numpy as np
sys.path.append("..")
from base import define

class FullyConnectedNetwork(object):
    def __init__(self, layer_num, learn_rate):
        self.layer_num = layer_num
        self.layer_list = []
        self.train_data = []
        self.learn_rate = learn_rate

    def add_layer(self, layer):
        if self.layer_num <= len(self.layer_list):
            logging.error("the network has too many layers")
            return
        self.layer_list.append(layer)

    def add_train_data(self, tensor, label):
        self.train_data.append((tensor, label))

    def calc_error(self, output, label):
        output_layer = self.layer_list[-1]
        for i in range(0, output_layer.neuron_num):
            output_layer.neuron_list[i].error = label[i] - output_layer.neuron_list[i].output
        for i in range(self.layer_num - 2, 0, -1):
            self.calc_layer_error(i)

    def calc_layer_error(self, layer_index):
        pre_layer = self.layer_list[layer_index + 1]
        now_layer = self.layer_list[layer_index]
        for i in range(0, now_layer.neuron_num):
            now_layer.neuron_list[i].error = 0
            for pre_layer_neuron in pre_layer.neuron_list:
                now_layer.neuron_list[i].error += pre_layer_neuron.error * pre_layer_neuron.weights[i]

    def calc_output(self, input):
        for i in range(0, self.layer_list[0].neuron_num):
            self.layer_list[0].neuron_list[i].output = input[i]
        pre_layer = self.layer_list[0].layer_output()
        for layer in self.layer_list[1:]:
            layer.calc_output(pre_layer)
            pre_layer = layer.layer_output()

    def train_a_round(self):
        for data in self.train_data:
            self.calc_output(data[0])
            label = data[1]
            pre_layer = self.layer_list[-1]
            self.calc_error(pre_layer, label)
            for i in range(self.layer_num - 1, 0, -1):
                pre_output = self.layer_list[i - 1].layer_output()
                for neuron in self.layer_list[i].neuron_list:
                    neuron.update_weights(pre_output, self.learn_rate)

    def calc_loss(self):
        loss = 0.0
        for data in self.train_data:
            self.calc_output(data[0])
            label = data[1]
            pre_layer = self.layer_list[-1].layer_output()
            for i in range(pre_layer.ndim):
                loss += math.pow(pre_layer[i] - label[i], 2)
                print("output: %f, label: %f" %  (pre_layer[i], label[i]))
        print ("loss " + str(loss))
        #for layer in self.layer_list[-1:]:
        #    for neuron in layer.neuron_list:
        #        print("weights: " + str(neuron.weights))
        #        #print("output: " + str(neuron.output))
    def train(self, iter_num):
        for i in range(0, iter_num):
            self.train_a_round()
            if i % 1000 == 0:
                self.calc_loss()

if __name__ == '__main__':
    fully_connected_network = FullyConnectedNetwork(3, 0.01)
    input_layer = define.Layer(2)
    for i in range(0, input_layer.neuron_num):
        input_layer.add_neuron(define.Neuron(2))
    hide_layer = define.Layer(3)
    for i in range(0, hide_layer.neuron_num):
        hide_layer.add_neuron(define.Neuron(2))
    output_layer = define.Layer(1)
    for i in range(0, output_layer.neuron_num):
        output_layer.add_neuron(define.Neuron(3))
    fully_connected_network.add_layer(input_layer)
    fully_connected_network.add_layer(hide_layer)
    fully_connected_network.add_layer(output_layer)
    fully_connected_network.add_train_data(np.array([1, 1]), np.array([0]))
    fully_connected_network.add_train_data(np.array([1, 0]), np.array([1]))
    fully_connected_network.add_train_data(np.array([0, 1]), np.array([1]))
    fully_connected_network.add_train_data(np.array([0, 0]), np.array([0]))
    fully_connected_network.train(1000000)

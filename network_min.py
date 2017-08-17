#https://github.com/WilliamLewww/ML-Perceptron
#Fully Connected Recurrent Neural Network
#by William Lew
import numpy as np
import math
def __logsig(value, derivative = 0):
    if derivative == 1:
        return value * (1 - value)
    return 1 / (1 + np.exp(-value))
def generate_array(array, output = 0):
    if output == 1:
        return np.array(array).T
    return np.array(array)
def generate_synapse(array):
    temp_list = []
    for x in range(1, len(array)):
        temp_list.append(2*np.random.random((array[x - 1], array[x])) - 1)
    return temp_list
def feedforward_and_backpropagate(loop_count, synapse_list, input_case, output_case):
    for j in xrange(loop_count):
        layer_list = [input_case]
        for x in range(len(synapse_list)):
            layer_list.append(__logsig(np.dot(layer_list[x], synapse_list[x])))
        layer_delta_list = [(output_case - layer_list[len(layer_list) - 1]) * __logsig(layer_list[len(layer_list) - 1], 1)]
        for x in range(len(synapse_list) - 2, -1, -1):
            layer_delta_list.append(layer_delta_list[len(layer_delta_list) - 1].dot(synapse_list[x + 1].T) * __logsig(layer_list[x + 1], 1))
        for x in range(len(synapse_list)):
            synapse_list[x] += layer_list[x].T.dot(layer_delta_list[len(synapse_list) - x - 1])
    return layer_list[len(layer_list) - 1]

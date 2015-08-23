import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

import csv
import time

from numpy import genfromtxt

plt.ion()

class NodePatch(mpl.patches.Ellipse):
    def __init__(self, node, xy, width, height, **kwargs):
        self.node = node
        self.x = xy[0]
        self.y = xy[1]
        mpl.patches.Ellipse.__init__(self, xy, width, height, **kwargs)

class EdgeLine(mpl.lines.Line2D):
    def __init__(self, edge, x1, y1, x2, y2, **kwargs):
        self.edge = edge
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        mpl.lines.Line2D.__init__(self, [x1, y1], [x2, y2],
                 **kwargs)


class Node(object):
    #abstract base class
    pass

class Edge(object):
    #abstract base class
    pass

class Leaf(Node):
    pass

class Synapse(Edge):
    def __init__(self, presyn_node, weight):
        self.node = presyn_node
        self.weight = weight

        self.line = None


class Neuron(Node):
    def __init__(self, transfer_func, learning_rate):

        self.activation = 0
        self.in_synapses = []
        self.transfer = transfer_func

        #Links to the patch object created from this neuron.
        self.patch = None

        #Attributes used in learning
        self.rate = learning_rate
        self.error = 0.0

        self.coords = []


        self.__str__ = self.__repr__()

    def sum_inputs(self):
        val = 0.0
        for synapse in self.in_synapses:
            val += synapse.weight * synapse.node.activation

        self.activation = self.transfer(val)

    def __repr__(self):
        if self.coords != []:
            return 'Neuron object at (layer: %s, position: %s)' %(self.coords[0], self.coords[1])
        else:
            return 'Neuron object at (no network)'


class NeuralNet(object):
    def __init__(self, layer_specs):
        self.depth = len(layer_specs)
        if self.depth < 3:
            raise ValueError("Network must consist of at least 3 layers.")

        self.layers = []

        #Arrange all the nodes in their layers.
        afferent_layer = None
        for i, num_nodes in enumerate(layer_specs):
            layer = []
            for j in range(num_nodes):
                neuron = Neuron(sigmoid, 0.6)
                neuron.coords = (i, j)

                #Unless this is the input layer, connect each neuron to every neuron on the afferent layer.
                if i != 0:
                    neuron.in_synapses = [Synapse(node, np.random.randn())
                                          for node in self.layers[i-1]]

                layer.append(neuron)

            self.layers.append(layer)

    def feedforward(self, input_vector):
        if len(input_vector) != len(self.layers[0]):
            raise ValueError("Mismatch in length of input layer and input vector.")

        for i in range(len(input_vector)):
            self.layers[0][i].activation = input_vector[i]

        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.sum_inputs()

        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.activation)

        return output

    def backpropagate(self, target_vector):
        if len(target_vector) != len(self.layers[-1]):
            raise ValueError("Mismatch in length of output layer and target vector.")

        #Update output weights.
        for i in range(len(target_vector)):
            neuron = self.layers[-1][i]
            neuron.error = neuron.activation*(1 - neuron.activation)*(target_vector[i] - neuron.activation)

            for synapse in neuron.in_synapses:
                synapse.weight += neuron.rate * neuron.error * synapse.node.activation
                try:
                    synapse.line.set_linewidth(2*abs(synapse.weight))
                    if synapse.weight < 0:
                        synapse.line.set_color([0, 0, 1])
                    else:
                        synapse.line.set_color([1, 0, 0])
                except AttributeError:
                    pass

        #Update weights for all hidden layers.
        reversed_layers = list(reversed(self.layers))
        for layer in reversed_layers[1:-1]:
            efferent_layer = reversed_layers[reversed_layers.index(layer) - 1]

            for neuron in layer:
                #Sum the error across downstream neurons connected to this one.
                sum_eff_error = 0.0
                for eff_node in efferent_layer:
                    if synapse in eff_node.in_synapses:
                        sum_eff_error += eff_node.error*synapse.weight

                neuron.error = neuron.activation*(1 - neuron.activation)*sum_eff_error

                #Update weights
                for synapse in neuron.in_synapses:
                    synapse.weight += neuron.rate * neuron.error * synapse.node.activation
                    try:
                        synapse.line.set_linewidth(2*abs(synapse.weight))
                        if synapse.weight < 0:
                            synapse.line.set_color([0, 0, 1])
                        else:
                            synapse.line.set_color([1, 0, 0])
                    except AttributeError:
                        pass

class Visualizer(object):
    def __init__(self, model):
        self.fig, self.ax = plt.subplots(1,1)
        self.fig.canvas.mpl_connect("pick_event", self.pick_handler)

        self.model = model

        depth = model.depth
        network_width = max([len(x) for x in model.layers])

        self.ax.set_xlim((-network_width/2, network_width/2))
        self.ax.set_ylim((-0.5, depth-0.5))

        for layer in model.layers:
            for node in layer:
                [x, y] = self.coords_to_pos(node.coords, layer)
                node.patch = NodePatch(node, [x, y], 0.3, 0.3,
                                       edgecolor='k', facecolor= 'w', linewidth= 2, zorder= 10)
                h = self.ax.add_artist(node.patch)
                h.set_picker(1.5)

                for synapse in node.in_synapses:
                    if np.sign(synapse.weight) < 0:
                        color = [0, 0, 1]
                    else:
                        color = [1, 0, 0]

                    synapse.line = EdgeLine(synapse, h.x, synapse.node.patch.x, h.y, synapse.node.patch.y,
                                            linewidth= abs(synapse.weight), color= color, zorder= 1)

                    self.ax.add_artist(synapse.line)


    def coords_to_pos(self, coords, layer):
        y = coords[0]
        x = coords[1] - (len(layer)-1)/2
        return [x, y]

    def pick_handler(self, ev):
        print("artist:")
        print(ev.artist)
        print("type:")
        print(type(ev.artist))


def sigmoid(x):
    return 1/(1 + math.e**(-x))

def load_data(filename, target_length = None):
    obs_mat = []
    labels = []

    with open(filename) as f:
        for line in f:

            list_line = line.strip().split(',')
            if list_line[0] == '':
                continue

            if target_length is not None:
                labels.append(list_line[-1])
                list_line = list_line[0:-1]

            row = [float(x) for x in list_line]
            obs_mat.append(row)

    if target_length is not None:
        classes = set(labels)
        class_codes = dict(zip(classes, np.eye(target_length).tolist()))

        targets = []
        for label in labels:
            targets.append(class_codes[label])

        targets = np.array(targets)

        return [obs_mat, targets]

    return obs_mat

[X, targets] = load_data('iris_data.txt', target_length= 3)

nn = NeuralNet([4, 5, 3])

vis = Visualizer(nn)

for pss in range(25):
    print(pss)
    for i, instance in enumerate(X):
        plt.draw()
        time.sleep(0.1)
        nn.feedforward(instance)
        nn.backpropagate(targets[i])

halt = True
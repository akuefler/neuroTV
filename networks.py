import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

class NodePatch(mpl.patches.Ellipse):
    def __init__(self, node, xy, width, height, **kwargs):
        self.node = node
        self.x = xy[0]
        self.y = xy[1]
        mpl.patches.Ellipse.__init__(self, xy, width, height, **kwargs)


class Node(object):
    #abstract base class
    pass

class Leaf(Node):
    pass

class Neuron(Node):
    def __init__(self, transfer_func, learning_rate):

        self.activation = 0
        self.in_synapses = {}
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
        for node, weight in self.in_synapses.items():
            val += weight*node.activation

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
                neuron = Neuron(sigmoid, 0.95)
                neuron.coords = (i, j)

                #Unless this is the input layer, connect each neuron to every neuron on the afferent layer.
                if i != 0:
                    neuron.in_synapses = dict(zip(self.layers[i - 1],
                                                  np.random.randn(1, len(self.layers[i - 1])).tolist()[0]))

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

            for in_node, weight in neuron.in_synapses.items():
                neuron.in_synapses[in_node] = weight + neuron.rate * neuron.error * in_node.activation

        #Update weights for all hidden layers.
        reversed_layers = list(reversed(self.layers))
        for layer in reversed_layers[1:-1]:
            efferent_layer = reversed_layers[reversed_layers.index(layer) - 1]

            for neuron in layer:

                #Sum the error across downstream neurons connected to this one.
                sum_eff_error = 0.0
                for eff_node in efferent_layer:
                    if neuron in eff_node.in_synapses.keys():
                        sum_eff_error += eff_node.error*eff_node.in_synapses[neuron]

                neuron.error = neuron.activation*(1 - neuron.activation)*sum_eff_error

                #Update weights
                for aff_node, weight in neuron.in_synapses.items():
                    neuron.in_synapses[aff_node] = weight + neuron.rate * neuron.error * aff_node.activation

class Visualizer(object):
    def __init__(self, model):
        self.fig, self.ax = plt.subplots(1,1)
        self.fig.canvas.mpl_connect("pick_event", self.pick_handler)

        self.model = model

        depth = model.depth
        network_width = max([len(x) for x in model.layers])

        for layer in model.layers:
            for node in layer:
                [x, y] = self.coords_to_pos(node.coords, layer)
                node.patch = NodePatch(node, [x, y], 0.3, 0.3)
                h = self.ax.add_artist(node.patch)
                h.set_picker(1.5)

                for aff_node, weight in node.in_synapses.items():
                    if np.sign(weight) < 0:
                        color = [0, 0, 1]
                    else:
                        color = [1, 0, 0]

                    l = self.ax.plot([h.x, aff_node.patch.x], [h.y, aff_node.patch.y], color= color)
                    #l.set_picker(1.5)




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

nn = NeuralNet([4, 5, 3])

feature_vec = [0.3, 2, 0.6, 0.6]

vis = Visualizer(nn)


halt = True

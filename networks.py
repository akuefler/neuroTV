import numpy as np
import numpy.matlib as ml
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

import csv
import time

from numpy import genfromtxt

plt.ion()

class NodeArt(mpl.lines.Line2D):
    def __init__(self, node, xy, **kwargs):
        self.node = node
        self.x = xy[0]
        self.y = xy[1]

        if node.bias == True:
            marker = 'h'
            ms = 25
        else:
            marker = 'o'
            ms = 35

        mpl.lines.Line2D.__init__(self, [self.x], [self.y], marker= marker,
                                  markeredgewidth= 2.5, markersize= ms, **kwargs)


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
    def __init__(self, transfer_func, learning_rate, bias = False):

        self.activation = 0
        self.in_synapses = []
        self.transfer = transfer_func
        self.bias = bias

        #Will contain keys for basis of decision hyperplane, offset of plane, and points on the plane.
        self.boundary = None

        #Links to the artist created from this neuron.
        self.art = None

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
    def __init__(self, layer_specs, bias_on= False):
        self.depth = len(layer_specs)
        if self.depth < 3:
            raise ValueError("Network must consist of at least 3 layers.")

        self.layers = []
        self.bias_on = bias_on

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

        #Include bias nodes.
        if bias_on:
            for layer in self.layers[0:-1]:
                bias = Neuron(sigmoid, 0.6, bias= True)
                bias.activation = 1
                bias.coords = (self.layers.index(layer), len(layer))

                layer.append(bias)
                for neuron in self.layers[self.layers.index(layer) + 1]:
                    neuron.in_synapses.append(Synapse(bias, np.random.randn()))


    def feedforward(self, input_vector):

        if len(input_vector) + int(self.bias_on) != len(self.layers[0]):
            raise ValueError("Mismatch in length of input layer and input vector.")

        for i in range(len(input_vector)):
            self.layers[0][i].activation = input_vector[i]

        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.sum_inputs()

        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.activation)

        output = np.array(output)
        return output

    def backpropagate(self, target_vector, show= False):
        if len(target_vector) != len(self.layers[-1]):
            raise ValueError("Mismatch in length of output layer and target vector.")

        #Update output weights.
        for i in range(len(target_vector)):
            neuron = self.layers[-1][i]
            neuron.error = neuron.activation*(1 - neuron.activation)*(target_vector[i] - neuron.activation)

            for synapse in neuron.in_synapses:
                synapse.weight += neuron.rate * neuron.error * synapse.node.activation
                if show:
                    self.update_weight_graphics(synapse)

            if not neuron.bias:
                self.update_DB_vecs(neuron)

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
                    if show:
                        self.update_weight_graphics(synapse)

                if not neuron.bias:
                    self.update_DB_vecs(neuron)

    def update_weight_graphics(self, synapse):
        synapse.line.set_linewidth(1*abs(synapse.weight))
        if synapse.weight < 0:
            synapse.line.set_color([0, 0, 1])
        else:
            synapse.line.set_color([1, 0, 0])

    def update_DB_vecs(self, neuron):
        assert not neuron.bias, ("Bias nodes do not have decision boundaries.")
        neuron.boundary = {}

        n = 0
        for synapse in neuron.in_synapses:
            if not synapse.node.bias:
                n += 1

        w1 = neuron.in_synapses[0].weight
        w= []
        for syn in neuron.in_synapses[1:]:
            if syn.node.bias:
                offset = np.zeros(n)
                ###ISSUE: Might try ADDING syn.weight (bias) as well.
                offset[0] = (logit(0.5)-syn.weight)/w1

                #neuron.boundary['offset'] = offset
                continue
            w.append(syn.weight/w1)

        span = np.row_stack((w, np.eye(n - 1)))
        basis, R = np.linalg.qr(span)

        #neuron.boundary= {'basis':basis, 'span': span}
        neuron.boundary= {'basis':basis, 'span': span, 'offset':offset}

class DataDisplay(object):
    def __init__(self, visualizer, X= None):
        self.fig, ax = plt.subplots(1, 1)
        self.ax = ax

        self.X = X

        self.vis = visualizer

        self.widgets= {}

        rand_button = mpl.widgets.Button(plt.axes([0.86, 0.02, 0.12, 0.03]), 'Randomize')
        self.widgets['rand'] = rand_button
        self.widgets['rand'].on_clicked(self.random_proj)

        if self.X is not None:
            self.setup_DataDisplay()
            self.draw_data()

    def random_proj(self, ev):
        ##ISSUE: Should have some way to view projection onto the standard basis vectors.
        #if len(self.X[0,:]) > 2:
        if len(self.D[0]['data'][0]) > 2:
            #Z = np.random.rand(len(X[0,:]), 2)
            Z = np.random.rand(len(self.D[0]['data'][0]), 2)
            Q, R = np.linalg.qr(Z)
            #D = np.dot(X, Q)
            self.P = Q

        self.draw_data()

    def add_data(self, X= None, targets= None):
        self.X = X
        self.targets = targets

    def draw_data(self):
        self.ax.cla()

        for d in self.D.values():
            X_lo = np.dot(d['data'], self.P)
            self.ax.scatter(X_lo[:, 0], X_lo[:, 1], color= d['color'])

        #if self.vis.selected_node is not None:
            #self.vis.draw_DB_points(self.vis.selected_node.node)

        self.fig.canvas.draw()

    def setup_DataDisplay(self):
        self.D = []

        #Prepare Q
        if len(self.X[0,:]) > 2:
            Z = np.random.rand(len(self.X[0,:]), 2)
            Q, R = np.linalg.qr(Z)
            #D = np.dot(X, Q)
            self.P = Q
        else:
            self.P = np.eye(2)

        #Prepare color dictionary
        self.D = {}

        for i, row in enumerate(self.X):
            clss = np.where(self.targets[i] > 0)[0][0]
            if clss not in self.D.keys():
                self.D[clss] = {}
                self.D[clss]['data'] = []

            self.D[clss]['data'].append(row)


        for clss in range(len(self.targets[0])):
            ##ISSUE: Will only work if target vectors have only one 1 and rest 0s.
            self.D[clss]['color'] = [np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]
            self.D[clss]['data'] = np.array(self.D[clss]['data'])

        self.draw_data()


class Visualizer(object):
    def __init__(self, model):
        self.fig, ax = plt.subplots(1, 1)
        self.ax = ax

        #Initialize display modules
        self.dDisplay = DataDisplay(self)
        ##self.nDisplay = NetworkDisplay(self) #Not yet implemented.

        self.fig.canvas.mpl_connect("pick_event", self.pick_handler)

        self.model = model

        self.selected_node = None

        depth = model.depth
        network_width = max([len(x) for x in model.layers])

        self.ax.set_xlim((-network_width/2, network_width/2))
        self.ax.set_ylim((-0.5, depth-0.5))

        for layer in model.layers:
            for node in layer:
                [x, y] = self.coords_to_pos(node.coords, layer)
                node.art = NodeArt(node, [x, y],
                                       color='k', markerfacecolor= 'w', zorder= 10)
                h = self.ax.add_artist(node.art)
                h.set_picker(3.5)

                for synapse in node.in_synapses:
                    if np.sign(synapse.weight) < 0:
                        color = [0, 0, 1]
                    else:
                        color = [1, 0, 0]

                    synapse.line = EdgeLine(synapse, h.x, synapse.node.art.x, h.y, synapse.node.art.y,
                                            linewidth= abs(synapse.weight), color= color, zorder= 1)

                    self.ax.add_artist(synapse.line)


    def learn(self, X, targets, iterations, interval = 5):
        self.dDisplay.add_data(X, targets)
        self.dDisplay.setup_DataDisplay()

        ##Train
        for it in range(1, iterations+1):
            shuff_X, shuff_tars = shuffle_in_unison_inplace(X, targets)

            for i, instance in enumerate(shuff_X):
                self.model.feedforward(instance)
                self.model.backpropagate(shuff_tars[i], show= False)

            if it % interval == 0:
                self.model.backpropagate(shuff_tars[i], show= True)
                time.sleep(0.1)
                self.fig.canvas.draw()

    def change_space(self, layer):
        ##ISSUE: Because this appends a 1 vector each time, only goes up in dimensionality. Never down.
        for key, d in self.dDisplay.D.items():
            self.dDisplay.D[key]['data'] = np.column_stack((d['data'], np.ones((len(d['data']), 1))))

        self.dDisplay.P = []
        W = []

        for node in layer:
            row = []
            if node.bias:
                continue

            for syn in node.in_synapses:
                row.append(syn.weight)

            W.append(row)

        W = np.array(W)

        #ISSUE: Should I project onto W, or an orthonormalized W?
        self.dDisplay.P = W.transpose()

        self.dDisplay.draw_data()







    def draw_DB_points(self, node):
        M = np.dot(node.boundary['pts'].transpose(), self.dDisplay.P)

        if hasattr(self, 'plane'):
            try:
                self.plane.remove()
            except ValueError:
                pass

        #self.plane = self.data_ax.scatter(M[:,0], M[:, 1], color = 'y')
        self.plane = mpl.lines.Line2D(M[:,0], M[:, 1], color = 'y', marker= '.', linestyle= '.')
        self.dDisplay.ax.add_artist(self.plane)

    def compute_DB_points(self, node, with_offset= True):
        B = node.boundary['basis']

        n = np.shape(B)[1]
        m = 15
        #m = 100 #Number of points to be drawn

        M = []

        #Generates permutation matrix whose elements are scaling constants for the basis vectors.
        C = np.indices((m,) * n).reshape(n, -1).T - np.ceil(m/2)
        C = np.array(C)

        M = np.array(M)
        #pts = np.dot(B, M)

        #pts = np.dot(M, B.transpose())
        pts = np.dot(C, B.transpose()).transpose()

        if with_offset:
            O = ml.repmat(node.boundary['offset'], np.shape(pts)[1], 1).transpose()
        else:
            O = np.zeros(np.shape(pts))

        node.boundary['pts'] = pts + O


    def coords_to_pos(self, coords, layer):
        y = coords[0]
        x = coords[1] - (len(layer)-1)/2
        return [x, y]

    def pick_handler(self, ev):
        self.selected_node = ev.artist
        self.selected_node.set_markerfacecolor('y')

        if self.selected_node.node.boundary is not None:
            #self.compute_DB_points(self.selected_node.node)
            #self.draw_DB_points(self.selected_node.node)
            layer = self.model.layers[self.selected_node.node.coords[0]]
            self.change_space(layer)

        if hasattr(self, 'last_selected_node'):
            self.last_selected_node.set_markerfacecolor('w')

        self.last_selected_node = self.selected_node


def sigmoid(x):
    return 1/(1 + math.e**(-x))

def logit(x):
    return np.log(x/(1 - x))

def center(X):
    for row in X:
        row -= np.mean(X)

    return X

def load_data(filename, labeled = True):
    obs_mat = []
    labels = []

    with open(filename) as f:
        for line in f:

            list_line = line.strip().split(',')
            if list_line[0] == '':
                continue

            if labeled:
                labels.append(list_line[-1])
                list_line = list_line[0:-1]

            row = [float(x) for x in list_line]
            obs_mat.append(row)

    if labeled:
        classes = set(labels)
        class_codes = dict(zip(classes, np.eye(len(classes)).tolist()))

        targets = []
        for label in labels:
            targets.append(class_codes[label])

        targets = np.array(targets)

        return [obs_mat, targets]

    return obs_mat

def decide(vec):
    out= (vec > 0.5).astype(int)
    return out

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

[X, targets] = load_data('two_class_2D.txt', labeled= True)

nn = NeuralNet([2, 3, 2], bias_on= True)

vis = Visualizer(nn)

#X = np.array(X)[:, 2:4]
X = np.array(X)
X = center(X)

##Train
vis.learn(X, targets, 15, 5)

##Test
correct = 0
for i, instance in enumerate(X):
    out = nn.feedforward(instance)
    out_int = decide(out)

    if i == 60:
        halt = True

    print("out_int: ", [int(x) for x in out_int])
    print("target:: ", [int(x) for x in targets[i].tolist()])

    if (out_int == targets[i]).all():
        correct += 1

print('accuracy: %s'%str(correct/len(X)))

#for node in vis.model.layers[1]:
    #if node.bias:
        #continue
    #vis.compute_DB_points(node)

#vis.draw_DB_points(vis.model.layers[1][0])

halt = True

#for x in np.arange(-2,7,0.1):
    #for y in np.arange(-2.5, 2, 0.1):
        #out = nn.feedforward([x, y])
        ##out_int = [0, 0, 0]
        ##out_int[out.index(max(out))] = 1
        #out_int = decide(out)

        #nn.vis.data_ax.scatter(x, y, color= out_int)
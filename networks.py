import numpy as np
import numpy.matlib as ml
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

import csv
import time

from scipy.special import expit

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


class NeuralMat(object):
    """
    Matrix implementation of a feedforward neural network.
    """
    def __init__(self, layer_specs, X, bias_on= False):
        self.Ws = [] #All weight matrices
        self.layer_specs = layer_specs
        self.bias_on = bias_on
        self.X = X

        for ix, layer in enumerate(layer_specs[0:-1]):
            d = layer + bias_on
            self.Ws.append(np.random.uniform(-1.0/np.sqrt(d), 1.0/np.sqrt(d),
                                             [d, layer_specs[ix + 1]]))

    def classify(self, X):
        """
        Given X an n x m matrix of n examples with m features, produces a list 'As' of activation matrices
        at each layer by multiplying X with weight matrices Ws, and passing outputs through activation function.
        Should be used to get activations after network has been trained.

        """
        #assert X.shape[1] == self.layer_specs[0], ("Mismatch in dimensionality of data and input layer.")
        A = X
        As = [A]
        for W in self.Ws:
            if self.bias_on:
                A = np.column_stack((A, np.ones(len(A))))

            O = np.dot(A, W)
            A = expit(O)

            As.append(A)

        self.As = As
        return As

    def forwardprop(self, instance):
        """
        Passes a single instance (row of X) through the weight matrices, producing activation vectors at each layer.
        Can be used to classify a single instance, or during the forward-propagation step of training.
        """
        x = instance.copy()
        if len(x.shape) == 1:
            x = x[np.newaxis]

        z = x.copy()
        zs = [z]
        for W in self.Ws:
            if self.bias_on:
                z = np.column_stack((z, [1.0]))

            z = sigmoid(np.dot(z, W))
            zs.append(z)

        self.zs = zs
        return z


    def backprop(self, t):
        """
        note: assumes sigmoid activation function.
        """
        lr = 1
        Es = [[] for i in range(len(self.zs))]

        for l in range(len(self.zs)-1, 0, -1):

            if l == len(self.zs) - 1:
                #Calculate output neuron errors:
                Es[l] = self.zs[l] * (np.ones(self.zs[l].shape) - self.zs[l]) * (t - self.zs[l])
            else:
                #Calculate hidden neuron errors:
                z = self.zs[l]
                E = Es[l + 1]
                if self.bias_on:
                    z = np.column_stack((z, np.ones(len(z))))
                    if l + 1 < len(self.zs) - 1:
                        E = E[:,0:-1]

                #Es[l] = z * (np.ones(z.shape) - z) * np.dot(Es[l + 1], self.Ws[l].T)
                Es[l] = z * (np.ones(z.shape) - z) * np.dot(E, self.Ws[l].T)


            z = self.zs[l - 1]
            E = Es[l]
            if self.bias_on:
                z = np.column_stack((z, np.ones(len(z))))
                if l < len(self.zs) - 1:
                    E = E[:,0:-1]

            #Update weights
            self.Ws[l - 1] = self.Ws[l - 1] + lr * np.dot(z.T, E)



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
                neuron = Neuron(expit, 0.6)
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
                bias = Neuron(expit, 0.6, bias= True)
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

        rand_button = mpl.widgets.Button(plt.axes([0.5, 0.02, 0.12, 0.03]), 'Randomize')
        rand_button.on_clicked(self.random_proj)

        if self.X is not None:
            self.currDim = self.X.shape[1]
            self.setup_DataDisplay()
            self.draw_data()

    def random_proj(self, ev):
        ##ISSUE: Should have some way to view projection onto the standard basis vectors.
        if self.currDim > 2:
            #Z = np.random.rand(len(X[0,:]), 2)
            Z = np.random.rand(self.currDim, 2)
            Q, R = np.linalg.qr(Z)
            #D = np.dot(X, Q)
            self.P = Q

        self.draw_data()

    def rotate_proj(self, ev):
        x_axes = [button.ax for button in self.rot_buttons['x']]
        y_axes = [button.ax for button in self.rot_buttons['y']]

        self.ax.set_autoscale_on(False)

        if ev.inaxes in x_axes:
            x_or_y = 0
            ang_num = x_axes.index(ev.inaxes)

        if ev.inaxes in y_axes:
            x_or_y = 1
            ang_num = y_axes.index(ev.inaxes)

        theta = 0.2

        R = np.eye(self.currDim - 1, self.currDim - 1)

        R[ang_num, ang_num] = np.cos(theta)
        R[ang_num, ang_num + 1] = -np.sin(theta)
        R[ang_num+1, ang_num] = np.sin(theta)
        R[ang_num+1, ang_num+1] = np.cos(theta)

        self.P[:, x_or_y] = np.dot(np.dot(np.dot(self.Q[x_or_y], R), self.Q[x_or_y].T), self.P[:, x_or_y].T).T

        self.draw_data()


    def add_data(self, X= None, targets= None):
        self.X = X
        self.targets = targets
        self.currDim = X.shape[1]

        self.setup_DataDisplay()
        self.update_dimension(self.currDim)

        #string = ("Current Dimensionality: %s" %self.currDim)
        #self.ax.set_title(string)

    def update_dimension(self, newDim):
        self.currDim = newDim

        self.create_P()

        #Prepare Qs
        self.Q = [[], []]

        Q = np.random.randn(self.currDim, self.currDim - 2) #ISSUE -1 or -2?
        Q1, r = np.linalg.qr(np.column_stack((self.P[:,0], Q)))
        self.Q[0] = Q1

        Q = np.random.randn(self.currDim, self.currDim - 2) #ISSUE -1 or -2?
        Q2, r = np.linalg.qr(np.column_stack((self.P[:,1], Q)))
        self.Q[1] = Q2

        if hasattr(self, 'rot_buttons'):
            [self.fig.delaxes(button.ax) for button in self.rot_buttons['x']]
            [self.fig.delaxes(button.ax) for button in self.rot_buttons['y']]

        self.rot_buttons = {'x':[], 'y':[]}
        plt.figure(self.fig.number) #Set current figure.
        for i in range(self.currDim - 2):
            rotx_axes = plt.axes([0.01, 0.02 + 0.1*i, 0.05, 0.05])
            roty_axes = plt.axes([0.95, 0.02 + 0.1*i, 0.05, 0.05])

            rotx_button = mpl.widgets.Button(rotx_axes, 'R')
            roty_button = mpl.widgets.Button(roty_axes, 'R')

            rotx_button.on_clicked(self.rotate_proj)
            roty_button.on_clicked(self.rotate_proj)

            self.rot_buttons['x'].append(rotx_button)
            self.rot_buttons['y'].append(roty_button)

        self.draw_data()
        self.ax.autoscale()


    def draw_data(self):
        self.ax.cla()
        string = ("Current Dimensionality: %s" % self.currDim)
        self.ax.set_title(string)

        for d in self.D.values():
            X_lo = np.dot(d['data'], self.P)
            self.ax.set_autoscale_on(False)
            self.ax.scatter(X_lo[:, 0], X_lo[:, 1], color= d['color'])

        #if self.vis.selected_node is not None:
            #self.vis.draw_DB_points(self.vis.selected_node.node)

        self.fig.canvas.draw()

    def create_P(self):
        ##Prepare P
        if self.currDim > 2:
            self.P = np.eye(self.currDim)
        else:
            self.P = np.eye(2)

    def setup_DataDisplay(self):
        """
        Creates a dictionary D that partitions the raw data X into separate sections based on their class labels,
        and assigns labels to those classes. Also creates orthonormal projection matrix P if data are hi-Dimensional.
        """
        self.create_P()

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
        self.ax.autoscale()


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
        #self.dDisplay.setup_DataDisplay()

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
        #for key, d in self.dDisplay.D.items():
            #data = np.column_stack((d['data'], np.ones((len(d['data']), 1))))
            #self.dDisplay.D[key]['data'] = np.dot(data, W)

        self.dDisplay.P = []
        W = []

        for node in layer:
            row = []
            if node.bias:
                continue

            for syn in node.in_synapses:
                row.append(syn.weight)

            W.append(row)

        W = np.array(W).T

        #ISSUE: Should I project onto W, or an orthonormalized W?
        #Q, R = np.linalg.qr(W.transpose())

        #ISSUE: Do I really want to use the weight matrix as the projection matrix? Or do I multiply, then show?
        #self.dDisplay.P = Q

        for key, d in self.dDisplay.D.items():
            data = np.column_stack((d['data'], np.ones((len(d['data']), 1))))
            self.dDisplay.D[key]['data'] = np.dot(data, W)

        self.dDisplay.update_dimension(W.shape[1])


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


def sigmoid(X):
    #Y = 1.0/(1 + math.e**(-X))
    Y = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i, j] = 1.0/(1.0 + math.e**(-1*X[i, j]))
    return Y


def logit(x):
    return np.log(x/(1 - x))

def standardize(X):
    Y = X.copy() - np.mean(X)
    Y = Y/np.std(Y)

    return Y

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
    #out= (vec > 0.5).astype(int)
    out = [0]*len(vec)
    out[list(vec).index(max(vec))] = 1

    return out

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

[X, targets] = load_data('concentric_rings.txt', labeled= True)
#[X, targets] = load_data('iris_data.txt', labeled= True)

def test_NeuralNet(X, train = False):
    nn = NeuralNet([2, 3, 2], bias_on= True)

    if train:
        for it in range(1, 100):
            shuff_X, shuff_tars = shuffle_in_unison_inplace(X, targets)

            for i, instance in enumerate(shuff_X):
                nn.feedforward(instance)
                nn.backpropagate(shuff_tars[i], show= False)


    for x in X:
        out = nn.feedforward(x)
        out_int = decide(out)
        print(out_int)

    halt = True

#X = np.array(X)[:, 2:4]
X = np.array(X)
X = standardize(X)

#test_NeuralNet(X, train= True)
nm = NeuralMat([2, 30, 2], X, bias_on= True)

for j in range(15):
    shuff_X, shuff_targets = shuffle_in_unison_inplace(X, targets)
    for i, x in enumerate(shuff_X):
        nm.forwardprop(x)
        nm.backprop(shuff_targets[i])

Y = nm.classify(X)[-1]
t = 0
c = 0
for i, y in enumerate(Y):
    print(decide(y))
    if (decide(y) == targets[i]).all():
        c += 1
    t += 1

print("Accuracy: ", c/t)

halt = True
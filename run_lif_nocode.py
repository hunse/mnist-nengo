import os
import gzip
import cPickle as pickle
import urllib

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import nengo

# --- parameters
presentation_time = 0.1
neurons_per_class = 10  # neurons per class in classifier
# pstc = 0.006
pstc = 0.005

# --- functions
def forward(x, weights, biases):
    lif = nengo.LIF()
    layers = []
    for w, b in zip(weights, biases):
        x = np.dot(x, w) + b
        x = lif.rates(x, 1, 1) / 63.04
        layers.append(x)
    return x, layers

def get_index(t):
    return min(int(t / presentation_time), n_test - 1)

def get_image(t):
    return test_images[get_index(t)]

def test_classifier(t, dots):
    return test_labels[get_index(t)] == labels[np.argmax(dots)]

# --- load the RBM data
# data = np.load('lif-126-error.npz')
data = np.load('results-spaun/params.npz')
weights = data['weights']
biases = data['biases']
Wc = data['Wc']
bc = data['bc']
n_classifier = bc.size

print [len(b) for b in biases] + [len(bc)]

# --- load the testing data
import mnist
# _, _, [test_images, test_labels] = mnist.load()
_, _, [test_images, test_labels] = mnist.augment()

for images in [test_images]:
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

# shuffle
rng = np.random.RandomState(92)
inds = rng.permutation(len(test_images))
test_images = test_images[inds]
test_labels = test_labels[inds]
n_test = len(test_images)

labels = np.unique(test_labels)
n_labels = labels.size
assert n_labels == n_classifier

# --- test as ANN
codes, layers = forward(test_images, weights, biases)
classes = np.dot(codes, Wc) + bc
inds = np.argmax(classes, axis=1)
errors = (test_labels != labels[inds])
print "ANN error:", errors.mean()
print "ANN classes: mean=%0.3f, std=%0.3f, min=%0.3f, max=%0.3f" % (
    classes.mean(), classes.std(0).mean(), classes.min(), classes.max())
for i, layer in enumerate(layers):
    print "Layer %d: sparsity=%0.3f, %0.3f" % (i, (layer > 0).mean(), (layer > 1).mean())

# try centroid classifier
code_means = np.array([codes[test_labels == label].mean(0) for label in labels])
dists = ((codes[:, :, None] - code_means.T)**2).sum(axis=1)
errors = (test_labels != labels[np.argmin(dists, axis=1)])
print "ANN centroid error:", errors.mean()

# try dot product classifier
code_mean = codes.mean(0)
dists = np.dot(codes - code_mean, (code_means - code_mean).T)
errors = (test_labels != labels[np.argmax(dists, axis=1)])
print "ANN dot error:", errors.mean()

if 1:
    plt.figure(101)
    plt.clf()
    r = len(layers)
    for i, layer in enumerate(layers):
        plt.subplot(r, 1, i+1)
        plt.hist(layer.flatten(), bins=15)

# --- create the model
neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.002)
max_rate = 63.04
intercept = 0
amp = 1. / max_rate
assert np.allclose(neuron_type.gain_bias([max_rate], [intercept]), (1, 1), atol=1e-2)

model = nengo.Network(seed=97)
with model:
    input_images = nengo.Node(output=get_image, label='images')

    # --- make nonlinear layers
    layers = []
    for i, [W, b] in enumerate(zip(weights, biases)):
        n = b.size
        layer = nengo.Ensemble(n, 1, label='layer %d' % i, neuron_type=neuron_type,
                               max_rates=max_rate*np.ones(n),
                               intercepts=intercept*np.ones(n))
        bias = nengo.Node(output=b)
        nengo.Connection(bias, layer.neurons, transform=np.eye(n), synapse=0)

        if i == 0:
            nengo.Connection(input_images, layer.neurons,
                             transform=W.T, synapse=pstc)
        else:
            nengo.Connection(layers[-1].neurons, layer.neurons,
                             transform=W.T * amp, synapse=pstc)

        layers.append(layer)

    # --- make cleanup
    class_layer = nengo.networks.EnsembleArray(
        neurons_per_class, n_classifier, label='class', radius=5)
    class_bias = nengo.Node(output=bc)
    nengo.Connection(class_bias, class_layer.input, synapse=0)
    nengo.Connection(layers[-1].neurons, class_layer.input,
                     transform=Wc.T * amp, synapse=pstc)

    test = nengo.Node(output=test_classifier, size_in=n_classifier)
    nengo.Connection(class_layer.output, test)

    # --- make centroid classifier node
    def centroid_test_fn(t, x):
        d = ((x - code_means)**2).sum(1)
        return test_labels[get_index(t)] == labels[np.argmin(d)]

    centroid_test = nengo.Node(centroid_test_fn, size_in=layers[-1].n_neurons)
    nengo.Connection(layers[-1].neurons, centroid_test,
                     transform=amp, synapse=pstc)

    # --- make dot classifier node
    def dot_test_fn(t, x):
        # d = np.dot(code_means, x)
        d = np.dot(code_means - code_mean, x - code_mean)
        return test_labels[get_index(t)] == labels[np.argmax(d)]

    dot_test = nengo.Node(dot_test_fn, size_in=layers[-1].n_neurons)
    nengo.Connection(layers[-1].neurons, dot_test,
                     transform=amp, synapse=pstc)

    # --- make probes
    probe_layers = [nengo.Probe(layer.neurons) for layer in layers]
    probe_class = nengo.Probe(class_layer.output, synapse=0.03)
    probe_test = nengo.Probe(test, synapse=0.01)
    probe_centroid = nengo.Probe(centroid_test, synapse=0.01)
    probe_dot = nengo.Probe(dot_test, synapse=0.01)


# --- simulation
sim = nengo.Simulator(model)
# sim.run(1000., progress_bar=None)
# sim.run(100.)
# sim.run(10.)
sim.run(5.)
# sim.run(1.)

t = sim.trange()

# --- compute error rate
zblocks = sim.data[probe_test].reshape(-1, 100)[:, 50:]  # 50 ms blocks at end of each 100
errors = np.mean(zblocks, axis=1) < 0.5
print "Neuron error:", errors.mean()

zblocks = sim.data[probe_centroid].reshape(-1, 100)[:, 50:]
errors = np.mean(zblocks, axis=1) < 0.5
print "Neuron centroid error:", errors.mean()

zblocks = sim.data[probe_dot].reshape(-1, 100)[:, 50:]
errors = np.mean(zblocks, axis=1) < 0.5
print "Neuron dot error:", errors.mean()

# z2 = np.argmax(y, axis=1) == labels.repeat(100)
# zblocks = z2.reshape(-1, 100)[:, 80:]  # 20 ms blocks at end of each 100
# errors = np.mean(zblocks, axis=1) < 0.5
# print errors.mean()

# --- plots
if sim.time <= 5:
    from nengo.utils.matplotlib import rasterplot

    def plot_bars():
        ylim = plt.ylim()
        for x in np.arange(0, t[-1], presentation_time):
            plt.plot([x, x], ylim, 'k--')

    inds = slice(0, int(t[-1]/presentation_time) + 1)
    images = test_images[inds]
    labels = test_labels[inds]
    allimage = np.zeros((28, 28 * len(images)), dtype=images.dtype)
    for i, image in enumerate(images):
        allimage[:, i * 28:(i + 1) * 28] = image.reshape(28, 28)

    plt.figure(1)
    plt.clf()
    r, c = 6, 1

    plt.subplot(r, c, 1)
    plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(r, c, 2)
    rasterplot(t, sim.data[probe_layers[0]][:,:200])
    plot_bars()
    plt.xticks([])
    plt.ylabel('layer 1 (500)')

    plt.subplot(r, c, 3)
    rasterplot(t, sim.data[probe_layers[1]])
    plt.xticks([])
    plt.yticks(np.linspace(0, 200, 5))
    plot_bars()
    plt.ylabel('layer 2 (200)')

    plt.subplot(r, c, 4)
    plt.plot(t, sim.data[probe_class])
    plot_bars()
    plt.ylabel('class')

    plt.subplot(r, c, 5)
    plt.plot(t, sim.data[probe_test])
    plt.ylim([-0.1, 1.1])
    plot_bars()
    plt.ylabel('correct')

    plt.subplot(r, c, 6)
    plt.plot(t, sim.data[probe_dot])
    plt.ylim([-0.1, 1.1])
    plot_bars()
    plt.xlabel('time [s]')
    plt.ylabel('correct')

    plt.tight_layout()

    plt.savefig('run_lif_nocode.png')

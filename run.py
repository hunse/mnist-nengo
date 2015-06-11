from __future__ import print_function

import argparse
import os
import sys

import nengo
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import mnist

urls = {
    'lif-111-error.npz': 'http://files.figshare.com/2106879/lif_111_error.npz',
    'lif-126-error.npz': 'http://files.figshare.com/2106875/lif_126_error.npz',
}

# --- arguments
parser = argparse.ArgumentParser(description="Run network in spiking neurons")
parser.add_argument('--gui', action='store_true', help="Run in the GUI")
parser.add_argument('--augment', action='store_true')
parser.add_argument('--presentations', type=float, default=20)
parser.add_argument('loadfile', help="Parameter file to load")
parser.add_argument('savefile', nargs='?', default=None, help="Where to save output")
args = parser.parse_args()

# --- parameters
n_pres = args.presentations if not args.gui else 10000
pres_time = 0.1
neurons_per_class = 10  # neurons per class in classifier
synapse = 0.005
# synapse = nengo.synapses.Alpha(0.005)

# --- functions
def get_index(t):
    return int(t / pres_time) % len(images)

def get_image(t):
    return images[get_index(t)]

def test_classifier(t, dots):
    return labels[get_index(t)] == classes[np.argmax(dots)]

# --- load the RBM data
if not os.path.exists(args.loadfile) and args.loadfile in urls:
    urllib.urlretrieve(urls[args.loadfile], args.loadfile)
if os.path.exists(args.loadfile):
    data = np.load(args.loadfile)
    weights = data['weights']
    biases = data['biases']
    Wc = data['Wc']
    bc = data['bc']
    n_classifier = bc.size
    print("Loaded %s classifier" % str([len(b) for b in biases] + [len(bc)]))
else:
    raise ValueError("Cannot find or download '%s'" % args.loadfile)

# --- load the testing data
_, _, [images, labels] = (
    mnist.augment() if args.augment else mnist.load())

images -= images.mean(axis=0, keepdims=True)
images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

# shuffle
rng = np.random.RandomState(92)
inds = rng.permutation(len(images))[:n_pres]
images = images[inds]
labels = labels[inds]
n_test = len(images)

classes = np.unique(labels)
n_classes = classes.size
assert n_classes == n_classifier

# --- create the model
neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.002)
max_rate = np.array(63.04)
intercept = np.array(0)
amp = 1. / max_rate
assert np.allclose(neuron_type.gain_bias(max_rate, intercept), (1, 1), atol=1e-2)

model = nengo.Network(seed=97)
with model:
    input_images = nengo.Node(output=get_image, label='images')

    # --- make nonlinear layers
    layers = []
    for i, [W, b] in enumerate(zip(weights, biases)):
        n = b.size
        layer = nengo.Ensemble(
            n, 1, label='layer %d' % i, neuron_type=neuron_type,
            max_rates=max_rate*np.ones(n), intercepts=intercept*np.ones(n))
        bias = nengo.Node(output=b, label='bias %d' % i)
        nengo.Connection(bias, layer.neurons, synapse=None)

        if i == 0:
            nengo.Connection(input_images, layer.neurons,
                             transform=W.T, synapse=synapse)
        else:
            nengo.Connection(layers[-1].neurons, layer.neurons,
                             transform=W.T * amp, synapse=synapse)

        layers.append(layer)

    # --- make classifier
    class_layer = nengo.networks.EnsembleArray(
        neurons_per_class, n_classifier, label='class', radius=5)
    class_bias = nengo.Node(output=bc)
    nengo.Connection(class_bias, class_layer.input, synapse=None)
    nengo.Connection(layers[-1].neurons, class_layer.input,
                     transform=Wc.T * amp, synapse=synapse)

    test = nengo.Node(output=test_classifier, size_in=n_classifier)
    nengo.Connection(class_layer.output, test)

# --- simulation
if args.gui:
    import nengo_gui
    nengo_gui.Viz(__file__).start()
    sys.exit(0)

with model:
    # --- make probes
    if n_pres <= 100:
        probe_layers = [nengo.Probe(layer.neurons) for layer in layers]
    else:
        probe_layers = []
    probe_class = nengo.Probe(class_layer.output, synapse=0.03)
    probe_test = nengo.Probe(test, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(pres_time * n_pres)

t = sim.trange()

layers = tuple(sim.data[p] for p in probe_layers)
classifier = sim.data[probe_class]
test = sim.data[probe_test]

if args.savefile is not None:
    np.savez(args.savefile,
             t=t, classes=classes, images=images, labels=labels,
             classifier=classifier, test=test, pres_time=pres_time)
    print("Saved data at '%s'" % args.savefile)

# --- view results (see also view.py)
from view import compute_spiking_error, view_spiking

errors = compute_spiking_error(t, test, pres_time)
print("Spiking network error: %0.2f%%" % (100 * errors.mean()))

imgfile = (os.path.splitext(args.savefile)[0] + '.png'
           if args.savefile is not None else None)
view_spiking(t, images, labels, classifier, test, pres_time,
             layers=layers, savefile=imgfile)

"""
Training an autoencoder with LIF-likes
"""
import os

import numpy as np
import matplotlib.pyplot as plt

os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
import theano
import theano.tensor as tt

import mnist
import neurons
import plotting
from autoencoder import (
    rms, show_recons, FileObject, Autoencoder, DeepAutoencoder)

SPAUN = False
results_dir = 'results-spaun' if SPAUN else 'results-lif'

plt.ion()

# --- define our rate neuron model
neuron = ('softlif', dict(
    sigma=0.01, tau_rc=0.02, tau_ref=0.002, gain=1, bias=1, amp=1. / 63.04))
neuron_fn = neurons.get_theano_fn(*neuron)

# --- load the data
train, valid, test = mnist.augment() if SPAUN else mnist.load()
train_images, _ = train
valid_images, _ = valid
test_images, _ = test

for images in [train_images, valid_images, test_images]:
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

# --- pretrain with SGD backprop
shapes = [(28, 28), 500, 200]
funcs = [None, neuron_fn, neuron_fn]
rf_shapes = [(9, 9), None]
rates = [1., 1.]
# rates = [0.05, 0.05]

n_layers = len(shapes) - 1
assert len(funcs) == len(shapes)
assert len(rf_shapes) == n_layers
assert len(rates) == n_layers

n_epochs = 15
batch_size = 100

deep = DeepAutoencoder()
data = train_images
for i in range(n_layers):
    savename = results_dir + "/lif-auto-%d.npz" % i
    if not os.path.exists(savename):
        auto = Autoencoder(
            shapes[i], shapes[i+1], rf_shape=rf_shapes[i],
            vis_func=funcs[i], hid_func=funcs[i+1])
        deep.autos.append(auto)
        auto.auto_sgd(data, deep, test_images,
                      n_epochs=n_epochs, rate=rates[i])
        auto.to_file(savename)
    else:
        auto = FileObject.from_file(savename)
        assert type(auto) is Autoencoder
        deep.autos.append(auto)

    data = auto.encode(data)

plt.figure(99)
plt.clf()
recons = deep.reconstruct(test_images)
show_recons(test_images, recons)
print "recons error", rms(test_images - recons, axis=1).mean()

deep.auto_sgd(train_images, test_images, rate=0.3, n_epochs=30)
print "recons error", rms(test_images - recons, axis=1).mean()

# --- train classifier with backprop
savename = results_dir + "/classifier-hinge.npz"
if not os.path.exists(savename):
    deep.train_classifier(train, test)
    np.savez(savename, W=deep.W, b=deep.b)
else:
    savedata = np.load(savename)
    deep.W, deep.b = savedata['W'], savedata['b']

print "mean error", deep.test(test).mean()

# --- train with backprop
deep.sgd(train, test, n_epochs=150, tradeoff=1, noise=0.3, shift=True)
print "mean error", deep.test(test).mean()

# --- save parameters
d = {}
d['weights'] = [auto.W.get_value() for auto in deep.autos]
d['biases'] = [auto.c.get_value() for auto in deep.autos]
if all(hasattr(auto, 'V') for auto in deep.autos):
    d['rec_weights'] = [auto.V.get_value() for auto in deep.autos]
    d['rec_biases'] = [auto.b.get_value() for auto in deep.autos]
d['Wc'] = deep.W
d['bc'] = deep.b
np.savez(results_dir + '/params.npz', **d)

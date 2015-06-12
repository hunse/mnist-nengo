"""
Training an autoencoder with LIF-likes
"""
import argparse
import datetime
import os

parser = argparse.ArgumentParser(description="Train a deep network on MNIST")
parser.add_argument('--gpu', action='store_true', help="Train on the GPU")
parser.add_argument('--spaun', action='store_true',
                    help="Train with augmented dataset for Spaun")
parser.add_argument('savefile', nargs='?', default=None, help="Where to save output")
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt

if args.gpu:
    os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
import theano
import theano.tensor as tt

import mnist
import neurons
import plotting
from autoencoder import (
    rms, show_recons, FileObject, Autoencoder, DeepAutoencoder)

plt.ion()

# --- define the network architecture
if 1:
    # architecture one
    shapes = [(28, 28), 500, 200]
    rf_shapes = [(9, 9), None]
    rates = [1., 1.]
else:
    # architecture two
    shapes = [(28, 28), 1000, 500, 200]
    rf_shapes = [(9, 9), None, None]
    rates = [1., 1., 1.]

n_layers = len(shapes) - 1
assert len(rf_shapes) == n_layers
assert len(rates) == n_layers

# --- define our rate neuron model
neuron = ('softlif', dict(
    sigma=0.01, tau_rc=0.02, tau_ref=0.002, gain=1, bias=1, amp=1. / 63.04))
neuron_fn = neurons.get_theano_fn(*neuron)

# --- load the data
train, valid, test = mnist.load(
    normalize=True, shuffle=True, spaun=args.spaun)
train_images, test_images = train[0], test[0]

# --- pretrain with SGD backprop
n_epochs = 15
batch_size = 100

deep = DeepAutoencoder()
data = train_images
for i in range(n_layers):
    vis_func = None if i == 0 else neuron_fn

    # create autoencoder for the next layer
    auto = Autoencoder(
        shapes[i], shapes[i+1], rf_shape=rf_shapes[i],
        vis_func=vis_func, hid_func=neuron_fn)
    deep.autos.append(auto)

    # train the autoencoder using SGD
    auto.auto_sgd(data, deep, test_images, n_epochs=n_epochs, rate=rates[i])

    # hidden layer activations become training data for next layer
    data = auto.encode(data)

plt.figure(99)
plt.clf()
recons = deep.reconstruct(test_images)
show_recons(test_images, recons)
print "recons error", rms(test_images - recons, axis=1).mean()

deep.auto_sgd(train_images, test_images, rate=0.3, n_epochs=30)
print "recons error", rms(test_images - recons, axis=1).mean()

# --- train classifier with backprop
deep.train_classifier(train, test)
print "mean error", deep.test(test).mean()

# --- train with backprop
deep.sgd(train, test, n_epochs=150, tradeoff=1, noise=0.3, shift=True, rate=0.1)
# deep.sgd(train, test, n_epochs=150, tradeoff=1, noise=0.3, shift=True, rate=0.05)
# deep.sgd(train, test, n_epochs=150, tradeoff=1, noise=0.3, shift=True, rate=0.01)
print "mean error", deep.test(test).mean()

# --- save parameters
savefile = args.savefile
if savefile is None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    savefile = 'params_%s_%s.npz' % (neuron[0], timestamp)

d = {}
d['weights'] = [auto.W.get_value() for auto in deep.autos]
d['biases'] = [auto.c.get_value() for auto in deep.autos]
if all(hasattr(auto, 'V') for auto in deep.autos):
    d['rec_weights'] = [auto.V.get_value() for auto in deep.autos]
    d['rec_biases'] = [auto.b.get_value() for auto in deep.autos]
d['Wc'] = deep.W
d['bc'] = deep.b
d['neuron'] = neuron

np.savez(savefile, **d)

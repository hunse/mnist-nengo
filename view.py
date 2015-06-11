from __future__ import print_function

import argparse
import os

import nengo
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import mnist
import neurons


def _propup_static(params, images, kind='lif'):
    weights = params['weights']
    biases = params['biases']
    Wc = params['Wc']
    bc = params['bc']
    n_classifier = bc.size

    neuron_params = dict(
        sigma=0.01, tau_rc=0.02, tau_ref=0.002, gain=1, bias=1, amp=1. / 63.04)
    if kind != 'softlif':
        neuron_params.pop('sigma')
    neuron_fn = neurons.get_numpy_fn(kind, neuron_params)

    def forward(x, weights, biases):
        layers = []
        for w, b in zip(weights, biases):
            x = neuron_fn(np.dot(x, w) + b)
            layers.append(x)
        return x, layers

    codes, layers = forward(images, weights, biases)
    yc = np.dot(codes, Wc) + bc
    return layers, codes, yc


def compute_static_error(params, images, labels, kind='lif'):
    layers, codes, yc = _propup_static(params, images, kind=kind)
    inds = np.argmax(yc, axis=1)
    classes = np.unique(labels)
    errors = (labels != classes[inds])
    return errors


def view_static(params, images, labels, kind='lif'):
    layers, codes, yc = _propup_static(params, images, kind=kind)

    for i, layer in enumerate(layers):
        print("Layer %d: mean=%0.3f; sparsity=%0.3f (>0), %0.3f (>1)" % (
            i, layer.mean(), (layer > 0).mean(), (layer > 1).mean()))

    print("Classifier: mean=%0.3f, std=%0.3f, min=%0.3f, max=%0.3f" % (
        yc.mean(), yc.std(0).mean(), yc.min(), yc.max()))

    plt.figure()
    # plt.clf()
    r = len(layers)
    for i, layer in enumerate(layers):
        plt.subplot(r, 1, i+1)
        plt.hist(layer.flatten(), bins=15)

    plt.show()


def compute_spiking_error(t, test, pres_time, check_time=0.05, cutoff=0.5):
    assert test.ndim == 1 or test.ndim == 2 and test.shape[1] == 1
    dt = float(t[1] - t[0])
    pres_len = int(round(pres_time / dt))
    check_len = int(round(check_time / dt))

    assert test.size % pres_len == 0
    if test.size % pres_len != 0:
        test_pad = np.zeros(test.size / pres_len + 1, dtype=test.dtype)
        test_pad[:test.size] = test
        test = test_pad

    # take blocks at the end of each presentation
    blocks = test.reshape(-1, pres_len)[:, -check_time:]
    errors = np.mean(blocks, axis=1) < cutoff
    return errors


def view_spiking(t, images, labels, classifier, test, pres_time, max_pres=20,
                 layers=[], savefile=None):
    from nengo.utils.matplotlib import rasterplot

    def plot_bars():
        ylim = plt.ylim()
        for x in np.arange(0, t[-1], pres_time):
            plt.plot([x, x], ylim, 'k--')

    n_pres = min(int(t[-1] / pres_time), max_pres)
    images = images[:n_pres]
    labels = labels[:n_pres]

    max_t = n_pres * pres_time
    tmask = t <= max_t
    t = t[tmask]
    classifier = classifier[tmask]
    test = test[tmask]

    allimage = np.zeros((28, 28 * len(images)), dtype=images.dtype)
    for i, image in enumerate(images):
        allimage[:, i * 28:(i + 1) * 28] = image.reshape(28, 28)

    plt.figure()
    # plt.clf()
    r, c = 3 + len(layers), 1
    i = np.array([0])
    def next_subplot():
        i[:] += 1
        return plt.subplot(r, c, i)

    next_subplot()
    plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')
    plt.xticks([])
    plt.yticks([])

    max_neurons = 200
    for layer in layers:
        next_subplot()
        if layer.shape[1] > max_neurons:
            layer = layer[:, :max_neurons]
        rasterplot(t, layer)
        plot_bars()
        plt.xticks([])
        plt.ylabel('layer 1 (500)')

    next_subplot()
    plt.plot(t, classifier)
    plot_bars()
    plt.ylabel('class')

    next_subplot()
    plt.plot(t, test)
    plt.ylim([-0.1, 1.1])
    plot_bars()
    plt.ylabel('correct')

    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)
        print("Saved image at '%s'" % savefile)

    plt.show()


if __name__ == '__main__':
    # --- arguments
    parser = argparse.ArgumentParser(
        description="View network or spiking network results")
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('loadfile', help="Parameter file to load")
    args = parser.parse_args()

    if not os.path.exists(args.loadfile):
        raise IOError("Cannot find '%s'" % args.loadfile)

    data = np.load(args.loadfile)
    if all(a in data for a in ['weights', 'biases', 'Wc', 'bc']):
        # Static network params file

        # --- load the testing data
        _, _, [images, labels] = (
            mnist.augment() if args.augment else mnist.load())

        images -= images.mean(axis=0, keepdims=True)
        images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)
        assert np.unique(labels).size == data['bc'].size

        # --- compute the error
        errors = compute_static_error(data, images, labels, kind='softlif')
        print("----- Static network with softlif -----")
        print("Static error: %0.2f%%" % (100 * errors.mean()))

        errors = compute_static_error(data, images, labels, kind='lif')
        print("----- Static network with lif -----")
        print("Static error: %0.2f%%" % (100 * errors.mean()))
        view_static(data, images, labels, kind='lif')

    elif all(a in data for a in ['t', 'classifier', 'test']):
        # Spiking run record file

        # --- compute the error
        args = dict((a, data[a]) for a in ['t', 'test', 'pres_time'])
        errors = compute_spiking_error(**args)
        print("Spiking network error: %0.2f%%" % (100 * errors.mean()))

        args = dict((a, data[a]) for a in [
            't', 'images', 'labels', 'classifier', 'test', 'pres_time'])
        view_spiking(**args)

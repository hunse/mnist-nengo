"""
Training an autoencoder with LIF-likes
"""

import collections

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
os.environ['THEANO_FLAGS'] = 'mode=DEBUG_MODE'
import theano
import theano.tensor as tt
import theano.sandbox.rng_mrg

from hinge import multi_hinge_margin
import plotting

from hunse_tools import tic, toc

plt.ion()


def rms(x, **kwargs):
    return np.sqrt((x**2).mean(**kwargs))


def mnist(filename='mnist.pkl.gz'):
    import gzip
    import os
    import cPickle as pickle
    import urllib

    if not os.path.exists(filename):
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, filename=filename)

    with gzip.open(filename, 'rb') as f:
        train, valid, test = pickle.load(f)

    return train, valid, test


def normalize(images):
    """Normalize a set of images"""
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)


def show_recons(x, z):
    plotting.compare([x.reshape(-1, 28, 28), z.reshape(-1, 28, 28)],
                     rows=5, cols=20, vlims=(-1, 2))


def sparse_mask(vis_shape, n_hid, rf_shape, rng=np.random):
    assert isinstance(vis_shape, tuple) and len(vis_shape) == 2
    assert isinstance(rf_shape, tuple) and len(rf_shape) == 2
    M, N = vis_shape
    m, n = rf_shape
    n_vis = M * N

    # find random positions for top-left corner of each RF
    i = rng.randint(low=0, high=M-m+1, size=n_hid)
    j = rng.randint(low=0, high=N-n+1, size=n_hid)

    mask = np.zeros((M, N, n_hid), dtype='bool')
    for k in xrange(n_hid):
        mask[i[k]:i[k]+m, j[k]:j[k]+n, k] = True

    return mask.reshape(n_vis, n_hid)


def split_params(param_vect, numpy_params):
    split = []
    i = 0
    for p in numpy_params:
        split.append(param_vect[i:i + p.size].reshape(p.shape))
        i += p.size
    return split


def join_params(param_arrays):
    return np.hstack([p.flatten() for p in param_arrays])


class FileObject(object):
    """
    A object that can be saved to file
    """
    def to_file(self, file_name):
        d = {}
        d['__class__'] = self.__class__
        d['__dict__'] = self.__getstate__()
        np.savez(file_name, **d)

    @staticmethod
    def from_file(file_name):
        npzfile = np.load(file_name)
        cls = npzfile['__class__'].item()
        d = npzfile['__dict__'].item()

        self = cls.__new__(cls)
        self.__setstate__(d)
        return self


class Autoencoder(FileObject):
    """Autoencoder with tied weights"""

    def __init__(self, vis_shape, n_hid,
                 W=None, V=None, c=None, b=None, mask=None,
                 rf_shape=None, hid_func=None, vis_func=None, seed=22):
        dtype = theano.config.floatX

        self.vis_shape = vis_shape if isinstance(vis_shape, tuple) else (vis_shape,)
        self.n_vis = np.prod(vis_shape)
        self.n_hid = n_hid
        self.hid_func = hid_func
        self.vis_func = vis_func
        self.seed = seed

        rng = np.random.RandomState(seed=self.seed)
        self.theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=self.seed)

        # create initial weights and biases
        if W is None:
            Wmag = 4 * np.sqrt(6. / (self.n_vis + self.n_hid))
            W = rng.uniform(
                low=-Wmag, high=Wmag, size=(self.n_vis, self.n_hid)
            ).astype(dtype)

        if c is None:
            c = np.zeros(self.n_hid, dtype=dtype)

        if b is None:
            b = np.zeros(self.n_vis, dtype=dtype)

        # create initial sparsity mask
        self.rf_shape = rf_shape
        self.mask = mask
        if rf_shape is not None and mask is None:
            self.mask = sparse_mask(vis_shape, n_hid, rf_shape, rng=rng)

        if mask is not None:
            W = W * self.mask  # make initial W sparse
            if V is not None:
                V = V * self.mask.T

        # create states for weights and biases
        self.W = theano.shared(W.astype(dtype), name='W')
        self.c = theano.shared(c.astype(dtype), name='c')
        self.b = theano.shared(b.astype(dtype), name='b')
        if V is not None:
            self.V = theano.shared(V.astype(dtype), name='V')

    def __getstate__(self):
        d = dict(self.__dict__)
        for k, v in d.items():
            if k in ['W', 'V', 'c', 'b']:
                d[k] = v.get_value()
        return d

    def __setstate__(self, state):
        for k, v in state.items():
            if k in ['W', 'V', 'c', 'b']:
                self.__dict__[k] = theano.shared(v, name=k)
            else:
                self.__dict__[k] = v

    @property
    def filters(self):
        if self.mask is None:
            return self.W.get_value().T.reshape((self.n_hid,) + self.vis_shape)
        else:
            filters = self.W.get_value().T[self.mask.T]
            shape = (self.n_hid,) + self.rf_shape
            return filters.reshape(shape)

    def propup(self, x, noise=0):
        a = tt.dot(x, self.W) + self.c
        if noise > 0:
            a += self.theano_rng.normal(
                size=a.shape, std=noise, dtype=theano.config.floatX)
        return self.hid_func(a) if self.hid_func is not None else a

    def propdown(self, y):
        V = self.V if hasattr(self, 'V') else self.W.T
        a = tt.dot(y, V) + self.b
        return self.vis_func(a) if self.vis_func is not None else a

    @property
    def encode(self):
        data = tt.matrix('data')
        code = self.propup(data)
        return theano.function([data], code)

    @property
    def decode(self):
        code = tt.matrix('code')
        data = self.propdown(code)
        return theano.function([code], data)

    @property
    def reconstruct(self):
        data = tt.matrix('data')
        code = self.propup(data)
        recs = self.propdown(code)
        return theano.function([data], recs)

    def check_params(self):
        for param in [self.W, self.c, self.b]:
            if param is not None:
                assert np.isfinite(param.get_value()).all()

    def auto_sgd(self, images, deep=None, test_images=None,
                 batch_size=100, rate=0.1, noise=1., n_epochs=10):
        assert not hasattr(self, 'V')

        dtype = theano.config.floatX
        params = [self.W, self.c, self.b]

        # --- compute backprop function
        x = tt.matrix('images')
        # xn = x + self.theano_rng.normal(size=x.shape, std=noise, dtype=dtype)


        y = self.propup(xn)
        z = self.propdown(y)

        # compute coding error
        rmses = tt.sqrt(tt.mean((x - z)**2, axis=1))
        error = tt.mean(rmses)

        # compute gradients
        grads = tt.grad(error, params)
        updates = collections.OrderedDict()
        for param, grad in zip(params, grads):
            updates[param] = param - tt.cast(rate, dtype) * grad

        if self.mask is not None:
            updates[self.W] = updates[self.W] * self.mask

        train_dbn = theano.function([x], error, updates=updates)
        # reconstruct = deep.reconstruct if deep is not None else None
        encode = self.encode
        decode = self.decode

        # --- perform SGD
        batches = images.reshape(-1, batch_size, images.shape[1])
        assert np.isfinite(batches).all()

        for epoch in range(n_epochs):
            costs = []
            for batch in batches:
                costs.append(train_dbn(batch))
                self.check_params()

            print "Epoch %d: %0.3f (sparsity: pop: %0.3f, life: %0.3f)" % (epoch, np.mean(costs))

            if test_images is not None:
                # plot reconstructions on test set
                plt.figure(2)
                plt.clf()
                test = test_images
                codes = encode(test)
                recs = decode(codes)
                show_recons(test, recs)
                plt.draw()

                print "Test set: (error: %0.3f) (sparsity: %0.3f)" % (
                    rms(test - recs, axis=1).mean(), (codes > 0).mean())

            # plot filters
            plt.figure(3)
            plt.clf()
            plotting.filters(self.filters, rows=10, cols=20)
            plt.draw()


# --- load the data
train, valid, test = mnist()
train_images, _ = train
valid_images, _ = valid
test_images, _ = test

for images in [train_images, valid_images, test_images]:
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

# --- train autoencoder
n_hid = 500
rf_shapes = [(9, 9), None]
rates = [1., 0.3]
n_layers = len(shapes) - 1
assert len(linear) == len(shapes)
assert len(rf_shapes) == n_layers
assert len(rates) == n_layers

n_epochs = 15
rate = 1.0
batch_size = 100

auto = Autoencoder(
    (28, 28), n_hid, rf_shape=rf_shape,
    vis_func=None, hid_func=nlif)

auto.auto_sgd(data, test_images, n_epochs=n_epochs, rate=rate)


plt.figure(99)
plt.clf()
recons = auto.reconstruct(test_images)
show_recons(test_images, recons)

print "recons error", rms(test_images - recons, axis=1).mean()

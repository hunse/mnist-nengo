import numpy as np


def load(filename='mnist.pkl.gz'):
    import gzip
    import os
    import cPickle as pickle
    import urllib

    if not os.path.exists(filename):
        if filename.endswith('mnist.pkl.gz'):
            url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
            urllib.urlretrieve(url, filename=filename)
        else:
            raise NotImplementedError(
                "I do not know where to find '%s'" % filename)

    with gzip.open(filename, 'rb') as f:
        train, valid, test = pickle.load(f)

    return train, valid, test


def augment(seed=11):
    rng = np.random.RandomState(seed)

    mtrain, mvalid, mtest = load('mnist.pkl.gz')
    atrain, _, _ = load('spaun_sym.pkl.gz')  # 'valid' and 'test' == 'train'

    x, y = atrain[0][10:], atrain[1][10:]

    def aug(data, r):
        n = data[0].shape[0] / 10  # approximate examples per label
        na = n / r                 # examples per augmented category

        xx = np.vstack([data[0], np.tile(x, (na, 1))])
        yy = np.hstack([data[1], np.tile(y, na)])

        # shuffle
        i = rng.permutation(xx.shape[0])
        xx, yy = xx[i], yy[i]

        return xx, yy

    return aug(mtrain, 5), aug(mvalid, 5), aug(mtest, 5)


def normalize(images):
    """Normalize a set of images in-place"""
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)


def test_augment():
    import matplotlib.pyplot as plt

    atrain, _, _ = load('spaun_sym.pkl.gz')
    assert len(atrain[0]) == 24

    plt.figure()
    axes = [plt.subplot(4, 6, i+1) for i in range(24)]
    for i in range(24):
        plt.subplot(4, 6, i+1)
        plt.imshow(atrain[0][i].reshape(28, 28), cmap='gray', interpolation=None)
        plt.title('%s' % atrain[1][i])

    plt.show()


if __name__ == '__main__':
    test_augment()

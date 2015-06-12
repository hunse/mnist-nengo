import cPickle as pickle
import gzip
import os
import urllib

import numpy as np

urls = {
    'mnist.pkl.gz': 'http://deeplearning.net/data/mnist/mnist.pkl.gz',
    'spaun_sym.pkl.gz': 'http://files.figshare.com/2106874/spaun_sym.pkl.gz',
}


def read_file(filepath):
    if not os.path.exists(filepath):
        if filepath in urls:
            urllib.urlretrieve(urls[filepath], filename=filepath)
            print("Fetched '%s' to '%s'" % (urls[filepath], filepath))
        else:
            raise NotImplementedError(
                "I do not know where to find '%s'" % filepath)

    with gzip.open(filepath, 'rb') as f:
        train, valid, test = pickle.load(f)

    return train, valid, test


def load(normalize=False, shuffle=False, spaun=False, seed=8):
    sets = read_file('mnist.pkl.gz')

    if spaun:
        sets = _augment(*sets)

    if shuffle or augment:  # always shuffle on augment
        rng = np.random.RandomState(seed)
        sets = tuple(_shuffle(*s, rng=rng) for s in sets)

    if normalize:
        for images, labels in sets:
            _normalize(images)

    return sets


def _augment(train, valid, test, ratio=0.2):
    atrain, _, _ = read_file('spaun_sym.pkl.gz')  # 'valid' and 'test' == 'train'
    x, y = atrain[0][10:], atrain[1][10:]

    def aug(data, ratio):
        images, labels = data
        n = images.shape[0] / 10  # approximate examples per label
        na = int(n * ratio)       # examples per augmented category

        xx = np.vstack([images, np.tile(x, (na, 1))])
        yy = np.hstack([labels, np.tile(y, na)])

        return xx, yy

    return aug(train, ratio), aug(valid, ratio), aug(test, ratio)


def _shuffle(images, labels, rng=np.random):
    assert images.shape[0] == labels.shape[0]
    i = rng.permutation(images.shape[0])
    return images[i], labels[i]


def _normalize(images):
    """Normalize a set of images in-place"""
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)


def test_augment():
    import matplotlib.pyplot as plt

    # atrain, _, _ = load('spaun_sym.pkl.gz')
    # assert len(atrain[0]) == 24

    atrain, _, _ = load(augment=True)

    plt.figure()
    axes = [plt.subplot(4, 6, i+1) for i in range(24)]
    for i in range(24):
        plt.subplot(4, 6, i+1)
        plt.imshow(atrain[0][i].reshape(28, 28), cmap='gray', interpolation=None)
        plt.title('%s' % atrain[1][i])

    plt.show()


if __name__ == '__main__':
    test_augment()

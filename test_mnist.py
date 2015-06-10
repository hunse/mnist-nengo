import numpy as np
import matplotlib.pyplot as plt

import mnist


def test_augment():
    train, valid, test = mnist.augment()

    plt.figure()

    r, c = 5, 10

    for i in range(r * c):
        plt.subplot(r, c, i + 1)
        plt.imshow(train[0][i].reshape(28, 28), cmap='gray')

    plt.show()


if __name__ == '__main__':
    test_augment()

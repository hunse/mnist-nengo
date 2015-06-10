import numpy as np
import nengo

data = np.load('lif-126-error.npz')
weights = data['weights']
biases = data['biases']
Wc = data['Wc']
bc = data['bc']


def classify(x):
    lif = nengo.LIF()
    layers = []
    for w, b in zip(weights, biases):
        x = np.dot(x, w) + b
        x = lif.rates(x, 1, 1) / 63.04
        layers.append(x)

    x = np.dot(x, Wc) + bc
    layers.append(x)

    y = np.argmax(x, axis=1)
    return y, layers


import mnist
atrain, avalid, atest = mnist.load('spaun_sym.pkl.gz')

y, _ = classify(atrain[0])
print y
print atrain[1]

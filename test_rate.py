import nengo

with nengo.Network() as model:
    a = nengo.Ensemble(100, 1, neuron_type=nengo.LIFRate())
    ap = nengo.Probe(a)

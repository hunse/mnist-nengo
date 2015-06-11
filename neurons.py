import nengo
import numpy as np


def softrelu(x, sigma=1.):
    y = x / sigma
    z = np.array(x)
    z[y < 34.0] = sigma * np.log1p(np.exp(y[y < 34.0]))
    # ^ 34.0 gives exact answer in 32 or 64 bit but doesn't overflow in 32 bit
    return z


class SoftLIFRate(nengo.neurons.LIFRate):
    sigma = nengo.params.NumberParam(low=0)

    def __init__(self, sigma=1., **lif_args):
        super(SoftLIFRate, self).__init__(**lif_args)
        self.sigma = sigma

    @property
    def _argreprs(self):
        args = super(SoftLIFRate, self)._argreprs
        if self.sigma != 1.:
            args.append("sigma=%s" % self.sigma)
        return args

    def rates(self, x, gain, bias):
        J = gain * x + bias
        out = np.zeros_like(J)
        SoftLIFRate.step_math(self, dt=1, J=J, output=out)
        return out

    def step_math(self, dt, J, output):
        """Compute rates in Hz for input current (incl. bias)"""
        j = softrelu(J - 1, sigma=self.sigma)
        output[:] = 0  # faster than output[j <= 0] = 0
        output[j > 0] = 1. / (
            self.tau_ref + self.tau_rc * np.log1p(1. / j[j > 0]))


def s_softrelu(x, sigma):
    import theano.tensor as tt
    y = x / sigma
    return tt.switch(y < 34.0, sigma * tt.log1p(tt.exp(y)), x)


def s_lif(x, tau_ref, tau_rc, gain, bias, amp):
    import theano.tensor as tt
    j = gain * x + bias - 1
    v = amp / (tau_ref + tau_rc * tt.log1p(1. / j))
    return tt.switch(j > 0, v, 0.0)


def s_softlif(x, sigma, tau_ref, tau_rc, gain, bias, amp):
    import theano.tensor as tt
    j = gain * x + bias - 1
    j = s_softrelu(j, sigma)
    v = amp / (tau_ref + tau_rc * tt.log1p(1. / j))
    return tt.switch(j > 0, v, 0.0)


def get_numpy_fn(kind, params):
    if kind == 'lif':
        lif = nengo.LIFRate(tau_rc=params['tau_rc'], tau_ref=params['tau_ref'])
        return lambda x: (
            lif.rates(x, params['gain'], params['bias']) * params['amp'])
    elif kind == 'softlif':
        softlif = SoftLIFRate(
            tau_rc=params['tau_rc'], tau_ref=params['tau_ref'], sigma=params['sigma'])
        return lambda x: (
            softlif.rates(x, params['gain'], params['bias']) * params['amp'])
    else:
        raise ValueError("Unknown neuron type '%s'" % kind)


def get_theano_fn(kind, params):
    import theano
    import theano.tensor as tt

    params = dict(params)
    for param in params:
        params[param] = tt.cast(params[param], dtype=theano.config.floatX)

    if kind == 'lif':
        return lambda x: s_lif(x, **params)
    elif kind == 'softlif':
        return lambda x: s_softlif(x, **params)
    else:
        raise ValueError("Unknown neuron type '%s'" % kind)


def test_theano():
    import theano
    import theano.tensor as tt
    import matplotlib.pyplot as plt

    lif_params = dict(tau_rc=0.02, tau_ref=0.002, gain=1, bias=1, amp=1. / 63.04)
    softlif_params = dict(lif_params)
    softlif_params['sigma'] = 0.01

    x = np.linspace(-1, 1)

    lif = get_theano_fn('lif', lif_params)
    sx = tt.vector()
    lif = theano.function([sx], lif(sx))

    softlif = get_theano_fn('softlif', softlif_params)
    sx = tt.vector()
    softlif = theano.function([sx], softlif(sx))

    y_lif = lif(x)
    y_softlif = softlif(x)

    plt.figure()
    plt.plot(x, y_lif)
    plt.plot(x, y_softlif)
    plt.show()


if __name__ == '__main__':
    test_theano()

import numpy as np
#import from theano
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
#import from lasagne
import lasagne
from lasagne.layers.base import Layer

class RBM(Layer):
    def __init__(self, incoming, num_units, kGibbs=1, numpy_rng=None, theano_rng=None,
                 W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Normal(std=0.01, mean=0.0),
                 c=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.sigmoid,
                 **kwargs):
        super(RBM, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units

        #Parameters init
        self.W = self.add_param(W, (self.num_inputs, self.num_units), name="W")
        self.b = self.add_param(b, (self.num_inputs,), name="b", regularizable=False)
        self.c = self.add_param(c, (self.num_units,), name="c", regularizable=False)

        #Number of steps for Gibbs sampler
        self.kGibbs = kGibbs

        #Setting random generator
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, kGibbs=None, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)

        if kGibbs is None:
            kGibbs=self.kGibbs

        #positive phase
        X0 = input
        [H0_mean, H0_sample] = self.sample_h_given_x(X0)
        [H1_mean, H1_sample] = H0_mean, H0_sample

        for i in xrange(int(kGibbs)):
            [X1_mean, X1_sample] = self.sample_x_given_h(H1_sample)
            [H1_mean, H1_sample] = self.sample_h_given_x(X1_sample)

        #return
        Xk = X1_sample
        Hk = H1_sample
        return Xk

    #Auxiliary functions for generating samples from RBM
    def sample_x_given_h(self, H0):
        X1_mean = T.nnet.sigmoid( T.dot(H0, self.W.T) + self.b )
        X1_sample = self.theano_rng.binomial(size=X1_mean.shape,n=1, p=X1_mean,
                                         dtype=theano.config.floatX) # @UndefinedVariable # get a sample of the visible given their activation
        return [X1_mean, X1_sample]

    def sample_h_given_x(self, X0):
        H1_mean = T.nnet.sigmoid( T.dot(X0, self.W) + self.c )
        H1_sample = self.theano_rng.binomial(size=H1_mean.shape, n=1, p=H1_mean,
                                         dtype=theano.config.floatX)# @UndefinedVariable # get a sample of the hiddens given their activation
        return [H1_mean, H1_sample]

    def free_energy(self,X):
        ''' Function to compute the free energy '''
        X_score = T.dot(X, self.W) + self.c
        vbias_term = T.dot(X, self.b)
        hidden_term = T.sum( T.nnet.softplus(X_score), axis=1)
        return -hidden_term - vbias_term

    def objective(self, X0, Xk):
        return T.mean( self.free_energy(X0) ) - T.mean( self.free_energy(Xk) )

    def predict(self,X0):
        [H0_mean, H0_sample] = self.sample_h_given_x(X0)
        [X1_mean, X1_sample] = self.sample_x_given_h(H0_sample)
        return X1_mean

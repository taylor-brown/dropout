"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import PIL.Image
from collections import OrderedDict
import time

import numpy
import theano
from theano.ifelse import ifelse
import theano.tensor as T
from theano.tensor.elemwise import TensorType
from theano.tensor.nnet import conv
from theano.tensor.nnet.conv3d2d import conv3d
import numpy as np
from sklearn import metrics
from theano.tensor.signal import downsample

from mlp import MLP, _dropout_from_layer

from data_utils import get_data, train_end, test_end, valid_end
from utils import tile_raster_images


dtensor5 = TensorType('float32', (False,) * 5)


def relu(intermediate):
    return (T.sgn(intermediate) + 1.0) * intermediate * .5


class LeNetConvPool3dLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W=None, b=None,
                 poolsize=(2, 2), activation=relu):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: dtensor5
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 5
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width,filter depth)

        :type image_shape: tuple or list of length 5
        :param image_shape: (batch size, num input feature maps,
                             image height, image width, num time steps)

        :type poolsize: tuple or list of length 3
        :param poolsize: the downsampling (pooling) factor (#rows,#cols,#timesteps)
        """

        assert image_shape[2] == filter_shape[2]
        self.input = input
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        if W is None:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width * filter depth" /
            #   pooling size => 1
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / 1)
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
                              borrow=True)
        if b is None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
        self.W = W
        self.b = b

        # convolve input feature maps with filters
        # conv_out = conv.conv2d(input=input, filters=self.W,
        #         filter_shape=filter_shape, image_shape=image_shape)
        conv_out = conv3d(signals=input, filters=self.W,
                          signals_shape=image_shape, filters_shape=filter_shape)

        pooled_out = conv_out # no pooling

        # # downsample each feature map individually, using maxpooling
        # pooled_out_temporal = downsample.max_pool_2d(input=conv_out,
        #                                     ds=poolsize, ignore_border=True)
        # #downsample twice - once for spatial, then over temporal
        # pooled_out = downsample.max_pool_2d(input=pooled_out_temporal,
        #                        ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


        intermediate = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')

        # use rectified linear output
        self.output = activation(intermediate)

        # store parameters of this layer
        self.params = [self.W, self.b]


class DropoutLeNetConvPool3dLayer(LeNetConvPool3dLayer):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), p=0.5):
        super(DropoutLeNetConvPool3dLayer, self).__init__(
            rng=rng, input=input, filter_shape=filter_shape, image_shape=image_shape,
            poolsize=poolsize)

        self.output = _dropout_from_layer(rng, self.output, p=p)


class Conv3dMLP(MLP):
    """
    class for using 3d convolutions
    """

    def get_dropout_layer(self, n_in, n_out, next_dropout_layer_input, rectified_linear_activation, rng, use_bias,
                          previous_layer):
        kern = self.nkerns.pop()
        filter_delta = 2
        if previous_layer:
            batch_size = previous_layer.image_shape[0]
            lwh = previous_layer.image_shape[1] - previous_layer.filter_shape[1] + 1 - filter_delta
            d = previous_layer.filter_shape[2]
            image_shape = (batch_size, lwh, d, lwh, lwh)
            filter_shape = (kern, lwh, d, lwh, lwh)

        else:
            image_shape = self.image_shape
            filter_shape = self.filter_shape

        return DropoutLeNetConvPool3dLayer(rng, input=next_dropout_layer_input,
                                           image_shape=image_shape,
                                           filter_shape=filter_shape, )

    def get_hidden_layer(self, first_layer, n_in, n_out, next_dropout_layer, next_layer_input,
                         rectified_linear_activation, rng, use_bias):

        return LeNetConvPool3dLayer(rng, input=next_layer_input,
                                    image_shape=next_dropout_layer.image_shape,
                                    filter_shape=next_dropout_layer.filter_shape,
                                    W=next_dropout_layer.W * (0.8 if first_layer else 0.5),
                                    b=next_dropout_layer.b, )

    def flatten(self, next_dropout_layer_input, next_layer_input, previous_layer):
        next_dropout_layer_input = next_dropout_layer_input.flatten(2)
        next_layer_input = previous_layer.output.flatten(2)
        return next_dropout_layer_input, next_layer_input

    def __init__(self, rng, input, layer_sizes, batch_size, nkerns, image_shape, filter_shape,
                 append_log_regression=True,
                 first_layer=True):
        self.batch_size = batch_size
        # nkerns.reverse()
        self.nkerns = nkerns
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        super(Conv3dMLP, self).__init__(rng, input, layer_sizes, True, append_log_regression, first_layer)


def get_mlp_default(rng, use_bias, x):
    layer_sizes = [28 * 28, 1200, 1200, 10]
    return MLP(rng=rng, input=x,
               layer_sizes=layer_sizes, use_bias=use_bias)


def get_mlp_conv(rng, use_bias, x):
    batch_size = 64
    print 'warning: hard coded batch size!!!'
    kerns = [7 ** 3 * 4, 32]
    mconv = Conv3dMLP(rng, x, list(kerns), batch_size, list(kerns), (batch_size, 7, 4, 7, 7),
                      filter_shape=(kerns[-1], 5, 4, 5, 5), append_log_regression=False, first_layer=True)

    layer_sizes = [kerns[-1] * 3 ** 3, 1000, 1000, 2]
    return MLP(rng=rng, input=mconv.layers[-1].output.flatten(2),
               layer_sizes=layer_sizes, use_bias=True, inputMlp=mconv)


def get_mlp_relu(rng, use_bias, x):
    layer_sizes = [7 ** 3 * 4, 1000, 1000, 1000, 2]
    return MLP(rng=rng, input=x,
               layer_sizes=layer_sizes, use_bias=True)


def test_mlp(
        initial_learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        batch_size,
        dropout,
        results_file_name,
        dataset,
        use_bias,
        get_mlp=get_mlp_default):
    """
    The dataset is the one from the mlp demo on deeplearning.net.  This training
    function is lifted from there almost exactly.

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    datasets = get_data(True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    data_y = datasets[3]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    epoch = T.scalar()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
                                             dtype=theano.config.floatX))

    rng = np.random.RandomState()

    # construct the MLP class
    classifier = get_mlp(rng, use_bias, x)

    # Build the expresson for the cost function.
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)

    # Compile theano function for testing.
    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(y),
                                 givens={
                                     x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                     y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    # theano.printing.pydotprint(test_model, outfile="test_file.png")
    test_auc = theano.function([index],
                               (classifier.layers[-1]).p_y_given_x,
                               givens={
                                   x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                   # y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                               })

    # Compile theano function for validation.
    validate_model = theano.function(inputs=[index],
                                     outputs=classifier.errors(y),
                                     givens={
                                         x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # validate_auc = theano.function([index], classifier.layers[-1].p_y_given_x,
    validate_auc = theano.function([index],
                                   (classifier.layers[-1]).p_y_given_x,
                                   givens={
                                       x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                       # y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                                   })

    #theano.printing.pydotprint(validate_model, outfile="validate_file.png",
    #        var_with_name_simple=True)

    # Compute gradients of the model wrt parameters
    gparams = []
    for param in classifier.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    # ... and allocate mmeory for momentum'd versions of the gradient
    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)
    T_mom = 50.
    # Compute momentum for the current epoch
    mom = ifelse(epoch < T_mom,
                 # bug fix...
                 (epoch / T_mom) * 0.5 + (1 - epoch / T_mom) * 0.5,
                 0.99)

    # Update the step direction using momentum
    updates = {}
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam

    # ... and take a step along that direction
    for param, gparam_mom in zip(classifier.params, gparams_mom):
        stepped_param = param - learning_rate * updates[gparam_mom]

        # This is a silly hack to constrain the norms of the rows of the weight
        # matrices.  This just checks if there are two dimensions to the
        # parameter and constrains it if so... maybe this is a bit silly but it
        # should work for now.
        if param.get_value(borrow=True).ndim == 2:
            squared_norms = T.sum(stepped_param ** 2, axis=1).reshape((stepped_param.shape[0], 1))
            scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    output = dropout_cost if dropout else cost
    # grads = T.grad(output, classifier.params)
    # updates = []
    # for param_i, grad_i in zip(classifier.params, grads):
    #     updates.append((param_i, param_i - learning_rate * grad_i))

    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.

    train_model = theano.function(inputs=[epoch, index], outputs=output,
                                  updates=updates,
                                  on_unused_input='warn',
                                  givens=OrderedDict({
                                      x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size]
                                  }))
    # theano.printing.pydotprint(train_model, outfile="train_file.png")

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
                                          updates={learning_rate: learning_rate * learning_rate_decay})

    if True:
        for loc, param in enumerate(classifier.params):
            if param.get_value().shape == (32,5,4,5,5):
                print 'saving images...'
                ff = param.get_value()[:,0,0,:,:].reshape((32,25))
                img = PIL.Image.fromarray(tile_raster_images(ff, (5, 5), (3, 5), tile_spacing=(1, 1)))
                img.save("ff-before" + str(loc) + ".png")
                ft = param.get_value()[0, :, 2, :, :].reshape((5, 25))
                img = PIL.Image.fromarray(tile_raster_images(ft, (5, 5), (1, 5), tile_spacing = (1,1)))
                img.save("ft-before" + str(loc) + ".png")


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_params = None
    best_validation_errors = np.inf
    best_iter = 0
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock()
    patient = True
    patience_limit = 15

    # results_file = open(results_file_name, 'wb')

    while epoch_counter < n_epochs and patient:
        # Train this epoch
        epoch_counter = epoch_counter + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(epoch_counter, minibatch_index)

        # Compute loss on validation set
        # validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        # this_validation_errors = np.sum(validation_losses)
        probs = numpy.concatenate([validate_auc(i) for i
                                   in xrange(n_valid_batches)])[:, 1]

        fpr, tpr, _ = metrics.roc_curve(data_y[test_end:test_end + len(probs)], probs)
        this_validation_errors = 1 - metrics.auc(fpr, tpr)

        # only calc test error if needed
        if this_validation_errors < best_validation_errors:
            t_probs = numpy.concatenate([test_auc(i) for i
                                         in xrange(n_test_batches)])[:, 1]
            fpr, tpr, _ = metrics.roc_curve(data_y[train_end:train_end + len(t_probs)], t_probs)
            test_score = 1 - metrics.auc(fpr, tpr)

        # Report and save progress.
        print "epoch {}, valid error {}, learning_rate={}{}".format(
            epoch_counter, this_validation_errors * 100,
            learning_rate.get_value(borrow=True),
            " ** " + str(test_score) if this_validation_errors < best_validation_errors else "")

        if this_validation_errors < best_validation_errors:
            best_iter = epoch_counter
            best_validation_errors = this_validation_errors
            if False:
                for loc, param in enumerate(classifier.params):
                    if param.get_value().shape == (32, 5, 4, 5, 5):
                        print 'saving images...'
                        ff = param.get_value()[:, 0, 0, :, :].reshape((32, 25))
                        img = PIL.Image.fromarray(tile_raster_images(ff, (5, 5), (3, 5), tile_spacing=(1, 1)))
                        img.save("ff-after" + str(epoch_counter) + ".png")
                        ft = param.get_value()[0, :, 2, :, :].reshape((5, 25))
                        img = PIL.Image.fromarray(tile_raster_images(ft, (5, 5), (1, 5), tile_spacing=(1, 1)))
                        img.save("ft-after" + str(epoch_counter) + ".png")

        # results_file.write("{0}\n".format(this_validation_errors))
        # results_file.flush()

        decay_learning_rate()
        if epoch_counter - best_iter > patience_limit:
            patient = False
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_errors * 100., best_iter, test_score * 100.))
    return test_score


def get_mlp_conv_paramd(initial_learning_rate,
                        learning_rate_decay,
                        squared_filter_length_limit,
                        n_epochs,
                        batch_size,
                        dropout,
                        nkerns):
    def get_mlp_conv_local(rng, use_bias, x):
        kerns = [7 ** 3 * 4, nkerns]
        mconv = Conv3dMLP(rng, x, list(kerns), batch_size, list(kerns), (batch_size, 7, 4, 7, 7),
                          filter_shape=(kerns[-1], 5, 4, 5, 5), append_log_regression=False, first_layer=True)

        layer_sizes = [kerns[-1] * 3 ** 3, 1000, 1000, 2]
        return MLP(rng=rng, input=mconv.layers[-1].output.flatten(2),
                   layer_sizes=layer_sizes, use_bias=True, inputMlp=mconv)

    print 'running:', initial_learning_rate, learning_rate_decay, \
        squared_filter_length_limit, n_epochs, batch_size, dropout, nkerns

    return test_mlp(initial_learning_rate=initial_learning_rate,
                    learning_rate_decay=learning_rate_decay,
                    squared_filter_length_limit=squared_filter_length_limit,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    dropout=dropout,
                    dataset='',
                    results_file_name='',
                    use_bias=True,
                    get_mlp=get_mlp_conv_local)

def run_subsample():
    initial_learning_rate = 5.
    learning_rate_decay = 0.95
    squared_filter_length_limit = 50.
    n_epochs = 300
    batch_size = 64
    dropout=True
    kerns = 32

    initial_learning_rate = 4.06
    learning_rate_decay = 0.925925
    squared_filter_length_limit = 39.
    n_epochs = 2973

    tot = []
    for i in range(10):
        tot.append(get_mlp_conv_paramd(initial_learning_rate,
                                   learning_rate_decay,
                                   squared_filter_length_limit,
                                   n_epochs,
                                   batch_size,
                                   dropout,
                                   kerns))
    print 'aucs:', tot
    print 'avg', sum(tot) / 10, 'sd:', numpy.std(tot)

if __name__ == '__main__':
    import sys

    initial_learning_rate = 5.
    learning_rate_decay = 0.95
    squared_filter_length_limit = 15.
    n_epochs = 300
    batch_size = 64

    initial_learning_rate = 4.06
    learning_rate_decay = 0.925925
    squared_filter_length_limit = 39.
    n_epochs = 2973


    # dataset = '/home/taylor/data/mnist/mnist_batches.npy'
    dataset = '/home/taylor/git/DeepLearningTutorials/data/mnist.pkl.gz'

    if len(sys.argv) < 2:
        print "Usage: {0} [dropout|backprop]".format(sys.argv[0])
        exit(1)

    elif sys.argv[1] == "dropout":
        dropout = True
        results_file_name = "results_dropout.txt"

    elif sys.argv[1] == "backprop":
        dropout = False
        results_file_name = "results_backprop.txt"

    else:
        print "I don't know how to '{0}'".format(sys.argv[1])
        exit(1)

    get_mlp_conv_paramd(initial_learning_rate,
                        learning_rate_decay,
                        squared_filter_length_limit,
                        n_epochs,
                        batch_size,
                        dropout,
                        32)
    # test_mlp(initial_learning_rate=initial_learning_rate,
    #          learning_rate_decay=learning_rate_decay,
    #          squared_filter_length_limit=squared_filter_length_limit,
    #          n_epochs=n_epochs,
    #          batch_size=batch_size,
    #          dropout=dropout,
    #          dataset=dataset,
    #          results_file_name=results_file_name,
    #          use_bias=True,
    #          get_mlp=get_mlp_conv)

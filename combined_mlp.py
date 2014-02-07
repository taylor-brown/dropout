from collections import OrderedDict
from sklearn import metrics
from numpy.linalg import linalg
import numpy as np
import theano
from theano.ifelse import ifelse
import theano.tensor as T
from convolutional2d_mlp import ConvMLP

from convolutional_mlp import Conv3dMLP
from data_utils import get_data_dyn_split, test_end, train_end
from mlp import MLP

import pylab as pl

__author__ = 'taylor'


def test_mlp(
        initial_learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        batch_size,
        dropout,
        use_bias,
        get_mlp):
    """
    The dataset is the one from the mlp demo on deeplearning.net.  This training
    function is lifted from there almost exactly.

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """

    datasets = get_data_dyn_split(True)
    train_set_x_stat, train_set_x_dyn, train_set_y = datasets[0]
    valid_set_x_stat, valid_set_x_dyn, valid_set_y = datasets[1]
    test_set_x_stat, test_set_x_dyn, test_set_y = datasets[2]
    data_y = datasets[3]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x_dyn.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x_dyn.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x_dyn.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    epoch = T.scalar()
    x_stat = T.matrix('x_stat')  # the data is presented as rasterized images
    x_dyn = T.matrix('x_dyn')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
                                             dtype=theano.config.floatX))

    rng = np.random.RandomState()

    # construct the MLP class
    classifier = get_mlp(rng, use_bias, x_stat, x_dyn)

    # Build the expresson for the cost function.
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)

    # Compile theano function for testing.
    # theano.printing.pydotprint(test_model, outfile="test_file.png")
    test_auc = theano.function([index],
                               (classifier.layers[-1]).p_y_given_x,
                               givens={
                                   x_stat: test_set_x_stat[index * batch_size:(index + 1) * batch_size],
                                   x_dyn: test_set_x_dyn[index * batch_size:(index + 1) * batch_size],
                                   # y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                               }, on_unused_input='ignore')

    # validate_auc = theano.function([index], classifier.layers[-1].p_y_given_x,
    validate_auc = theano.function([index],
                                   (classifier.layers[-1]).p_y_given_x,
                                   givens={
                                       x_stat: valid_set_x_stat[index * batch_size:(index + 1) * batch_size],
                                       x_dyn: valid_set_x_dyn[index * batch_size:(index + 1) * batch_size],
                                       # y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                                   }, on_unused_input='ignore')

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
                                      x_stat: train_set_x_stat[index * batch_size:(index + 1) * batch_size],
                                      x_dyn: train_set_x_dyn[index * batch_size:(index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size]
                                  }))
    # theano.printing.pydotprint(train_model, outfile="train_file.png")

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
                                          updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_params = None
    best_validation_errors = np.inf
    best_iter = 0
    test_score = 0.
    epoch_counter = 0
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
        probs = np.concatenate([validate_auc(i) for i
                                in xrange(n_valid_batches)])[:, 1]

        fpr, tpr, _ = metrics.roc_curve(data_y[test_end:test_end + len(probs)], probs)
        this_validation_errors = 1 - metrics.auc(fpr, tpr)

        # only calc test error if needed
        if this_validation_errors < best_validation_errors:
            t_probs = np.concatenate([test_auc(i) for i
                                      in xrange(n_test_batches)])[:, 1]
            vfpr, vtpr, _ = metrics.roc_curve(data_y[train_end:train_end + len(t_probs)], t_probs)
            test_score = 1 - metrics.auc(vfpr, vtpr)

        # Report and save progress.
        print "epoch {}, valid error {}, learning_rate={}{}".format(
            epoch_counter, this_validation_errors * 100,
            learning_rate.get_value(borrow=True),
            " ** " + str(test_score) if this_validation_errors < best_validation_errors else "")

        if this_validation_errors < best_validation_errors:
            best_iter = epoch_counter
            best_validation_errors = this_validation_errors

        # results_file.write("{0}\n".format(this_validation_errors))
        # results_file.flush()

        decay_learning_rate()
        if epoch_counter - best_iter > patience_limit:
            patient = False
        # pl.plot(vfpr, vtpr)
    # pl.plot(fpr, tpr)
    # pl.plot([0, 1], [0, 1], 'k--')
    # pl.xlim([0.0, 1.0])
    # pl.ylim([0.0, 1.0])
    # pl.xlabel('False Positive Rate')
    # pl.ylabel('True Positive Rate')
    # pl.title('Receiver operating characteristic example')
    # pl.legend(loc="lower right")
    # pl.show()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_errors * 100., best_iter, test_score * 100.))
    return test_score


def get_mlp_paramd(initial_learning_rate,
                   learning_rate_decay,
                   squared_filter_length_limit,
                   n_epochs,
                   batch_size,
                   dropout,
                   nkerns):
    def get_mlp_conv_local(rng, use_bias, x_stat, x_dyn):
        kerns3d = [7 ** 3 * 2, nkerns]
        mconv3d = Conv3dMLP(rng, x_dyn, list(kerns3d), batch_size, list(kerns3d), (batch_size, 7, 2, 7, 7),
                            filter_shape=(kerns3d[-1], 5, 2, 5, 5), append_log_regression=False, first_layer=True)
        # mlp_3d = MLP(rng=rng, input=mconv3d.layers[-1].output.flatten(2), layer_sizes=[kerns3d[-1] * 3 ** 3, 500],
        #              use_bias=True, append_log_regression=False, first_layer=False)

        kerns2d = [7 ** 2 * 2, nkerns]
        mconv2d = ConvMLP(rng, x_stat, list(kerns2d), batch_size, list(kerns2d), (batch_size, 2, 7, 7),
                          filter_shape=(kerns2d[-1], 2, 5, 5), append_log_regression=False, first_layer=True)
        # mlp_2d = MLP(rng=rng, input=mconv2d.layers[-1].output.flatten(2), layer_sizes=[kerns2d[-1] * 3 * 3, 500],
        #              use_bias=True, append_log_regression=False, first_layer=False)
        layer_sizes = [1000, 1000, 2]
        layer_sizes = [(kerns2d[-1] * 3 * 3) + (kerns3d[-1] * 3 ** 3), 1000, 2]

        # minput = T.concatenate([mlp_3d.layers[-1].output, mlp_2d.layers[-1].output], axis=1)
        minput = T.concatenate([mconv2d.layers[-1].output.flatten(2), mconv3d.layers[-1].output.flatten(2)], axis=1)
        return MLP(rng=rng, input=minput,
                   layer_sizes=layer_sizes, use_bias=True, first_layer=False, inputMlp=[mconv2d, mconv3d])


    print 'running:', initial_learning_rate, learning_rate_decay, \
        squared_filter_length_limit, n_epochs, batch_size, dropout, nkerns

    return test_mlp(initial_learning_rate=initial_learning_rate,
                    learning_rate_decay=learning_rate_decay,
                    squared_filter_length_limit=squared_filter_length_limit,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    dropout=dropout,
                    use_bias=True,
                    get_mlp=get_mlp_conv_local)


if __name__ == '__main__':

    initial_learning_rate = 10.
    learning_rate_decay = 0.95
    squared_filter_length_limit = 15.
    n_epochs = 300
    batch_size = 64
    dropout = True
    nkerns = 32

    initial_learning_rate = 4.06
    learning_rate_decay = 0.925925
    squared_filter_length_limit = 39.
    n_epochs = 2973

    get_mlp_paramd(initial_learning_rate,
                   learning_rate_decay,
                   squared_filter_length_limit,
                   n_epochs,
                   batch_size,
                   dropout,
                   nkerns)


def run_subsample():
    initial_learning_rate = 4.06
    learning_rate_decay = 0.925925
    squared_filter_length_limit = 39.
    n_epochs = 2973
    batch_size = 62
    dropout = True
    kerns = 256

    initial_learning_rate = 4.06
    learning_rate_decay = 0.925925
    squared_filter_length_limit = 39.
    n_epochs = 2973
    kerns=32

    tot = []
    for i in range(10):
        tot.append(get_mlp_paramd(initial_learning_rate,
                                  learning_rate_decay,
                                  squared_filter_length_limit,
                                  n_epochs,
                                  batch_size,
                                  dropout,
                                  kerns))
    print 'aucs:', tot
    print 'avg', sum(tot) / 10, 'sd:', np.std(tot)

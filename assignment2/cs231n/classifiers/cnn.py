from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        w1_dim = (num_filters,input_dim[0], filter_size, filter_size)  #in this case, (32, 3, 7, 7)
        #print('w1_dim is :', w1_dim)
        self.params['W1'] = weight_scale* np.random.randn(w1_dim[0],w1_dim[1], w1_dim[2], w1_dim[3])
        self.params['b1'] = np.zeros(num_filters)
        w2_dim = (hidden_dim, num_filters, input_dim[1]//2,input_dim[2]//2) #in this case, (hidden_dim, 32, 16, 16)
        #print('w2_dim is :', w2_dim)
        self.params['W2'] = weight_scale*np.random.randn(w2_dim[0],w2_dim[1], w2_dim[2], w2_dim[3])
        self.params['b2'] = np.zeros(hidden_dim)
        w3_dim = (num_classes, hidden_dim, 1,1)
        self.params['W3'] = weight_scale*np.random.randn(w3_dim[0],w3_dim[1],w3_dim[2],w3_dim[3])
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        conv_param2 = {'stride':1, 'pad':0}
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_out, cache1 = conv_forward_naive(X, W1, b1, conv_param)
        #print('conv_out.shape is : ',conv_out.shape)
        relu1_out, cache2 = relu_forward(conv_out)
        maxpool_out, cache3 = max_pool_forward_naive(relu1_out, pool_param)
        #print('maxpool_out.shape is :',maxpool_out.shape)
        affine1_out, cache4 = conv_forward_naive(maxpool_out,W2, b2, conv_param2)
        #print('affine1_out.shape is : ',affine1_out.shape)
        relu2_out, cache5 = relu_forward(affine1_out)
        affine2_out, cache6 = conv_forward_naive(relu2_out, W3, b3, conv_param2)
        #print('affine2_out.shape is :',affine2_out.shape)
        affine2_out_reshape = affine2_out.reshape(affine2_out.shape[0],affine2_out.shape[1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        scores = affine2_out_reshape
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, highest_grads = softmax_loss(affine2_out_reshape, y)
        l2_regular = np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)         
        loss += self.reg*l2_regular
        
        highest_grads = highest_grads.reshape(affine2_out.shape)
        
        upcoming_grad, dW3, db3 = conv_backward_naive(highest_grads, cache6)
        upcoming_grad = relu_backward(upcoming_grad, cache5)
        upcoming_grad, dW2, db2 = conv_backward_naive(upcoming_grad, cache4)
        upcoming_grad = max_pool_backward_naive(upcoming_grad, cache3)
        upcoming_grad = relu_backward(upcoming_grad, cache2)
        upcoming_grad, dW1, db1 = conv_backward_naive(upcoming_grad, cache1)
        
        dW3 += self.reg*2*W3
        dW2 += self.reg*2*W2
        dW1 += self.reg*2*W1
        
        grads['W3'] = dW3
        grads['W2'] = dW2
        grads['W1'] = dW1
        grads['b3'] = db3
        grads['b2'] = db2
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

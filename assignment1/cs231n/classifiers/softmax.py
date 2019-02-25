import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_data = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range( num_data ):
    scores = np.dot(X[i], W)
    exp_scores = np.exp(scores)
    #print('exp_scores is : ',exp_scores)
    total = np.sum(exp_scores)
    exp_scores /= total
    loss += -np.log(exp_scores[y[i]])
    dW[: ,y[i]] -= X[i]
    dW[: ,0:y[i]] += np.reshape(X[i],(dW.shape[0],-1))*exp_scores[0:y[i]]
    dW[: ,y[i]+1: ] += np.reshape(X[i] ,(dW.shape[0],-1))*exp_scores[y[i]+1 :]
    #print('shape :',np.reshape(X[i],(dW.shape[0],-1)).shape)
  loss /= num_data
  dW /= num_data
  
  loss += reg*(np.sum(W*W))
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_data = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #Let's find Loss
  XW = np.matmul(X,W)
  XW = np.exp(XW)
  XW_E = np.copy(XW)
  XW /= np.reshape(np.sum(XW, axis = 1), (XW.shape[0],1) )
 
  y_scores = XW[ range(num_data) , y ]
  y_scores = -np.log(y_scores)
  loss = np.sum(y_scores)
  loss /= num_data
  loss += reg*(np.sum(W*W))
    
  #Let's find Gradient
  #W_temp = np.copy(W)
  #XW[ range(num_data) , y ] = -1
  #X_transpose = np.transpose(X)
  
  #XW = np.reshape(np.sum(XW, axis = 0), (1, W.shape[1]))
  #dW = np.matmul(X_transpose,XW)

  #여기서 부터는, 백프로파게이션을 복습하려고 다시 한 것이다. 다시 해도 100퍼센트 이해가 되는것은 아니지만, 
  #backpropagation 의 핵심 아이디어는, 미분을 '나누어서'한다는 것 같다. 따라서 어떠한 복잡한 식이더라도 ,
  #graph형태로 표현하고 chain rule을 사용한다면 기울기를 구할수 있다.
  temp1 = XW
  #temp1[ range(num_data), y] = 1

  temp1[range(num_data),y] -= 1
  dW = X.T @ ( temp1 ) 
  #dW[:,y[i]] = 1
  dW /= num_data
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


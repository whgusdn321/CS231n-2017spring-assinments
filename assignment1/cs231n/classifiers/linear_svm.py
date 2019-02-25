import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  print('ssss')
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_features = X.shape[1] 
  loss = 0.0
  loss2 = 0.0
  WW = np.copy(W)
  h = 0.00001
    
  #find loss first.
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]
        
  loss /= num_train
  dW /= num_train
  loss += reg*np.sum(W*W)
  dW += reg * W
  #dW += 2*W ?? I don't know this part of Calculus
    
  #for a in xrange(num_features):
  #  for b in xrange(num_classes):      
  #    loss2 = 0
  #    WW[a][b] = WW[a][b]+ h
  #    for i in xrange(num_train):        
  #      scores2 = X[i].dot(WW)
  #      correct_class_score2 = scores2[y[i]]
  #      for j in xrange(num_classes):
  #        if j == y[i]:
  #          continue         
  #        margin2 = scores2[j] - correct_class_score2 +1          
  #        if margin2 > 0:
  #          loss2 += margin2
  #    # Right now the loss is a sum over all training examples, but we want it
  #    # to be an average instead so we divide by num_train.
  #    loss2 /= num_train
  #    # Add regularization to the loss.
  #    loss2 += reg * np.sum(WW*WW)
        
      #############################################################################  
      # TODO:                                                                     #
      # Compute the gradient of the loss function and store it dW.                #
      # Rather than first computing the loss and then computing the derivative,   #
      # it may be simpler to compute the derivative at the same time that the     #
      # loss is being computed. As a result you may need to modify some of the    #
      # code above to compute the gradient.                                       #
      #############################################################################\      
  #    gradient = (loss2 - loss) / h 
  #    dW[a][b] = gradient
  #    WW[a][b] = WW[a][b] - h      
      
    
  #print('naive dW is :',dW)
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  xw = np.matmul(X,W)
  #print('before xw is : ',xw)
    
  for i in range(xw.shape[0]):
    xw[i] = xw[i] - xw[i][y[i]] +1

  #print('after xw is :', xw) 

  loss = np.sum( (xw + np.absolute(xw) )/2.0 ) - (num_train)*1
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  xw = np.matmul(X,W)
  #print('ㅋㅋㅋz: ')
    
  for i in range(xw.shape[0]):
    xw[i] = xw[i] - xw[i][y[i]] +1
 
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #first, let's iterate xw[i] that, xw[i][y[i]] <- -9999 so that it cannot be counted as plus
  for i in range( xw.shape[0] ):
    xw[i][y[i]] = -9999
  
  #iterate xw's row and count the number of plus elements, and update w[y[i]]
  for i in range( xw.shape[0] ):
    #print('np.sum(xw[i] >0)  is :\n',(np.sum(xw[i]>0)))
    num_row_plus_elements = np.sum(xw[i]>0)
    dW[:,y[i]] -= (num_row_plus_elements * X[i])
  #iterate xw's column
  for i in range( xw.shape[1] ):
    column_plus_elements = (xw[:,i]>0)
    xx = np.sum(X[column_plus_elements], axis = 0)
    #print('XX is :' ,xx)
    dW[:,i] += xx #it can be errored here!!! 
    
  dW /= xw.shape[0]
  dW += reg * W
  #print('new dw is :\n',dW) 
    
    
    
  #h = 0.00001
  #it = np.nditer(W, flags = ['multi_index'], op_flags =['readonly'])
  #while not it.finished:
  #  ix = it.multi_index
  #  W[ix] += h
    
  #  xw = np.matmul(X,W)    
  #  for i in range(xw.shape[0]):
  #    xw[i] = xw[i] - xw[i][y[i]]+1 
   
  #  loss2 = np.sum( (xw+ np.absolute(xw) )/2 ) - (num_train)*1
  #  loss2 /= num_train
  #  loss2 += reg * np.sum(W * W)
    
  #  dW[ix] = (loss2 - loss)/h
  #  W[ix] -= h
  #  it.iternext()
  #  print('vectorized dW is :',dW)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

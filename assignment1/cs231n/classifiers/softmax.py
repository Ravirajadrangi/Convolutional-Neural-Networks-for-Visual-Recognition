import numpy as np
from random import shuffle


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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    for i, x in enumerate(X):
        scores_i = x.dot(W)

        # compute loss
        loss_i = -scores_i[y[i]] + np.log (np.sum(np.exp(scores_i))) # Li = -Sy + log (sum_over_j{e^Sj})
        loss += loss_i

        # compute gradient, for every column w_j of W
        
        for j, wj in enumerate(W.T):
            the_fraction_ij = np.exp(scores_i[j])/np.sum(np.exp(scores_i))
            if j == y[i]:
                dW[:,j] += x * (the_fraction_ij - 1)
            else:
                dW[:,j] += x * the_fraction_ij

    # So far the loss is a sum of losses over all training examples, but
    # we want it to be an average instead so we divide by num_examples.
    
    num_examples = X.shape[0]
    loss /= num_examples # loss is average loss over all examples
    dW /= num_examples
    
    # Compute regularization loss and gradient
    
    regularization_loss = 0.5 * reg * np.sum(W**2) # sum of squares of individual w
    regularization_dW = reg * W

    # incorporate regularization in outgoing loss and gradient
    
    loss += regularization_loss
    dW += regularization_dW

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_examples = X.shape[0]
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    scores = np.dot(X,W) # compute all scores in one shot

    # compute loss
    
    S_correct = scores[range(num_examples), y] # scores of the correct classes for all examples
    numerator = np.exp(S_correct)
    denominator = np.sum(np.exp(scores), axis=1)
    the_fraction = numerator/denominator
    loss = -np.mean(np.log(the_fraction))
    # a more compact but less readable version for loss computation can be:
    # loss = -np.mean(np.log( np.exp(S_correct)/np.sum(np.exp(S), axis=1)))

    # compute gradient
    
    # I haven't really done any vectorization here, yet. Here's where somebody has:
    # https://github.com/MyHumbleSelf/cs231n/blob/master/assignment1/cs231n/classifiers/softmax.py
    
    for i, x in enumerate(X):
    
        # compute gradient, for every column w_j of W
        for j, wj in enumerate(W.T):
            the_fraction_ij = np.exp(scores[i][j])/np.sum(np.exp(scores[i]))
            dW[:,j] += x * (the_fraction_ij - (j == y[i]))    
    
    dW /= num_examples
    
    # Compute regularization loss and gradient
    
    regularization_loss = 0.5 * reg * np.sum(W**2) # sum of squares of individual w
    regularization_dW = reg * W

    # incorporate regularization in outgoing loss and gradient
    
    loss += regularization_loss
    dW += regularization_dW
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

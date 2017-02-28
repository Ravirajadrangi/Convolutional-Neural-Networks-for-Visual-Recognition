import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
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
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    num_classes = W.shape[1]
    num_examples = X.shape[0]

    loss = 0.0
    dW = np.zeros(W.shape)
    
    for i in xrange(num_examples):
        scores = np.dot(X[i],W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0: # yup, the non-linearity
                loss += margin
                dW[:,j] += X[i,:] # jth column of dW
                dW[:,y[i]] += -X[i,:] # yth column of dW

    # Convert the sums to averages
    loss /= num_examples
    dW /= num_examples

    # Add regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss. Implement a vectorized version of the gradient for the    #
    # structured SVM loss, storing the result in dW.                            #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    num_classes = W.shape[1]
    num_examples = X.shape[0]
    num_dimensions = X.shape[1]

    loss = 0.0
    dW = np.zeros(W.shape)
    
    scores = np.dot(X, W)
    correct_class_scores = scores[range(num_examples), y]

    for i in xrange(num_examples):
        margins = scores[i] - correct_class_scores[i] + 1
        loss += np.sum(margins[margins > 0])
        loss -= 1 # subracting the static '1' generated for j == y[i]
        
        dW[:, margins > 0] += X[i,:].reshape((num_dimensions, 1))
        dW[:,y[i]] -= np.sum(margins > 0) * X[i,:] # yth column of dW
            
    # Convert the sums to averages
    loss /= num_examples
    dW /= num_examples

    # Add regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW

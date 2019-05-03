from __future__ import print_function, division
from builtins import range

import numpy as np
import theano
import theano.tensor as T
import q_learning

# deep neural networks are easier to write with frameworks like theano and tensorflow
# because you don't have to derive any of the gradients yourself

# first we looked at Q learning without any function approximation

# then we looked at Q Learning with linear function approximation
# and gradient descent using scikit learning
	# then we looked at the same method but without using Q Learning and
	# writing the model from scratch with numpy
# now we're going to recreate the same thing in Theano

# this is designed to remind you of all the important parts of a Theano neural network
# (1) Creating graph inputs
# (2) defining shared variables which are parameters that can be updated 
# (3) creating the cost function 
# (4) defining the updates
# (5) compiling functions to do training and prediction

# all we need to do is build an SGDRegressor to overwrite the one from the other Q Learning script
# most of the work is in the constructor
class SGDRegressor:
 def __init__(self, D):
  print("Hello Theano!")
  
  # we initialize w as usual and place it in a theano shared
  w = np.random.randn(D) / np.sqrt(D)
  self.w = theano.shared(w)
  self.lr = 10e-2
  
  # then we create out inputs and targets
  # X is two dimensional
  # Y is one dimenionsal
  X = T.matrix('X')
  Y = T.vector('Y')
  Y_hat = X.dot(self.w)
  delta = Y - Y_hat
  
  #squared error is the cost
  cost = delta.dot(delta)
  grad = T.grad(cost, self.w)
  updates = [(self.w, self.w * self.lr*grad)]
  
  self.train_op = theano.function( 
   inputs=[X,Y], 
   updates=updates,
  )
  self.predict_op = theano.function(
    inputs=[X],
    outputs=Y_hat,
   )
 
 def partial_fit(self, X, Y):
  self.train_op(X, Y)
 
 def predict(self, X):
  return self.predict_op(X)

# all we do is replace Q Learning as the SGDRegressor with the one we just made
if __name__ == '__main__':
 q_learning.SGDRegressor = SGDRegressor
 q_learning.main()
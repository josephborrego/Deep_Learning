from __future__ import print_function, division
from builtins import range

import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

class SGDRegressor:
 def __init__(self, D):
  self.w = np.random.randn(D) / np.sqrt(D)
  self.lr = 0.1
  
 def partial_fit(self, X, Y):
  self.w += self.lr*(Y - X.dot(self.w)).dot(X)
 
 def predict(self, X):
  return X.dot(self.w)
 
 
class FeatureTransformer:
 def __init__(self):
  #observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
  # state samples are poor, b/c you get velocities -> infinity
  observation_samples = np.random.random((20000, 4))*2 - 1
  scaler = StandardScaler()
  scaler.fit(observation_examples)
  
  #used to convert a state to a featurized representaion
  # we use RBF kernels with different variances to cover different parts of the space
  featurizer = Featureunion([
  ("rbf1", RBFSampler(gamma=0.5, n_components=1000)),
  ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
  ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
  ("rbf4", RBFSampler(gamma=0.1, n_components=1000)),
  ])
  feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))
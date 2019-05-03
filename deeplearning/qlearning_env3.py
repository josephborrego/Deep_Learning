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

# SGDRegressor defaults:
# loss='squared_loss', penalty='l2', alpha=0.0001
# l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True
# verbose=0, epsilon=0.1, random_state=None, learning_rate'invscaling',
# eta0=0.01, power_t=0.25, warm_start=False, average=False

# using squared error loss with L2 regularization. A learning rate of 10 ^-4
# an inverse scale learning rate which means it decreases by 1/80

#does all the vector transformations
#we standardize the observations so that they have mean 0 and variance 1
class FeatureTransformer:
 def __init__(self, env):
  observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
  scaler = StandardScaler()
  scaler.fit(observation_examples)
  
  #used to convert a state to a featurizes representation
  #we use RBF kernels with different variances to cover different parts of the space
  #we create a feature union of four rbf samplers with different variances
  #the number of components that we pass into the constructor means the number of exemplars
  #next we fit the rbf samplers to the scaled data
  #then we set the scaler and rbf samplers instance variables so we can use them later in the transform function
  
  featurizer = FeatureUnion([
  ("rbf1", RBFSampler(gamma=5.0, n_components=500)),
  ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
  ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
  ("rbf4", RBFSampler(gamma=0.5, n_components=500))
  ])
  featurizer.fit(scaler.transform(observation_examples))
  
  self.scaler = scaler
  self.featurizer = featurizer
 
 def transform(self, observations):
  scaled = self.scaler.transform(observations)
  return self.featurizer.transform(scaled)



#similar to our feature transform, which was more like a collection of transformers
#our model class is a collection of other models, one model for each action in the constructor

# we assign useful instance variables and instantiate our SGD regressors
# notice how we're calling parital here with a target 0
# this allows us to use the optimisitic initial values method of exploration
# the sgd regressor class requires to call a partial fit before making any predictions
# even if you didn't want to use the optimistic initial values you still have to do this anyway

class Model:
 def __init__(self, env, feature_transformer, learning_rate):
  self.env = env
  self.models = []
  self.feature_transformer = feature_transformer
  for i in range(env.action_space.n):
   model = SGDRegressor(learning_rate = learning_rate)
   model.partial_fit(feature_transformer.transform( [env.reset()]), [0])
   self.models.append(model)
 
 # the predict method transforms the state into a feature vector and makes a prediction of values, one for each action
 # this is returned as a numpy array
 # notice how we put s into a list before we call transform
 # this is because, by convention, data inputs in scikitlearn must be two dimensional
 # a single state is one dimensional
 # this turns it into a NxD matrix where n is one 
 
 def predict(self, s):
 X = self.feature_transformer.transform([s])
 assert(len(X.shape) == 2)
 return np.array([m.predict(X)[0] for m in self.models])
 
 #update function also transforms the input state into a feature vector
 #notice how we call a partial fit for the model that corresponds to the action we took
 #we pass G as a list
 #this is because scikitlearn expects targets to be one dimensional objects whereas G is just a scalar
 
 def update(self, s, a, G):
 X = self.feature_transformer.transform([s])
 assert(len(X.shape) == 2)
 self.models[a].partial_fit(X, [G])
 
 #sample function performs epsilon greedy
 
 def sample_action(self, s, eps):
 # eps = 0
 # Technically, we don't need to do epsilon-greedy
 # because SGDRegressor predicts 0 for all states
 # until they're updated. This works as the 
 # "Optimistic initial values" method, since all
 # the rewards for Mountain Car are -1
 if np.random.random() < eps:
  return self.env.action_space.sample()
 else: 
  return np.argmax(self.predict(s))
  
  
  # we choose out action, take the action, update the model and increment the counter, return the total reward
  # so we can plot them later
 def play_one(model, eps, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters < 10000:
   action = model.sample_action(observation, eps)
   prev_observation = observation
   observation, reward, done, info = env.step(action)
   #update the model
   G = reward + gamma*np.max(model.predict(observation)[0])
   model.update(prev_observation, action, G)
   totalreward += reward
   iters += 1
  return totalreward
  
  
  #this is a plot of the negative of the optimal value function
  # it's plausible in this case because the state is two dimensional which means we can make it a 3d plot

  def plot_cost_to_go(env, estimator, num_tiles=20):
   x = np.linspace(env.observation_space.low[0], env.observation_space.high[0] ,num=num_tiles)
   y = np.linspace(env.observation_space.low[1], env.observation_space.high[1] ,num=num_tiles)
   X, Y = np.meshgrid(x, y)
   # both X and Y will be of shape (num_tiles, num_tiles)
   Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
   # Z will also be of shape (num_tiles, num_tiles)
   
   fig = plt.figure(figsize=(10, 5))
   ax = fig.add_subplot(111, projection='3d')
   surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwrm, vmin=-1.0, vmax=1.0)
   ax.set_xlabel('Position')
   ax.set_ylabel('Velocity')
   ax.set_zlabel('Cost-To-Go == -V(s)')
   ax.set_title("Cost-To-Go Function")
   fig.colorbar(surf)
   plt.show()
  
  # important because of the running average is how you're scored on openai gym
  # you also want to make sure your agents performance is consistent and not just good by chance sometimes
  def plot_running_avg(totalrewards):
   N = len(totalrewards)
   running_avg = np.empty(N)
   for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
   plt.plot(running_avg)
   plt.title("Running Average")
   plt.show()
   
if __name__ == '__main__':
 env = gym.make('MountainCar-v0')
 ft = FeatureTransformer(env)
 model = Model(env, ft, "constant")
 # learning_rate = 10e-5
 # eps = 1.0
 gamma = 0.99
 
 if 'monitor' in sys.argv:
  filename = os.path.basename(__file__).split('.')[0]
  monitor_dir = './' + filename + '_' + str(datetime.now())
  env = wrappers.Monitor(env, monitor_dir)
  
 N = 300
 totalrewards = np.empty(N)
 for i in range(N):
  # eps = 1.0/(0.1*n+1)
  eps = 0.1*(0.97**n)
  # eps = 0.5/np.sqrt(n+1)
  totalreward = play_one(model, eps, gamma)
  totalrewards[n] = totalreward
  print("episode:", n, " total reward:", totalreward)
 print("average reward for last 100 episodes:", totalrewards[-100:].mean())
 print("total steps:" -totalrewards.sum())

 plt.plot(totalrewards)
 plt.title("Rewards")
 plt.show()

 plot_running_avg(totalrewards)

# plot the optimal state value function
 plot_cost_to_go(env, model) 
  
  
  
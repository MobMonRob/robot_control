import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, Bounds, linprog_verbose_callback, minimize_scalar
from scipy.spatial import distance


# Fixing random state for reproducibility
np.random.seed(19680801)
counter = 0

def main(defaultPose, minimalPose, maximalPose, workerTarget):
  print('defaultPose: ', defaultPose)

  n = 50
  xs = []
  ys = []
  zs = []

  plt.close('all')
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(workerTarget[0], workerTarget[1], workerTarget[2], c='r', marker='x')

  # Create the mesh in polar coordinates and compute corresponding Z.
  x = np.linspace(minimalPose[0], maximalPose[0], n)
  y = np.linspace(minimalPose[1], maximalPose[1], n)
  X, Y = np.meshgrid(x, y)
  z = 2

  # Express the mesh in the cartesian system.
  Distance = np.empty([n, n])
  for i in range(n):
      for j in range(n):
          Distance[i][j] = distance.euclidean(workerTarget, [X[i][j], Y[i][j], z])

  print('Distance ', Distance)
  print('X ', X[36][36])
  print('Y ', Y[36][36])
  print('Distance ', Distance[36][36])
  #print('X ', X)
  #print('Y ', Y)

  # Plot the surface.
  ax.plot_surface(X, Y, Distance, cmap=plt.cm.YlGnBu_r)

  # Tweak the limits and add latex math labels.
  #ax.set_zlim(0, 1)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()


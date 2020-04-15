import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, Bounds, linprog_verbose_callback, minimize_scalar
from scipy.spatial import distance


# Fixing random state for reproducibility
np.random.seed(19680801)
counter = 0

def randrange(n, vmin, vmax):
  '''
  Helper function to make an array of random numbers having shape (n, )
  with each number distributed Uniform(vmin, vmax).
  '''
  return (vmax - vmin)*np.random.rand(n) + vmin

def enviormentReward(x, workerTarget):
    global counter 
    counter += 1
    # print('x: ', x, 'counter: ', counter)
    reward = distance.euclidean(workerTarget, x)
    returnValue = np.around(reward, 1) + 0.1*np.random.rand() - 0.05
    # print('reward: ', reward, ' round: ', returnValue)
    return returnValue

def goldenSectionSearch(minimumBorder, maximumBorder):
    c = 0.618
    x1 = minimumBorder + ( 1 - c)*(maximumBorder-minimumBorder)
    x2 = minimumBorder + c*(maximumBorder-minimumBorder)
    return [x1, x2]


  

def main(defaultPose, minimalPose, maximalPose):
  print('defaultPose: ', defaultPose)

  workerTarget = np.array([10, 15, 12])
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  ax.scatter(workerTarget[0], workerTarget[1], workerTarget[2], c='r', marker='x')

  n = 10
  xs = []
  ys = []
  zs = []
  points= [[],[],[]]
  # For each set of style and range settings, plot n random points in the box
  # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].



  bounds = Bounds(minimalPose, maximalPose)
# Nelder-Mead   sehr viele...
# L-BFGS-B      3 // 52
# TNC           3 // 35
# COBYLA        viele, da konstante schrittweite
# SLSQP         7 // 35
# 


  x0 = [0, 0, 0]
  color = ['r', 'b', 'c', 'm', 'y', 'k']
  colorValue = []
  ax.scatter([x0[0]], [x0[1]], [x0[2]], c='g', marker='o')
  global counter 
  if False: # minimalize
    for i in range(7):
      counter = 0
      res = minimize(enviormentReward, x0, (workerTarget), method='SLSQP', tol=1e-6, bounds=bounds, options={'maxiter': i, 'eps': 1e-01})
      print('nit:', res.nit, 'x:', res.x, 'fun', res.fun)
      xs.append(res.x[0])
      ys.append(res.x[1])
      zs.append(res.x[2])
      colorValue.append(color[i])
  elif True: # goldenSectionSearch
    c = 0.618
    x1 = []
    x2 = []
    for i in range(3): #start points per dimension
      [x10, x20] = goldenSectionSearch(minimalPose[i], maximalPose[i])
      x1.append(x10)
      x2.append(x20)
      points[i].append(x1[i])
      points[i].append(x2[i])
    fx1 = enviormentReward(x1, workerTarget)
    fx2 = enviormentReward(x2, workerTarget)
    colorValue.append('g')
    colorValue.append('g')
    for i in range(6):
      print(': ', i)
      print('fx1: ', fx1)
      print('fx1: ', fx2)
      if (fx1 < fx2):
        maximalPose = x2
        x2 = x1
        fx2 = fx1
        for j in range(3):
          x1[j] = minimalPose[j] + ( 1 - c)*(maximalPose[j]-minimalPose[j])
          points[j].append(x1[j])
        fx1 = enviormentReward(x1, workerTarget)
      else:
        minimalPose = x1
        x1 = x2
        fx1 = fx2
        for j in range(3):
          x2[j] = minimalPose[j] + c*(maximalPose[j]-minimalPose[j])
          points[j].append(x2[j])
        fx2 = enviormentReward(x2, workerTarget)
      if(maximalPose[1]-minimalPose[0] < 1):
        print('SUCCESS')
      colorValue.append(color[i])


  ax.scatter(points[0], points[1], points[2], c=colorValue, marker='o')

  #for c, m in [('b', 'o')]:
   # xs.append(randrange(n, minimalPose[0], maximalPose[0]))
    #ys.append(randrange(n, minimalPose[1], maximalPose[1]))
   # zs.append(randrange(n, minimalPose[2], maximalPose[2]))
    #ax.scatter(xs, ys, zs, c=c, marker=m)
  


  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  plt.show()

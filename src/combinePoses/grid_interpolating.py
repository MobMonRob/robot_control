from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# from scipy.spatial import Delaunay
from pyhull.simplex import Simplex
from pyhull.voronoi import VoronoiTess
from pyhull.delaunay import DelaunayTri
import random

def f(x, y, z, target):
  I = np.ones(len(x))
  random.seed(10)
  randInt = np.zeros(len(x))
  for i in range(len(randInt)):
    randInt[i] = random.randint(0, 20)
  print("randInt: ", randInt)
  return np.sqrt((x-I*target[0])**2 + (y-I*target[1])**2 + (z-I*target[2])**2 + randInt**2)


def baryCoeff(A, B, C):
  return np.array([[C[1]-B[1], B[0]-C[0], C[0]*B[1]-B[0]*C[1]],
                   [A[1]-C[1], C[0]-A[0], A[0]*C[1]-C[0]*A[1]],
                   [B[1]-A[1], A[0]-B[0], B[0]*A[1]-A[0]*B[1]]])

def grid_interpolating(defaultPose, minimalPose, maximalPose, workerTarget):
  print('defaultPose: ', defaultPose)


  nx = 20 #50
  ny = 15 #60
  nz = 5
  nxj = 20j
  nyj = 15j
  nzj = 10j

  #points = np.random.rand(8, 3)*10
  points = np.array([[-10, -10, -10], [10, 10, 10], [0, 0, 0], [-10, -10, 10], [-10, 10, -10], [-10, 10, 10],[10, -10, -10], [10, 10, -10], [0, 0, 0], [10, -10, -10], [10, -10, 10], [10, 10, -10]])
  points = np.array([[-10, -10, -10], [-10, -10, 10], [-10, 10, -10], [10, -10, -10], [0, 0, 0], [-5, -5, -5], [5, 5, 5], [0, 5, 5], [5, 5, -5], [5, -5, -5], [-5, 5, -5], [10, 10, 10], [10, 5, 10]])
  #points = np.array([[-10, -10, -10], [-10, -10, 10], [-10, 10, -10], [10, -10, -10]])
  values = f(points[:,0], points[:,1],  points[:,2], workerTarget)
  print('points: ', points)
  print('values; ', values)

  grid_x, grid_y, grid_z = np.mgrid[minimalPose[0]:maximalPose[0]:nxj, minimalPose[1]:maximalPose[1]:nyj, minimalPose[2]:maximalPose[2]:nzj ]

  gridDist = griddata(points, values, (grid_x, grid_y, grid_z), method='linear')


  plt.close('all')
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(workerTarget[0], workerTarget[1], workerTarget[2], c='r', marker='x')

  if False: 
    tri = DelaunayTri(points, joggle='Qz')
    shape = np.shape(grid_x)
    grid_xr = np.reshape(grid_x, -1)
    grid_yr = np.reshape(grid_y, -1)
    grid_zr = np.reshape(grid_z, -1)
    grid_z2 = np.empty(len(grid_zr))
    grid_z2.fill(-1)
    print(grid_z2)
    for simplex in tri.simplices:
      print('points: ', simplex.coords)
      simplexValues = np.zeros(4)
      for idxV, vertex in enumerate(simplex.coords):
        for idx, point in enumerate(points):
          if np.all(point == vertex):
            simplexValues[idxV] = values[idx]
      for idx in range(0, len(grid_xr)):
        if simplex.in_simplex([grid_xr[idx], grid_yr[idx], grid_zr[idx]]):
          grid_z2[idx] = np.dot(simplexValues, np.transpose(simplex.bary_coords([grid_xr[idx], grid_yr[idx], grid_zr[idx]])))
    grid_z2 = np.where(grid_z2==-1, np.nan, grid_z2) 
    grid_z2 = np.reshape(grid_z2, shape)
    #ax.plot_trisurf(points[:,0], points[:,1],  points[:,2], triangles=tri.vertices.copy())
  
  ax.plot(points[:,0], points[:,1], points[:,2], 'o')

  x = np.linspace(minimalPose[0], maximalPose[0], nx)
  y = np.linspace(minimalPose[1], maximalPose[1], ny)
  Y, X = np.meshgrid(y, x)

  norm = mpl.colors.Normalize(vmin=np.nanmin(gridDist), vmax=np.nanmax(gridDist))
  print('m: ', np.nanmin(gridDist), ' M: ', np.nanmax(gridDist))
  #ebenen
  #for index_z, z in enumerate(np.linspace(minimalPose[2], maximalPose[2], nz)):
    #cset = ax.contourf(X, Y, gridDist[:, :, int(np.round(index_z*nzj*-1j/nz))], zdir='z', offset=z, cmap=cm.jet, norm=norm)

  #points
  boxMarkerN = ax.scatter(grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), marker='o', c=gridDist.flatten(), s=20)
  #plt.colorbar(cset, orientation='vertical', extend='both...')  



  # Tweak the limits and add latex math labels.
  #ax.set_zlim(0, 1)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import qLearning
from matplotlib import cm

def plot3D():
    plt.close('all')
    fig = plt.figure()
    boxMarkerN = []
    ax = []
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    ifSurfaces = False #Hauptdiagonalem
    ifCorners = False #
    ifEdges = True
    
    if ifEdges:
        plt.title("Ergonomic Weighting")
        ifSurfaces = True
    
    if ifCorners:
        plt.title("Ergonomic Weighting")
    
    if ifSurfaces:
        plt.title("Ergonomic Weighting")

    plotIndex = 0
    nxj = 2j
    nyj = 2j
    nzj = 2j
    minimumPose = [-1, -1, -1]
    maximumPose = [1, 1, 1]
    minimumPose = np.array(minimumPose)
    maximumPose = np.array(maximumPose)
    #dif = maximumPose - minimumPose
    #minimumPose += -0.2 * dif
    #maximumPose += 0.2 * dif

    grid_x, grid_y, grid_z = np.mgrid[minimumPose[0]:maximumPose[0]:nxj, minimumPose[1]:maximumPose[1]:nyj, minimumPose[2]:maximumPose[2]:nzj ]
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    grid_z = grid_z.flatten()


    ax.append(fig.add_subplot(111, projection='3d'))
    #ax[plotIndex].scatter(points[:,0], points[:,1], points[:,2], marker='x', color='r', s=60)

    boxMarkerN.append(ax[plotIndex].scatter(grid_x, grid_y, grid_z, marker='o', c='k', s=30, norm=norm))
    boxMarkerN.append(ax[plotIndex].scatter(0, 0, 0, marker='o', c='k', s=30, norm=norm))

    if ifCorners:
        cornerPoints = [-1, 0, 1]
    else:
        cornerPoints = [-1, 1]

    for x in cornerPoints:
        for y in cornerPoints:
            for z in cornerPoints:
                for factor in [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]:
                    ax[plotIndex].plot([x, x*factor[0]], [y, y*factor[1]], [z, z*factor[2]], c='k', alpha=.5)
                if ifSurfaces:
                    for factor in [[-1, -1, -1]]: #sufaces
                        ax[plotIndex].plot([x, x*factor[0]], [y, y*factor[1]], [z, z*factor[2]], c='k', alpha=.5)
                if ifEdges:
                    for factor in [[-1, -1, 1], [-1, 1, -1], [1, -1, -1]]:
                        ax[plotIndex].plot([x, x*factor[0]], [y, y*factor[1]], [z, z*factor[2]], c='k', alpha=.5)

    if ifEdges:
        for cornerPoints in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            ax[plotIndex].plot([cornerPoints[0], -cornerPoints[0]], [cornerPoints[1], -cornerPoints[1]], [cornerPoints[2], -cornerPoints[2]], c='k', alpha=.5)
    if ifEdges or ifCorners:
        for cornerPoints in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            boxMarkerN.append(ax[plotIndex].scatter(cornerPoints[0], cornerPoints[1], cornerPoints[2], marker='o', c='k', s=30, norm=norm))
            boxMarkerN.append(ax[plotIndex].scatter(-cornerPoints[0], -cornerPoints[1], -cornerPoints[2], marker='o', c='k', s=30, norm=norm))

    if ifCorners:
        for cornerPoints in [[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0], 
                             [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1], 
                             [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]]:
            boxMarkerN.append(ax[plotIndex].scatter(cornerPoints[0], cornerPoints[1], cornerPoints[2], marker='o', c='k', s=30, norm=norm))

    
    


    boundingBox = [-1.2, 1.2]

    ax[plotIndex].set_xlim(boundingBox[0], boundingBox[1])
    ax[plotIndex].set_ylim(boundingBox[0], boundingBox[1])
    ax[plotIndex].set_zlim(boundingBox[0], boundingBox[1])

    ax[plotIndex].set_xlabel('X')
    ax[plotIndex].set_ylabel('Y')
    ax[plotIndex].set_zlabel('Z')
    # Customize the view angle
    ax[plotIndex].view_init(elev=-5., azim=-30) #azim um z, links händisch
    #fig.colorbar(boxMarkerN[0], shrink=0.5, aspect=5)
    plt.show(block=True)

if __name__ == "__main__":
    #plot3D()
    el = "RightScapulaQ_q0"
    print(el[:el.find("Q")])
import numpy as np
import tiling
import matplotlib.pyplot as plt
import matplotlib as mpl
import qLearning

class ErgonomicMeasure:
    def __init__(self, tilingsShapePure, qValue:qLearning.QValueFunction, samplePoints):
        self.tilingsShape = self.calculateTiling(tilingsShapePure)
        self.qValue = qValue
        self.tilings = tiling.create_tilings(self.tilingsShape)
        self.num_tilings = len(self.tilings)
        self.pose_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in
                            self.tilings]
        self.datastructure = [] #neu
        self.pose_value_tables = [] #neu
        self.samplePoints = samplePoints #0: big Box, 1: small Box
        self.dist_samplePoints_pose = [[[] for samplePoint in boxType ] for boxType in self.samplePoints] #für jeden SamplePoint gibt es eine Liste.
        
        self.radius = 0.25
        self.boxType = 0

    def calculateTiling(self, tilingsShapePure):
        tilingsShapePure["featureRange"] = []
        tilingsShapePure["bins"] = []
        factor = 1
        for box, sampling in zip(tilingsShapePure["boundingBox"], tilingsShapePure["sampling"]):
            tilingsShapePure["featureRange"].append([(box[0]+sampling/2), (box[1]-sampling/2)])
            tilingsShapePure["bins"].append(int(round((box[1]-box[0])*factor/sampling + 1)))
        return tilingsShapePure

    def pose6Dto3D(self, pose):
        return np.array(pose[:3])

    def pose3Dto6D(self, pose):
        newPose = np.zeros(6)
        newPose[:3] = np.array(pose)
        return newPose

    def addRun(self, newRun):
        self.datastructure.append(newRun)
        state = self.qValue.getState(0, self.qValue.stateScene, [newRun])
        valueVectorPairs = self.qValue.valueVectorPairs(state)
        #[{ "displacementVector": displacementVector, "value": value, "actionIdx": self.actions.index(action) }]
        originPose = self.pose6Dto3D(newRun["pose"])
        for vector in valueVectorPairs:
            if not(np.isnan((vector["value"]))):
                self.pose_value_tables.append({"pose": originPose + vector["displacementVector"], "value": vector["value"]})
                for samplePoint, dist_samplePoint_pose in zip(self.samplePoints[self.boxType], self.dist_samplePoints_pose[self.boxType]):
                    distance = np.linalg.norm(samplePoint["pose"] - (originPose + vector["displacementVector"]))
                    dist_samplePoint_pose.append({
                        "dist": distance,
                        "value": vector["value"],
                        "distWeight": 1/distance if distance > 0.02 else 50})

    def extrapolateData(self, boxType, index, radius):
        power = 3
        weights = 0
        result = 0
        counter = 0
        #newStorage = []
        for el in self.dist_samplePoints_pose[boxType][index]:
            if el["dist"] < radius: #TODO evaluate distance measure; höhere Kontrast bei < 0.2
                counter +=1
                weight = np.power(el["distWeight"], power)
                result +=el["value"] * weight
                weights += weight
                #newStorage.append((el["dist"], weight, el["value"] * weight))
        if weights == 0:
            return 0
        else:
           # newStorage.sort(key=lambda tup: tup[0])
            #newStorage = newStorage[: 20 if len(newStorage) > 20 else len(newStorage)]
          #  print(len(newStorage), " res/wei: ", result/weights, " - ", sum([el[2] for el in newStorage])/sum([el[1] for el in newStorage]))
           # print("counter: ", counter)
            return result/weights

    #extrapolate:
        #ein punkt wird abgefragt, danach wird mithilfe dem abstand zu allen anderen Punkten (der Posen) eine Gewichtung berechnet
        #die Gewichtung multipliziert mit dem jeweiligen Value ergibt den Value für den abgefragten Punkt
            #je nachdem eingrenzung der einzubeziehenden Punkte (maximale dist, punkte darpber hinaus gehen mit einer gewichtung von 0 ein - alternativ die n (oder X%) nächsten Punkte)
        #diese Funktion kann auch für ein 3D Plot aufgerufen werden.




    def printErgonomicMeasure3D(self, boundingBox=None, radius=None):
        if radius is None:
            radius = 0.3

        plt.close('all')
        fig = plt.figure()
        boxMarkerN = []
        ax = []
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        gridValMin = 1
        gridValMax = 0

        plotIndex = 0
        #nxj = 10j
        #nyj = 10j
        #nzj = 10j
        #minimumPose = []
        #maximumPose = []
        #for i in range(3):
        #    minimumPose.append(self.tilings[0][i][0])
        #    maximumPose.append(self.tilings[-1][i][-1])
        #minimumPose = np.array(minimumPose)
        #maximumPose = np.array(maximumPose)
        #dif = maximumPose - minimumPose
        #minimumPose += -0.2 * dif
        #maximumPose += 0.2 * dif

        #grid_x, grid_y, grid_z = np.mgrid[minimumPose[0]:maximumPose[0]:nxj, minimumPose[1]:maximumPose[1]:nyj, minimumPose[2]:maximumPose[2]:nzj ]
        #grid_x = grid_x.flatten()
        #grid_y = grid_y.flatten()
        #grid_z = grid_z.flatten()
        #gridVal = np.ndarray(np.shape(grid_x)[0])
        #for i in range(np.shape(grid_x)[0]):
        #    gridVal[i] = self.valuePoseActionTables([grid_x[i], grid_y[i], grid_z[i]], action)
        #gridVal = np.where(gridVal==0, np.NaN, gridVal)

        grid_x = []
        grid_y = []
        grid_z = []
        gridVal = []
        for i, samplePoint in enumerate(self.samplePoints[self.boxType]):
            grid_x.append(samplePoint["pose"][0])
            grid_y.append(samplePoint["pose"][1])
            grid_z.append(samplePoint["pose"][2])
            gridVal.append(self.extrapolateData(self.boxType, i, radius))

        ax.append(fig.add_subplot(111, projection='3d'))
        gridValMin = min([gridValMin, np.nanmin(gridVal)])
        gridValMax = max([gridValMax, np.nanmax(gridVal)])
        # print( " gridValMin: ", gridValMin, "gridValMax: ", gridValMax)

        points = np.ndarray((len(self.datastructure), 3))
        for i, data in enumerate(self.datastructure):
            points[i, :] = self.pose6Dto3D(data["pose"])
        ax[plotIndex].scatter(points[:,0], points[:,1], points[:,2], marker='x', color='r', s=60)

        print("No: ", plotIndex, ' m: ', np.nanmin(gridVal), ' M: ', np.nanmax(gridVal))
        boxMarkerN.append(ax[plotIndex].scatter(grid_x, grid_y, grid_z, marker='o', c=gridVal, s=30, norm=norm))

        if not(boundingBox == None):
            ax[plotIndex].set_xlim(boundingBox[0][0], boundingBox[0][1])
            ax[plotIndex].set_ylim(boundingBox[1][0], boundingBox[1][1])
            ax[plotIndex].set_zlim(boundingBox[2][0], boundingBox[2][1])

        ax[plotIndex].set_xlabel('X')
        ax[plotIndex].set_ylabel('Y')
        ax[plotIndex].set_zlabel('Z')
        plt.title("Ergonomic Weighting")
        # Customize the view angle
        ax[plotIndex].view_init(elev=-5., azim=-30) #azim um z, links händisch
        fig.colorbar(boxMarkerN[0], shrink=0.5, aspect=5)
        plt.show(block=True)


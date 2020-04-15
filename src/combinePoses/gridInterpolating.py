from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import tiling
import random

class GridInterpolating:
    def __init__(self, tilingsShapePure, samplePoints):
        # tiling_shape ist raum der Poses ~> Werte für die Distanzen werden eingezeichnet
        # ich kann extrapolieren über die Distanzen - mit allen erlaubten Tricks
        # ich kann live ein Koordinatensystem implementieren und ausrechnen wo der Roboter das Paket hinliefern soll ~> anschließend über time weiter optimieren
        self.tilingsShape = self.calculateTiling(tilingsShapePure)
        self.tilings = tiling.create_tilings(self.tilingsShape)
        self.num_tilings = len(self.tilings)
        self.pose_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in
                            self.tilings]
        self.distance_tables = self.resetDistanceTable()
        self.samplePoints = samplePoints #0: big Box, 1: small Box
        self.boxType = 0
        self.datastructure = []

    def calculateTiling(self, tilingsShapePure):
        tilingsShapePure["featureRange"] = []
        tilingsShapePure["bins"] = []
        factor = 1
        for box, sampling in zip(tilingsShapePure["boundingBox"], tilingsShapePure["sampling"]):
            tilingsShapePure["featureRange"].append([(box[0]+sampling/2), (box[1]-sampling/2)])
            tilingsShapePure["bins"].append(int(round((box[1]-box[0])*factor/sampling + 1)))
        return tilingsShapePure

    def resetDistanceTable(self):
        return [np.zeros(shape=(pose_size)) for pose_size in self.pose_sizes]

    def value(self, state): #value der Distance-Table
        state_codings = tiling.get_tile_coding(state, self.tilings)
        value = 0
        for coding, distance_table in zip(state_codings, self.distance_tables):
            value += distance_table[tuple(coding)]
        return value / self.num_tilings

    def setValue(self, state, value): #value der Distance-Table
        state_codings = tiling.get_tile_coding(state, self.tilings)
        for coding, distance_table in zip(state_codings, self.distance_tables):
            distance_table[tuple(coding)] = value
            

    def pose6Dto3D(self, pose):
        return np.array(pose[:3])

    def pose3Dto6D(self, pose):
        newPose = np.zeros(6)
        newPose[:3] = np.array(pose)
        return newPose

    def addRun(self, data):
        #print("Dist: ", data["path"][0]["distance"], "Duration: ", data["path"][0]["duration"])
        #print("ScoreGet: ", data["score"]["getBox"], "ScoreDeposit: ", data["score"]["depositBox"])
        # data = {'
        #           handCenterAvg': {
        #               'depositBox': {'timestamp': 692.5, 'x': -669.9995580645161, 'y': 175.13632148064517, 'z': 977.4173903225809}, 
        #               'getBox': {'timestamp': 428.0, 'x': -495.8644473684211, 'y': -111.67252105263155, 'z': 1226.3176315789476}}, 
        #           'jointAvg': {
        #               'depositBox': {'LELBW': 130.20615683883068, 'LSHOULDER2D': 58.988181959577716, 'OUTOFMIDDLEDISTANCE': 20.495496948571628, 'RELBW': 112.2459169185274, 'RSHOULDER2D': 99.56097362774788, 'timestamp': 692.5}, 
        #               'getBox': {'LELBW': 105.80730431280043, 'LSHOULDER2D': 41.82232366011042, 'OUTOFMIDDLEDISTANCE': 173.70418795184082, 'RELBW': 91.35137327078805, 'RSHOULDER2D': 51.819210479844536, 'timestamp': 428.0}}, 
        #           'nameIndex': '001', 
        #           'path': [
    #                   {
        #                   'distance': 417.7722813087254, 
        #                   'duration': 264.5, 
        #                   'end': 'depositBox', 
        #                   'path': array([-174.1351107 ,  286.80884253, -248.90024126]), 
        #                   'start': 'getBox'}], 
        #           'pose': array([ 0.1,  0.4, -0.1,  0. ,  0. ,  0. ]), 
        #           'poseIndex': 1, 
        #           'score': {'depositBox': 0.07099126987882937, 'getBox': 0.25017562874233623}, 
        #           'action': {'halt': 62.50175628742336, 'back': 60.0, 'front': None, 'left': 64.2644981499922, 'right': None, 'down': None, 'up': None, 'bottomBack': None, 'bottomFront': None, 'bottomLeft': None, 'bottomRight': None, 'backLeft': 64.2644981499922, 'backRight': None, 'frontLeft': None, 'frontRight': None, 'topBack': 60.0, 'topFront': None, 'topLeft': None, 'topRight': None, 'backLeftBottom': None, 'backRightBottom': None, 'frontLeftBottom': None, 'frontRightBottom': None, 'backLeftTop': None, 'backRightTop': None, 'frontLeftTop': None, 'frontRightTop': None}
        #         }
        self.datastructure.append(data)

    def scaling0to1(self, values):
        minValue = np.nanmin(values)
        maxValue = np.nanmax(values)
        if (not (minValue is np.NaN)) and (not (maxValue is np.NaN)):
            return (values-minValue) / (maxValue - minValue)
        else:
            return values

    def calcDistanceTable3D(self, doNotPlot):
        ifGridFixRaster = False
        plt.close('all')
        fig = plt.figure()
        ax = []
        boxMarkerN = []
        gridVal = None
        for plotNumber in range(1):
            nxj = np.shape(self.distance_tables)[1] * 1j #TODO: ist das so richtig oder sollte das die Breite der Bereiche werden?
            nyj = np.shape(self.distance_tables)[2] * 1j
            nzj = np.shape(self.distance_tables)[3] * 1j
            minimumPose = []
            maximumPose = []
            differencePose = []
            for i in range(3):
                minimumPose.append(self.tilingsShape["boundingBox"][i][0])
                maximumPose.append(self.tilingsShape["boundingBox"][i][1])
                differencePose.append(maximumPose[i]-minimumPose[i])
            minimumPose = np.array(minimumPose)
            maximumPose = np.array(maximumPose)

            if ifGridFixRaster:
                grid_x, grid_y, grid_z = np.mgrid[minimumPose[0]:maximumPose[0]:nxj, minimumPose[1]:maximumPose[1]:nyj, minimumPose[2]:maximumPose[2]:nzj ]
                grid_x = grid_x.flatten()
                grid_y = grid_y.flatten()
                grid_z = grid_z.flatten()
            else:
                grid_x = []
                grid_y = []
                grid_z = []
                for i, samplePoint in enumerate(self.samplePoints[self.boxType]):
                    grid_x.append(samplePoint["pose"][0])
                    grid_y.append(samplePoint["pose"][1])
                    grid_z.append(samplePoint["pose"][2])
            
            points = np.ndarray((len(self.datastructure)+8, 3))
            valuesDistance = np.zeros((len(self.datastructure)+8)) #TODO: durch zero und die zusätzlichen nullen, bekommt das 0-1scalling eine feste Stütztstelle...; wird unten jedoch raus genommen!
            valuesDuration = np.zeros((len(self.datastructure)+8))

            for i, data in enumerate(self.datastructure):
                points[i, :] = self.pose6Dto3D(data["pose"])
                valuesDistance[i] = data["path"][0]["distance"] # Scalar value is the distance the smaller the better #TODO test other measurements
                valuesDuration[i] = data["path"][0]["duration"]

            idx_nan = np.array([~np.isnan(valuesDistance), ~np.isnan(valuesDuration)]).all(axis=0) #Bei Nan-werten wird der gesamte Punkt entfernt
            points = points[idx_nan]
            valuesDistance = valuesDistance[idx_nan]
            valuesDuration = valuesDuration[idx_nan]

            valuesDistance = list(1-self.scaling0to1(valuesDistance)) # 1 is the best, 0 the worst     
            valuesDuration = list(1-self.scaling0to1(valuesDuration)) # 1 is the best, 0 the worst

            #print("ValDistance: m: ", np.amin(valuesDistance), " ", np.amax(valuesDistance))
            #print("ValDuration: m: ", np.amin(valuesDuration), " ", np.amax(valuesDuration))
            
            factor = 10
            for i in range(2):
                x = minimumPose[0]-factor*differencePose[0] if i == 0 else maximumPose[0]+factor*differencePose[0]
                for j in range(2):
                    y = minimumPose[1]-factor*differencePose[1] if j == 0 else maximumPose[1]+factor*differencePose[1]
                    for k in range(2):
                        z = minimumPose[2]-factor*differencePose[2] if k == 0 else maximumPose[2]+factor*differencePose[2]
                        points[-(i*4+j*2+k+1), :] = np.array([x, y, z])
                        valuesDistance[-(i*4+j*2+k+1)] = 0
                        valuesDuration[-(i*4+j*2+k+1)] = 0
            
            gridValDistance = np.zeros(np.shape(grid_x))
            gridValDuration = np.zeros(np.shape(grid_x))
            #TODO vor dem interpolieren ein Tiling mit durchschnittswertbildung innerhalb eines Bereichs ~> dadurch Mittelung von verschiedenen Ergebnissen
            if len(points) > 3 and len(self.datastructure) > 1: #zum interpolierne mindestens 4 Punkte, zum 0-1scaling mindestens 2 Punkte
                gridValDistance = griddata(points, valuesDistance, (grid_x, grid_y, grid_z), method='linear') #linear interpolating #TODO test other interpolations
                #alternativ: https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
                gridValDuration = griddata(points, valuesDuration, (grid_x, grid_y, grid_z), method='linear') #linear interpolating #TODO test other interpolations


            #print("gridValDistance: m: ", np.amin(gridValDistance), " ", np.amax(gridValDistance))
            #print("gridValDuration: m: ", np.amin(gridValDuration), " ", np.amax(gridValDuration))

            # Addition / Maximum: #TODO: decide / evaluate
            gridVal = gridValDistance + gridValDuration #beide Werte sind im selben Bereich skaliert, beide Werte müssen gut sein, ein besonders guter Wert kann den anderen kompensieren 
            #gridVal = np.amax(np.array([gridValDuration, gridValDistance]), axis=0) #der beste Wert bestimmt, es ist dann egal wie schlecht der andere Wert ist
            #gridVal = np.amin(np.array([gridValDuration, gridValDistance]), axis=0) #der minimale Wert bestimmt, es ist dann egal wie gut der andere Wert ist
            
            if False: #TODO: Activate or deactivate this parameter
                #Generates a Array with only x percentageOfNonNaNPoints
                percentageOfNonNaNPoints = 0.2 #TODO: evaluate this parameter
                numberOfNonNaNElements = gridVal.size - np.count_nonzero(np.isnan(gridVal))
                print("numberOfNonNaNElements: ", numberOfNonNaNElements)
                quantityOfPoints = int(round(numberOfNonNaNElements * percentageOfNonNaNPoints))
                indices = np.unravel_index(np.argpartition(gridVal, quantityOfPoints, axis=None)[:quantityOfPoints], np.shape(gridVal))
                newGridVal = np.zeros(np.shape(grid_x))
                newGridVal[indices] = gridVal[indices]
                newGridVal = np.where(newGridVal == 0, np.NaN, newGridVal)
                gridVal = newGridVal
            gridVal = self.scaling0to1(gridVal)

            #print("gridVal: m: ", np.amin(gridVal), " ", np.amax(gridVal))
            
            self.distance_tables = self.resetDistanceTable()
            for i in range(len(grid_x)):
                self.setValue([grid_x[i], grid_y[i], grid_z[i]], gridVal[i])

            if doNotPlot == True:
                return

            ax.append(fig.add_subplot(111 + plotNumber, projection='3d'))

            ax[plotNumber].scatter(points[:,0], points[:,1], points[:,2], marker='x', color='r', s=60)

            gridValMin = 0 if np.isnan(np.nanmin(gridVal)) else np.nanmin(gridVal)
            gridValMax = 1 if np.isnan(np.nanmax(gridVal)) else np.nanmax(gridVal)
            if gridValMin < gridValMax:
                norm = mpl.colors.Normalize(vmin=gridValMin, vmax=gridValMax)
            else:
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
            boxMarkerN.append(ax[plotNumber].scatter(grid_x, grid_y, grid_z, marker='o', c=gridVal, s=30, norm=norm))

            #ax[plotNumber].set_aspect('equal')
            ax[plotNumber].set_xlim(minimumPose[1], maximumPose[1])
            ax[plotNumber].set_ylim(minimumPose[1], maximumPose[1])
            ax[plotNumber].set_zlim(minimumPose[2], maximumPose[2])
            ax[plotNumber].set_xlabel('X')
            ax[plotNumber].set_ylabel('Y')
            ax[plotNumber].set_zlabel('Z')
            plt.title("Where to go")
            # Customize the view angle
            ax[plotNumber].view_init(elev=-5., azim=-30) #azim um z, links händisch
        fig.colorbar(boxMarkerN[0], shrink=0.5, aspect=5)
        plt.show()

        return gridVal

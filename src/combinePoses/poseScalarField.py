import numpy as np
import tiling
import matplotlib.pyplot as plt
import matplotlib as mpl
import qLearning

class PoseScalarField:
    def __init__(self, tilingsShapePure, qValue:qLearning.QValueFunction):
        self.tilingsShape = self.calculateTiling(tilingsShapePure)
        self.qValue = qValue
        self.tilings = tiling.create_tilings(self.tilingsShape)
        self.num_tilings = len(self.tilings)
        self.pose_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in
                            self.tilings]
        self.pose_tables = self.resetPoseTable()
        self.pose_action_tables = self.resetPoseActionTable()


    def calculateTiling(self, tilingsShapePure):
        tilingsShapePure["featureRange"] = []
        tilingsShapePure["bins"] = []
        factor = 1
        for box, sampling in zip(tilingsShapePure["boundingBox"], tilingsShapePure["sampling"]):
            tilingsShapePure["featureRange"].append([(box[0]+sampling/2), (box[1]-sampling/2)])
            tilingsShapePure["bins"].append(int(round((box[1]-box[0])*factor/sampling + 1)))
        return tilingsShapePure

    def resetPoseTable(self):
        return [np.zeros(shape=(pose_size)) for pose_size in self.pose_sizes]

    def resetPoseActionTable(self):
        return [np.zeros(shape=(pose_size + (self.qValue.actions.length,))) for pose_size in self.pose_sizes]

    def pose6Dto3D(self, pose):
        return np.array(pose[:3])

    def pose3Dto6D(self, pose):
        newPose = np.zeros(6)
        newPose[:3] = np.array(pose)
        return newPose

    def getScalarField(self, dataStructure, pose6D):
        dataStructure = [{"jointAvg": dataStructure}]
        self.pose_tables = self.resetPoseTable()
        print("self.pose_tables: ", self.pose_tables)
        state = self.qValue.getState(0, self.qValue.stateScene, dataStructure)
        valueVectorPairs = self.qValue.valueVectorPairs(state)
        print("valueVectorPairs: ", valueVectorPairs)
        originPose = self.pose6Dto3D(pose6D)
        pose_codings = tiling.get_tile_coding(originPose, self.tilings)
        codingTable = []
        for element in valueVectorPairs:
            displacementVector = np.array(element["displacementVector"])
            newPose = originPose + displacementVector #TODO: ist das hier richtig?
            pose_codings = tiling.get_tile_coding(newPose, self.tilings)
            value = element["value"]
            for coding, pose_table in zip(pose_codings, self.pose_tables):
                if np.any(coding[0] + coding[1]*100 + coding[2]*10000 in codingTable):
                    print("bereits vorhanden... wähle anderes sampling: ", coding)
                else:
                    codingTable.append(coding[0] + coding[1]*100 + coding[2]*10000)
                pose_table[tuple(coding)] = value

    def printPoseTable3D(self):
        # für eine pose alle actions
        plt.close('all')
        fig = plt.figure()
        ax = []
        boxMarkerN = []
        gridValMin = 1
        gridValMax = 0
        for plotNumber in range(1):
            nxj = np.shape(self.pose_tables)[1] * 1j
            nyj = np.shape(self.pose_tables)[2] * 1j
            nzj = np.shape(self.pose_tables)[3] * 1j
            minimumPose = []
            maximumPose = []
            for i in range(3):
                minimumPose.append(self.tilingsShape["boundingBox"][i][0])
                maximumPose.append(self.tilingsShape["boundingBox"][i][1])
            minimumPose = np.array(minimumPose)
            maximumPose = np.array(maximumPose)

            grid_x, grid_y, grid_z = np.mgrid[minimumPose[0]:maximumPose[0]:nxj, minimumPose[1]:maximumPose[1]:nyj, minimumPose[2]:maximumPose[2]:nzj ]
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()
            grid_z = grid_z.flatten()
            gridVal = np.zeros(np.shape(grid_x)[0])
            for i in range(np.shape(grid_x)[0]):
                pose_codings = tiling.get_tile_coding([grid_x[i], grid_y[i], grid_z[i]], self.tilings)
                for coding, pose_table in zip(pose_codings, self.pose_tables):
                    #print("coding: ", coding, " pose; ", pose_table[tuple(coding)])
                    gridVal[i] += pose_table[tuple(coding)]
                gridVal[i] /= self.num_tilings
            gridVal = np.where(gridVal == 0, np.NaN, gridVal)

            ax.append(fig.add_subplot(111 + plotNumber, projection='3d'))
            gridValMin = min([gridValMin, np.nanmin(gridVal)])
            gridValMax = max([gridValMax, np.nanmax(gridVal)])
            # print( " gridValMin: ", gridValMin, "gridValMax: ", gridValMax)
            norm = mpl.colors.Normalize(vmin=gridValMin, vmax=gridValMax)
            print("No: ", plotNumber, ' m: ', np.nanmin(gridVal), ' M: ', np.nanmax(gridVal))
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

    def getDecisionScalarField(self, dataStructure):
        print("len: ", len(dataStructure))
        #schreibt alle elemente der Datastructure in die pose_action_tables
        self.pose_action_tables = self.resetPoseActionTable()
        for el in dataStructure:
            state = self.qValue.getState(0, self.qValue.stateScene, [el])
            valueVectorPairs = self.qValue.valueVectorPairs(state)
            #valueVectorPairs =  [{'displacementVector': array([0., 0., 0.]), 'value': nan, 'actionIdx': 0}, 
            # {'displacementVector': array([-0.1,  0. ,  0. ]), 'value': nan, 'actionIdx': 1},....]
            # el = {'handCenterAvg': 
            # {'depositBox': {'timestamp': 398.0, 'x': -607.092, 'y': 209.03591775, 'z': 1231.0925},
            #  'getBox': {'timestamp': 286.0, 'x': -506.46900000000005, 'y': -272.51025, 'z': 1341.4375}}, 
            # 'jointAvg':
            #  {'depositBox': {'HANDDISTANCE': 415.4124850699847, 'LELBW': 116.20698380983922, 'LFOREARM': 225.95153472018518, 'LSHOULDER2D': 48.9902370141013, 'LUPPERARM': 413.29950395888295, 'OUTOFMIDDLEDISTANCE': 183.42403653492383, 'RELBW': 71.18893798968479, 'RFOREARM': 236.50801522081682, 'RSHOULDER2D': 13.73358209986485, 'RUPPERARM': 418.8775285707044, 'SHOULDERDISTANCE': 340.91170633138495, 'timestamp': 398.0}, 
            # 'getBox': {'HANDDISTANCE': 414.63783559934245, 'LELBW': 123.27435898724582, 'LFOREARM': 215.563763188211, 'LSHOULDER2D': 65.05766101371253, 'LUPPERARM': 400.0593395108174, 'OUTOFMIDDLEDISTANCE': 160.31681215348596, 'RELBW': 122.6475539529318, 'RFOREARM': 232.98312677688182, 'RSHOULDER2D': 65.59852835250899, 'RUPPERARM': 382.69829045613574, 'SHOULDERDISTANCE': 302.20686433016147, 'timestamp': 286.0}}, 
            # 'nameIndex': '390', 
            # 'path': [{'distance': 504.1703271997581, 'duration': 112.0, 'end': 'depositBox', 'path': array([-100.623     ,  481.54616775, -110.345     ]), 'start': 'getBox'}], 
            # 'pose': array([-0.1,  0.4,  0. ,  0. ,  0. ,  0. ]), 'poseIndex': 387, 'score': {'depositBox': 0.26981563316141316,'getBox': 0.0}}
            originPose = self.pose6Dto3D(el["pose"]) #es wird für die jeweilige Pose angegeben, ob eine Action empfohlen wird
            pose_codings = tiling.get_tile_coding(originPose, self.tilings)
            for element in valueVectorPairs:
                if not np.isnan(element["value"]):
                    for coding, pose_table in zip(pose_codings, self.pose_action_tables):
                        if pose_table[tuple(coding)+ (element["actionIdx"],)] == 0:
                            pose_table[tuple(coding)+ (element["actionIdx"],)] = element["value"]
                        else:
                            pose_table[tuple(coding)+ (element["actionIdx"],)] = (element["value"]+pose_table[tuple(coding)+ (element["actionIdx"],)])/2
                        

    def valuePoseActionTables(self, pose, action):
        #liest eine pose mit iener action aus der pose_action_tables
        pose_codings = tiling.get_tile_coding(pose, self.tilings)  # [[5, 1], [4, 0], [3, 0]] ...
        action_idx = self.qValue.actions.index(action)
        value = 0
        for coding, pose_action_table in zip(pose_codings, self.pose_action_tables):
            value += pose_action_table[tuple(coding) + (action_idx,)]
        return value / self.num_tilings


    def printAllPosesForEachAction3D(self, dataStructure, boundingBox=None):
        #durch alle states gehen und immer den value für die entsprechende Action rausschreiben
        self.getDecisionScalarField(dataStructure) #generate data

        for plot in [{"plot": [0, 1, 2, 3, 4, 5, 6], "plotFormat": 241-0,}, {"plot": [7, 8, 9, 10, 11, 12, 13, 14], "plotFormat": 241}, {"plot": [15, 16, 17, 18, 19, 20], "plotFormat": 231}, {"plot": [21, 22, 23, 24, 25, 26], "plotFormat": 231}]:
            plt.close('all')
            fig = plt.figure()
            boxMarkerN = []
            ax = []
            #norm = mpl.colors.Normalize(vmin=0, vmax=0.7)
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            gridValMin = 1
            gridValMax = 0
                
            for plotNumber, action in enumerate(self.qValue.actions.keys()):
                if not(plotNumber in plot["plot"]): # ToDo shows only the main directions
                    continue
                plotIndex = plotNumber-plot["plot"][0]
                nxj = 10j
                nyj = 10j
                nzj = 10j
                minimumPose = []
                maximumPose = []
                for i in range(3):
                    minimumPose.append(self.tilings[0][i][0])
                    maximumPose.append(self.tilings[-1][i][-1])
                minimumPose = np.array(minimumPose)
                maximumPose = np.array(maximumPose)
                dif = maximumPose - minimumPose
                minimumPose += -0.2 * dif
                maximumPose += 0.2 * dif

                grid_x, grid_y, grid_z = np.mgrid[minimumPose[0]:maximumPose[0]:nxj, minimumPose[1]:maximumPose[1]:nyj, minimumPose[2]:maximumPose[2]:nzj ]
                grid_x = grid_x.flatten()
                grid_y = grid_y.flatten()
                grid_z = grid_z.flatten()
                gridVal = np.ndarray(np.shape(grid_x)[0])
                for i in range(np.shape(grid_x)[0]):
                    gridVal[i] = self.valuePoseActionTables([grid_x[i], grid_y[i], grid_z[i]], action)
                gridVal = np.where(gridVal==0, np.NaN, gridVal)

                ax.append(fig.add_subplot(plot["plotFormat"] + plotIndex, projection='3d'))
                gridValMin = min([gridValMin, np.nanmin(gridVal)])
                gridValMax = max([gridValMax, np.nanmax(gridVal)])
                # print( " gridValMin: ", gridValMin, "gridValMax: ", gridValMax)
                print("No: ", plotNumber, ' m: ', np.nanmin(gridVal), ' M: ', np.nanmax(gridVal))
                boxMarkerN.append(ax[plotIndex].scatter(grid_x, grid_y, grid_z, marker='o', c=gridVal, s=30, norm=norm))
                if not(boundingBox == None):
                    ax[plotIndex].set_xlim(boundingBox[0][0], boundingBox[0][1])
                    ax[plotIndex].set_ylim(boundingBox[1][0], boundingBox[1][1])
                    ax[plotIndex].set_zlim(boundingBox[2][0], boundingBox[2][1])

                ax[plotIndex].set_xlabel('X')
                ax[plotIndex].set_ylabel('Y')
                ax[plotIndex].set_zlabel('Z')
                plt.title(action)
                # Customize the view angle
                ax[plotIndex].view_init(elev=-5., azim=-30) #azim um z, links händisch
            fig.colorbar(boxMarkerN[0], shrink=0.5, aspect=5)
            plt.show()


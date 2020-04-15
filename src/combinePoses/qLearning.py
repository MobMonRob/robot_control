import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import tiling

#DataStructure :=
#[{
# 'nameIndex': 'B41', 
# 'poseIndex': 82, 
# 'pose': array([0.2 , 0.15, 0.  , 0.  , 0.  , 0.  ]), 
# 'score': {'getBox': 0.6267562724014337, 'depositBox': 0.07351851851851854}, 
# 'jointAvg': {
#     'getBox': {'timestamp': 234, 'LELBW': 68.83518518518518, 'RELBW': 68.46290322580644, 'LSHOULDER2D': 28.24814814814815, 'RSHOULDER2D': 21.587096774193547}, 
#     'depositBox': {'timestamp': 234, 'LELBW': 128.61944444444444, 'RELBW': 112.79444444444444, 'LSHOULDER2D': 57.97222222222222, 'RSHOULDER2D': 49.49444444444444}
#  }, 
# 'handCenterAvg': {
#     'getBox': {'timestamp': 234, 'x': 35.69179838709677, 'y': -124.85435483870968, 'z': 1270.304758064516}, 
#     'depositBox': {'timestamp': 234, 'x': -473.5820694444444, 'y': 295.9391624999999, 'z': 969.8308750000002}
#  },
#  'path': [
#       {'start': 'getBox', 'end': 'depositBox', 'duration': 181.0, 'path': array([-509.27386783,  420.79351734, -300.47388306]), 'distance': 725.7490000642434}
#   ],
# }]

#State: Ellenbogen- & Schultergelenke: 4D, diskret samplen
#Action: 7 Actions definieren, auswahl von N parametern über ~> jeweils reward ~> aufsummieren
# 7 reward Werte durch Interpolation Richtung ermitteln


class Actions:
    def __init__(self):
        self.actions = {
            "halt": 0,
            "back": 1,
            "front": 2,
            "left": 3,
            "right": 4,
            "down": 5,
            "up": 6,

            "bottomBack": 7,
            "bottomFront": 8,
            "bottomLeft": 9,
            "bottomRight": 10,
            "backLeft": 11,
            "backRight": 12,
            "frontLeft": 13,
            "frontRight": 14,
            "topBack": 15,
            "topFront": 16,
            "topLeft": 17,
            "topRight": 18,

            "backLeftBottom": 19,
            "backRightBottom": 20,
            "frontLeftBottom": 21,
            "frontRightBottom": 22,
            "backLeftTop": 23,
            "backRightTop": 24,
            "frontLeftTop": 25,
            "frontRightTop": 26
        }
        self.length = len(self.actions)
        radiusSurface = 0.1
        radiusEdge = 0.1
        radiusCorner = 0.1
        xS = 0.1
        xE = 0.07
        xC = 0.055
        
        #TODO: aktuell zeige ich über die Gewichtung (Min2) an, ob ein Aktion sinnvoll ist.
        # innerhalb von Gebieten, die jedoch komplett schlecht sind, ist dies nur unzureichend möglich, da es aktuell nicht darstellbar ist (konstantes 0 Nivau auch im Bereich stark außerhalb)
        # im Folgenden muss eine Entscheidung sowohl durch absolute Werte möglich sein (zwei verschiedene Stichproben vergleichen),
        # wie auch relativ, dies heißt in der Umgebung eines Punktes eine sinnvolle Aktion feststellen - auch wenn der Punkt sehr weit weg von einer möglich Lösung ist.
        # Als mögliche Lösung wird erachtet, das Gütemaß anzupassen. um die bisherige Strukture nicht zu verwässern, kann das Gütemaß einfach in den negativen Bereich erweitert werden (das Plateau bei 1 bleibt erhalten)
        # Welche Vorteile haben kontinuierlieche Abstiegsgradienten und welche haben ein Abstieg mit einem Plateau bei 0?

        self.directions = {
            # principal * vector[principal] is the direction to move
            # vector[auxiliary] are the other both directions
            "halt": {"sign": [1], "principal": [0], "auxiliary": [1, 2], "target": np.array([0.0, 0.0, 0.0]), "radius": radiusSurface, "position": "halt"},
            "back": { "sign": [-1], "principal": [0], "auxiliary": [1, 2], "target": np.array([-xS, 0.0, 0.0]), "radius": radiusSurface, "position": "center"},
            "front": { "sign": [1], "principal": [0], "auxiliary": [1, 2], "target": np.array([xS, 0.0, 0.0]), "radius": radiusSurface, "position": "center"},
            "left": { "sign": [-1], "principal": [1], "auxiliary": [0, 2], "target": np.array([0.0, -xS, 0.0]), "radius": radiusSurface, "position": "center"},
            "right": { "sign": [1], "principal": [1], "auxiliary": [0, 2], "target": np.array([0.0, xS, 0.0]), "radius": radiusSurface, "position": "center"},
            "down": { "sign": [-1], "principal": [2], "auxiliary": [0, 1], "target": np.array([0.0, 0.0, -xS]), "radius": radiusSurface, "position": "center"},
            "up": { "sign": [1], "principal": [2], "auxiliary": [0, 1], "target": np.array([0.0, 0.0, xS]), "radius": radiusSurface, "position": "center"},

            "bottomBack": { "sign": [-1, -1], "principal": [2, 0], "auxiliary": [1], "target": np.array([-xE, 0.0, -xE]), "radius": radiusEdge, "position": "center"},
            "bottomFront": { "sign": [-1, 1], "principal": [2, 0], "auxiliary": [1], "target": np.array([xE, 0.0, -xE]), "radius": radiusEdge, "position": "center"},
            "bottomLeft": { "sign": [-1, -1], "principal": [2, 1], "auxiliary": [0], "target": np.array([0.0, -xE, -xE]), "radius": radiusEdge, "position": "center"},
            "bottomRight": { "sign": [-1, 1], "principal": [2, 1], "auxiliary": [0], "target": np.array([0.0, xE, -xE]), "radius": radiusEdge, "position": "center"},
            "backLeft": { "sign": [-1, -1], "principal": [0, 1], "auxiliary": [2], "target": np.array([-xE, -xE, 0.0]), "radius": radiusEdge, "position": "center"},
            "backRight": { "sign": [-1, 1], "principal": [0, 1], "auxiliary": [2], "target": np.array([-xE, xE, 0.0]), "radius": radiusEdge, "position": "center"},
            "frontLeft": { "sign": [1, -1], "principal": [0, 1], "auxiliary": [2], "target": np.array([xE, -xE, 0.0]), "radius": radiusEdge, "position": "center"},
            "frontRight": { "sign": [1, 1], "principal": [0, 1], "auxiliary": [2], "target": np.array([xE, xE, 0.0]), "radius": radiusEdge, "position": "center"},
            "topBack": { "sign": [1, -1], "principal": [2, 0], "auxiliary": [1], "target": np.array([-xE, 0.0, xE]), "radius": radiusEdge, "position": "center"},
            "topFront": { "sign": [1, 1], "principal": [2, 0], "auxiliary": [1], "target": np.array([xE, 0.0, xE]), "radius": radiusEdge, "position": "center"},
            "topLeft": { "sign": [1, -1], "principal": [2, 1], "auxiliary": [0], "target": np.array([0.0, -xE, xE]), "radius": radiusEdge, "position": "center"},
            "topRight": { "sign": [1, 1], "principal": [2, 1], "auxiliary": [0], "target": np.array([0.0, xE, xE]), "radius": radiusEdge, "position": "center"},

            "backLeftBottom": { "sign": [-1, -1, -1], "principal": [0, 1, 2], "auxiliary": [], "target": np.array([-xC, -xC, -xC]), "radius": radiusCorner, "position": "center"},
            "backRightBottom": { "sign": [-1, 1, -1], "principal": [0, 1, 2], "auxiliary": [], "target": np.array([-xC, xC, -xC]), "radius": radiusCorner, "position": "center"},
            "frontLeftBottom": { "sign": [1, -1, -1], "principal": [0, 1, 2], "auxiliary": [], "target": np.array([xC, -xC, -xC]), "radius": radiusCorner, "position": "center"},
            "frontRightBottom": { "sign": [1, 1, -1], "principal": [0, 1, 2], "auxiliary": [], "target": np.array([xC, xC, -xC]), "radius": radiusCorner, "position": "center"},
            "backLeftTop": { "sign": [-1, -1, 1], "principal": [0, 1, 2], "auxiliary": [], "target": np.array([-xC, -xC, xC]), "radius": radiusCorner, "position": "center"},
            "backRightTop": { "sign": [-1, 1, 1], "principal": [0, 1, 2], "auxiliary": [], "target": np.array([-xC, xC, xC]), "radius": radiusCorner, "position": "center"},
            "frontLeftTop": { "sign": [1, -1, 1], "principal": [0, 1, 2], "auxiliary": [], "target": np.array([xC, -xC, xC]), "radius": radiusCorner, "position": "center"},
            "frontRightTop": { "sign": [1, 1, 1], "principal": [0, 1, 2], "auxiliary": [], "target": np.array([xC, xC, xC]), "radius": radiusCorner, "position": "center"},

        }

    def index(self, action):
        return self.actions[action]
    
    def keys(self):
        return self.actions.keys()
    
    def displacementVector(self, action):
        return self.directions[action]["target"]
    

class QValueFunction:
    def __init__(self, tilingsShape, actions:Actions, lr, dataStructure, dimensionsReduction=False, outOfMiddleDistance=False):
        self.tilings = tiling.create_tilings(tilingsShape, dimensionsReduction=dimensionsReduction, outOfMiddleDistance=outOfMiddleDistance)
        self.num_tilings = len(self.tilings)
        self.actions = actions
        self.lr = lr  # /self.num_tilings  # learning rate equally assigned to each tiling
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in
                            self.tilings]  # [(10, 10), (10, 10), (10, 10)]
        self.q_tables = [np.zeros(shape=(state_size + (self.actions.length,))) for state_size in self.state_sizes] #n_tiling: 1 bis 2 *  tiling: [7, 7, 5, 5] * actions: 7

        self.q_tablesList = [[[] for _ in range(state_size[0] * state_size[1] * state_size[2] * self.actions.length)] for state_size in self.state_sizes]
        self.stateScene = "getBox"
        self.dataStructure = dataStructure
        self.dimensionsReduction = dimensionsReduction
        self.outOfMiddleDistance = outOfMiddleDistance
        self.statePoseList = self.getPoseList() #x,y,z #annotations!!!
        # self.statePoseList = self.getPoseListHandCenter() #-x,-y,-z
        print(self.statePoseList)

    def getState(self, dataStructureIndex, stateScene, dataStructure=None):
        if dataStructure == None:
            data = self.dataStructure[dataStructureIndex]["jointAvg"][stateScene]
        else:
            data = dataStructure[dataStructureIndex]["jointAvg"][stateScene]
        if self.dimensionsReduction:
            if self.outOfMiddleDistance:
                #state = [np.nanmean([data["LELBW"]]), np.nanmean([data["LSHOULDER2D"]]), data["OUTOFMIDDLEDISTANCE"]]
                state = [np.nanmean([data["LELBW"], data["RELBW"]]), np.nanmean([data["LSHOULDER2D"], data["RSHOULDER2D"]]), data["OUTOFMIDDLEDISTANCE"]] #TODO
            else:
                state = [np.nanmean([data["LELBW"], data["RELBW"]]), np.nanmean([data["LSHOULDER2D"], data["RSHOULDER2D"]])]
        else:
            if self.outOfMiddleDistance:
                state = [data["LELBW"], data["RELBW"], data["LSHOULDER2D"], data["RSHOULDER2D"], data["OUTOFMIDDLEDISTANCE"]]
            else:
                state = [data["LELBW"], data["RELBW"], data["LSHOULDER2D"], data["RSHOULDER2D"]]
        return np.array(state)

    def getPoseList(self):
        statePoseList = []
        for i in range(len(self.dataStructure)):
            statePoseList.append(self.dataStructure[i]["pose"][:3])
        return np.array(statePoseList)

    def getPoseListHandCenter(self):
        statePoseList = []
        for i in range(len(self.dataStructure)):
            data = self.dataStructure[i]["handCenterAvg"][self.stateScene]
            statePoseList.append([data["x"]/1000, data["y"]/1000, data["z"]/1000])
        return np.array(statePoseList)

    #search the corresponding runs to a action depending on one fixed pose (dataStructureIndex)
    def getStateListDependingOnAction(self, dataStructureIndex, action, count):
        currentPose = self.statePoseList[dataStructureIndex]
        distListOfAllPoses = []
        distancePose = self.statePoseList - currentPose
        if action == "halt":
            for i in range(len(distancePose)):
                distListOfAllPoses.append(np.sqrt((distancePose[i, 0])**2 + distancePose[i, 1]**2 + distancePose[i, 2]**2))
        else:
            principalDirectionSign = self.actions.directions[action]["sign"]
            principalDirectionIndex = self.actions.directions[action]["principal"]
            auxiliaryDirectionIndex = self.actions.directions[action]["auxiliary"]
            target = self.actions.directions[action]["target"]
            if len(principalDirectionIndex) == 1:
                for i in range(len(distancePose)):
                    if i == dataStructureIndex: # ignore itself
                        distListOfAllPoses.append(1000001)
                        continue
                    # 1/6 of the cube:
                    if distancePose[i, principalDirectionIndex[0]] * principalDirectionSign[0] > 0 \
                        and np.abs(distancePose[i, principalDirectionIndex[0]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[0]]) \
                        and np.abs(distancePose[i, principalDirectionIndex[0]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[1]]): #is the pose in the relevant area for the action
                        distListOfAllPoses.append(np.linalg.norm(distancePose[i]-target))
                    else:
                        distListOfAllPoses.append(1000000)
            elif len(principalDirectionIndex) == 2:
                for i in range(len(distancePose)):
                    if i == dataStructureIndex: # ignore itself
                        distListOfAllPoses.append(1000001)
                        continue
                    # 1/12 of the cube, combination of the the other two
                    if distancePose[i, principalDirectionIndex[0]] * principalDirectionSign[0] > 0 \
                        and distancePose[i, principalDirectionIndex[1]] * principalDirectionSign[1] > 0 \
                        and np.abs(distancePose[i, principalDirectionIndex[0]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[0]]) \
                        and np.abs(distancePose[i, principalDirectionIndex[1]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[0]]):   
                        distListOfAllPoses.append(np.linalg.norm(distancePose[i]-target))
                    else:
                        distListOfAllPoses.append(1000000)
            elif len(principalDirectionIndex) == 3:
                for i in range(len(distancePose)):
                    if i == dataStructureIndex: # ignore itself
                        distListOfAllPoses.append(1000001)
                        continue
                    # 1/8 of the cube:
                    if distancePose[i, principalDirectionIndex[0]] * principalDirectionSign[0] > 0 \
                        and distancePose[i, principalDirectionIndex[1]] * principalDirectionSign[1] > 0 \
                        and distancePose[i, principalDirectionIndex[2]] * principalDirectionSign[2] > 0:
                        distListOfAllPoses.append(np.linalg.norm(distancePose[i]-target))
                    else:
                        distListOfAllPoses.append(1000000)

        maxRadius = self.actions.directions[action]["radius"]
        smallestDistPoseIndex = np.argsort(np.array(distListOfAllPoses))[:count] #argsort: nur noch die Index der Werte
        for index, poseIndex in enumerate(smallestDistPoseIndex):
            if distListOfAllPoses[poseIndex] > maxRadius: #falls Distanz größter als zulässiger Radius
                if index == 0:
                    return None
                smallestDistPoseIndex = smallestDistPoseIndex[:index]
                break
        #TODO gewichtung, je nach Position
        return smallestDistPoseIndex

    #search the corresponding runs to a action depending on one fixed pose (dataStructureIndex)
    def getStateListDependingOnAction2(self, dataStructureIndex, action, count):
        currentPose = self.statePoseList[dataStructureIndex]
        target = self.actions.directions[action]["target"]
        distancePose = self.statePoseList - currentPose #für alle Actions gültig!

        maxRadius = self.actions.directions[action]["radius"]
        validPairs = [] 
        if action == "halt":
            distListOfAllPoses = np.linalg.norm(distancePose, axis=1)
            possibleIndex = np.argwhere(distListOfAllPoses < maxRadius)   
            for i in possibleIndex:
                validPairs.append((i, distListOfAllPoses[i]))
        else:
            distListOfAllPoses = np.linalg.norm((distancePose-target), axis=1)
            possibleIndex = np.argwhere(distListOfAllPoses < maxRadius) #Radius um den Zielpunkt
            principalDirectionSign = self.actions.directions[action]["sign"]
            principalDirectionIndex = self.actions.directions[action]["principal"]
            auxiliaryDirectionIndex = self.actions.directions[action]["auxiliary"]
            if len(principalDirectionIndex) == 1:
                for i in possibleIndex:
                    if i == dataStructureIndex: # ignore itself
                        continue
                    # 1/6 of the cube:
                    if distancePose[i, principalDirectionIndex[0]] * principalDirectionSign[0] > 0 \
                        and np.abs(distancePose[i, principalDirectionIndex[0]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[0]]) \
                        and np.abs(distancePose[i, principalDirectionIndex[0]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[1]]): #is the pose in the relevant area for the action
                        validPairs.append((i, distListOfAllPoses[i]))
            elif len(principalDirectionIndex) == 2:
                for i in possibleIndex:
                    if i == dataStructureIndex: # ignore itself
                        continue
                    # 1/12 of the cube, combination of the the other two
                    if distancePose[i, principalDirectionIndex[0]] * principalDirectionSign[0] > 0 \
                        and distancePose[i, principalDirectionIndex[1]] * principalDirectionSign[1] > 0 \
                        and np.abs(distancePose[i, principalDirectionIndex[0]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[0]]) \
                        and np.abs(distancePose[i, principalDirectionIndex[1]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[0]]):
                        validPairs.append((i, distListOfAllPoses[i]))
            elif len(principalDirectionIndex) == 3:
                for i in possibleIndex:
                    if i == dataStructureIndex: # ignore itself
                        continue
                    # 1/8 of the cube:
                    if distancePose[i, principalDirectionIndex[0]] * principalDirectionSign[0] > 0 \
                        and distancePose[i, principalDirectionIndex[1]] * principalDirectionSign[1] > 0 \
                        and distancePose[i, principalDirectionIndex[2]] * principalDirectionSign[2] > 0:
                        validPairs.append((i, distListOfAllPoses[i]))

        if len(validPairs) == 0:
            return None
        validPairs.sort(key=lambda tup: tup[1])
        validPairs = validPairs[:count] #take only the values with the smallest distance to the targetposition
        validList = []
        for el in validPairs:
            validList.append(int(el[0]))
        #TODO gewichtung, je nach Position
        return validList

        #search the corresponding runs to a action depending on one fixed pose (dataStructureIndex)
    def getStateList3(self, dataStructureIndex, count):
        currentPose = self.statePoseList[dataStructureIndex] #TODO: zufällige Werte nehmen - nicht alle!
        distancePose = self.statePoseList - currentPose #für alle Actions gültig!
        validPairs = []
        maxList = {}
        meanList = {}
        medianList = {}
        maxSamples = 100 # begrenzung auf 100? muss nur den median gut abbilden
        for action in self.actions.keys():
            distListOfAllPoses = np.linalg.norm((distancePose-self.actions.directions[action]["target"]), axis=1)
            possibleIndex = np.argwhere(distListOfAllPoses < self.actions.directions[action]["radius"])
            possibleIndex = possibleIndex[random.sample(range(len(possibleIndex)), maxSamples if maxSamples < len(possibleIndex) else len(possibleIndex) )]

            if action == "halt":
                validPairs = [(i, distListOfAllPoses[i]) for i in possibleIndex]
            else:
                principalDirectionSign = self.actions.directions[action]["sign"]
                principalDirectionIndex = self.actions.directions[action]["principal"]
                auxiliaryDirectionIndex = self.actions.directions[action]["auxiliary"]
                if len(principalDirectionIndex) == 1:
                    validPairs = [(i, distListOfAllPoses[i]) for i in possibleIndex if distancePose[i, principalDirectionIndex[0]] * principalDirectionSign[0] > 0 \
                        and np.abs(distancePose[i, principalDirectionIndex[0]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[0]]) \
                        and np.abs(distancePose[i, principalDirectionIndex[0]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[1]])]
                elif len(principalDirectionIndex) == 2:
                    validPairs = [(i, distListOfAllPoses[i]) for i in possibleIndex if distancePose[i, principalDirectionIndex[0]] * principalDirectionSign[0] > 0 \
                        and distancePose[i, principalDirectionIndex[1]] * principalDirectionSign[1] > 0 \
                        and np.abs(distancePose[i, principalDirectionIndex[0]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[0]]) \
                        and np.abs(distancePose[i, principalDirectionIndex[1]]) >= np.abs(distancePose[i, auxiliaryDirectionIndex[0]])]
                elif len(principalDirectionIndex) == 3:
                    validPairs = [(i, distListOfAllPoses[i]) for i in possibleIndex if distancePose[i, principalDirectionIndex[0]] * principalDirectionSign[0] > 0 \
                            and distancePose[i, principalDirectionIndex[1]] * principalDirectionSign[1] > 0 \
                            and distancePose[i, principalDirectionIndex[2]] * principalDirectionSign[2] > 0]
        
            if len(validPairs) == 0:
                maxList[action] = None
                meanList[action] = None
                medianList[action] = None
                continue
            validPairs.sort(key=lambda tup: tup[1])
            validPairs = validPairs[:count]
            if action is not "halt":
                pass
                #print("Remove: : ", dataStructureIndex)
            reward = [self.dataStructure[int(el[0])]['score'][self.stateScene] for el in validPairs]
            maxList[action] = np.nanmax(reward)
            meanList[action] = np.nanmean(reward)
            medianList[action] = np.nanmedian(reward)
        #for action in self.actions.keys():
         #   print("Max: ", maxList[action], " Mean: ", meanList[action], " Median: ", medianList[action]) #median ist regelmäßig extremer als mean; das der arbeiter sich selbst eine optimale Haltung raussucht geht zu sehr unter?!
        #TODO gewichtung, je nach Position
        return medianList #maxList #meanList


    def value(self, state, action): #value der Q-Table
        state_codings = tiling.get_tile_coding(state, self.tilings)  # [[5, 1], [4, 0], [3, 0]] ...
        action_idx = self.actions.index(action)

        value = 0
        for coding, q_table in zip(state_codings, self.q_tables):
            value += q_table[tuple(coding) + (action_idx,)]
        return value / self.num_tilings
    
    def valueVectorPairs(self, state):
        # bestimmt die values für alle actions,
        # trägt diese als dict in eine liste
            #TODO Glättung der q_Table / der Value map  ~> value abtasten wie zur 3D Darstellung, danach glätten (7 unabhängige 3D Räume werden geglättet ODER immer 3 Räume koppeln. z.B.: left-halt-right)
        valueVectorPairs = [{ 
                "displacementVector": self.actions.displacementVector(action),
                "value": self.value(state, action),
                "actionIdx": self.actions.index(action)
            } for action in self.actions.keys()]
        return valueVectorPairs

    def update(self, state, action, target):
        state_codings = tiling.get_tile_coding(state, self.tilings)  # [[5, 1], [4, 0], [3, 0]] ...
        action_idx = self.actions.index(action)

        for coding, q_table in zip(state_codings, self.q_tables):
            delta = target - q_table[tuple(coding) + (action_idx,)]
            q_table[tuple(coding) + (action_idx,)] += self.lr * (delta)

    # updates all actions with the information of one state
    def updateAllActions(self, dataStructureIndex):
        state = self.getState(dataStructureIndex, self.stateScene)
        state_codings = tiling.get_tile_coding(state, self.tilings)
        self.dataStructure[dataStructureIndex]["action"] = {}
        for action in self.actions.keys():
            action_idx = self.actions.index(action)
            indexList = self.getStateListDependingOnAction(dataStructureIndex, action, 5) # sucht alle zu der Pose und Action passenden Actions #TODO wieviele Datensätze werden einbezogen (aktuell 5)
            self.dataStructure[dataStructureIndex]["action"][action] = None
            if indexList is not None:
                reward = 0
                for index in indexList:
                    reward += self.dataStructure[index]['score'][self.stateScene]
                reward /= len(indexList)  # TODO Gewichtung der verschiedenen Werte
                for coding, q_table in zip(state_codings, self.q_tables):
                    if not(q_table[tuple(coding) + (action_idx,)] == 0):
                        lr = 1
                    else:
                        lr = self.lr
                    delta = reward - q_table[tuple(coding) + (action_idx,)]
                    q_table[tuple(coding) + (action_idx,)] += lr * (delta) #TODO set learning rate
                    self.dataStructure[dataStructureIndex]["action"][action] = q_table[tuple(coding) + (action_idx,)] * 20 + 60 #nur zum ploten
                    # print("action: ", action, " value: ", q_table[tuple(coding) + (action_idx,)])

        # updates all actions with the information of one state
    def updateAllActionsMax(self, dataStructureIndex):
        state = self.getState(dataStructureIndex, self.stateScene)
        state_codings = tiling.get_tile_coding(state, self.tilings)
        self.dataStructure[dataStructureIndex]["action"] = {}
        rewardListAction = self.getStateList3(dataStructureIndex, len(self.dataStructure)) #TODO wieviele Datensätze werden einbezogen (aktuell ALLE)
        for action in self.actions.keys():
            action_idx = self.actions.index(action)
            reward = rewardListAction[action]
            self.dataStructure[dataStructureIndex]["action"][action] = None
            if reward is not None:
                for coding, q_table in zip(state_codings, self.q_tables): #MAXIMUM
                    if q_table[tuple(coding) + (action_idx,)] < reward:
                        q_table[tuple(coding) + (action_idx,)] = reward
                        #self.dataStructure[dataStructureIndex]["action"][action] = q_table[tuple(coding) + (action_idx,)] * 20 + 60 #nur zum ploten
                for coding, q_tableList in zip(state_codings, self.q_tablesList): #MEAN & in trainWithAllData
                    q_tableList[np.ravel_multi_index(tuple(coding) + (action_idx,), (self.state_sizes[0] + (self.actions.length,)))].append(reward)

    def trainWithAllData(self, generatedData):
        if generatedData:
            for i in random.choices(range(len(self.dataStructure)), k=50000):
                if i % 1 == 0:
                    print("Train : ", i)
                self.updateAllActionsMax(i)
        else:
            for i in range(len(self.dataStructure)):
                self.updateAllActions(i)
        #for i, el in enumerate(self.q_tablesList[0]): #(3, 5, 3, 3, 27)
        #    print(el)
        #    if i == 900:
        #        break
        for q_table, q_tableList in zip(self.q_tables, self.q_tablesList):
            for i, el in enumerate(q_tableList):
                #print(q_table[np.unravel_index(i, (self.state_sizes[0] + (self.actions.length,)))], " - ", np.nanmean(el))
                q_table[np.unravel_index(i, (self.state_sizes[0] + (self.actions.length,)))] = np.nanmean(el)
        

    def printQTable3D(self):
        plt.close('all')
        fig = plt.figure()
        ax = []
        boxMarkerN = []
        #norm = mpl.colors.Normalize(vmin=0, vmax=0.7)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        gridValMin = 1
        gridValMax = 0
        for plotNumber, action in enumerate(self.actions.keys()):
            if plotNumber > 7: # ToDo shows only the main directions
                continue
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
            if plotNumber == 7: #Maxplot
                for i in range(np.shape(grid_x)[0]):
                    actionValues = np.zeros(7)
                    for j, action2 in enumerate(self.actions.keys()):
                        actionValues[j] = self.value([grid_x[i], grid_y[i], grid_z[i]], action2)
                        if j == 6:
                            break
                    gridVal[i] = np.nanmax(actionValues)
            else:
                gridVal = [self.value([grid_x[i], grid_y[i], grid_z[i]], action) for i in range(np.shape(grid_x)[0])]
            
            gridVal = np.where(gridVal==0, np.NaN, gridVal)

            ax.append(fig.add_subplot(241 + plotNumber, projection='3d'))
            gridValMin = min([gridValMin, np.nanmin(gridVal)])
            gridValMax = max([gridValMax, np.nanmax(gridVal)])
            # print( " gridValMin: ", gridValMin, "gridValMax: ", gridValMax)
            print("No: ", plotNumber, ' m: ', np.nanmin(gridVal), ' M: ', np.nanmax(gridVal))
            boxMarkerN.append(ax[plotNumber].scatter(grid_x, grid_y, grid_z, marker='o', c=gridVal, s=30, norm=norm))

            #ax.set_zlim(0, 1)
            ax[plotNumber].set_xlabel('X-Elbow')
            ax[plotNumber].set_ylabel('Y-Shoulder')
            ax[plotNumber].set_zlabel('Z-Center')
            if plotNumber == 7:
                plt.title("Max-Plot")
            else:
                plt.title(action)
            # Customize the view angle
            ax[plotNumber].view_init(elev=-5., azim=-30) #azim um z, links händisch
        fig.colorbar(boxMarkerN[0], shrink=0.5, aspect=5)
        plt.show(block=True)

    def printStates3D(self, dataStructure):
        #plt.close('all')
        fig = plt.figure()
        ax = []
        boxMarkerN = []
        #norm = mpl.colors.Normalize(vmin=0, vmax=0.7)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        gridValMin = 1
        gridValMax = 0
        for plotNumber in range(1):
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

            grid_x = [] 
            grid_y = []
            grid_z = []
            gridVal = []
# 'score': {'getBox': 0.6267562724014337, 'depositBox': 0.07351851851851854}, 
# 'jointAvg': {
#     'getBox': {'timestamp': 234, 'LELBW': 68.83518518518518, 'RELBW': 68.46290322580644, 'LSHOULDER2D': 28.24814814814815, 'RSHOULDER2D': 21.587096774193547}, 
            for data in dataStructure:
                if data["pose"][2] < 0.2:
                    state = self.getState(0, self.stateScene, [data])
                    grid_x.append(state[0])
                    grid_y.append(state[1])
                    grid_z.append(state[2])
                    gridVal.append(data["score"][self.stateScene])

            ax.append(fig.add_subplot(111 + plotNumber, projection='3d'))
            gridValMin = min([gridValMin, np.nanmin(gridVal)])
            gridValMax = max([gridValMax, np.nanmax(gridVal)])
            # print( " gridValMin: ", gridValMin, "gridValMax: ", gridValMax)
            print("No: ", plotNumber, ' m: ', np.nanmin(gridVal), ' M: ', np.nanmax(gridVal))
            boxMarkerN.append(ax[plotNumber].scatter(grid_x, grid_y, grid_z, marker='o', c=gridVal, s=30, norm=norm))

            #ax.set_zlim(0, 1)
            ax[plotNumber].set_xlabel('Ellenbogengelenkwinkel in Grad')
            ax[plotNumber].set_ylabel('Schultergelenkwinkel in Grad')
            ax[plotNumber].set_zlabel('Abweichung aus der Mittelposition in mm')
            plt.title("Gelenkwinkel des Arbeiters für eine äquidistante Abtastung des Arbeitsraum")
            # Customize the view angle
            ax[plotNumber].view_init(elev=-5., azim=-30) #azim um z, links händisch
        fig.colorbar(boxMarkerN[0], shrink=0.5, aspect=5)
        plt.show()



class QLearning:
    def __init__(self, dataStructure):
        self.data = dataStructure

    def _qLearningWithTable(self, num_episodes=2000):
        self.qTable = np.zeros((len(self.data), 2))  # states x actions
        # states: Ellenbogen- & Schultergelenke 4D
        # actions:
        self.stateDict = []
        for i in range(len(self.data)):
            val = self.data[i]["jointAvg"]["getBox"]
            self.stateDict.append(np.array([val["LELBW"], val["RELBW"], val["LSHOULDER2D"], val["RSHOULDER2D"]]))
       
        y = 0.95
        lr = 0.8
        for i in range(num_episodes):
            s = random.randint(0, len(self.data)-1)
            done = False
            while not done:
                if np.sum(self.qTable[s,:]) == 0:
                    # make a random selection of actions
                    a = np.random.randint(0, 2)
                else:
                    # select the action with largest q value in state s
                    a = np.argmax(self.qTable[s, :])
                new_s, r, done, _ = env.step(a)
                self.qTable[s, a] += r + lr*(y*np.max(self.qTable[new_s, :]) - self.qTable[s, a])
                s = new_s
        return True

    def start(self):
        self._qLearningWithTable()
        return 0



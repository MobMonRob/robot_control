import numpy as np
import tiling
import matplotlib.pyplot as plt
import matplotlib as mpl
import qLearning
import gridInterpolating
import ergonomicMeasure
import UR_communication as xml_rpc_communication
import random
from datetime import datetime
import os

#https://stackoverflow.com/questions/14992521/python-weighted-random
class WeightedRandomizer:
    def __init__(self, weights):
        self.__max = .0
        self.__weights = []
        for (value, weight) in weights:
            self.__max += weight
            self.__weights.append ( (self.__max, value) )

    def random (self):
        r = random.random() * self.__max
        for ceil, value in self.__weights:
            if ceil > r: return value

def testWeightedRandomizer():
    w = {'A': 1.0, 'B': 1.0, 'C': 18.0}
    wr = WeightedRandomizer (w)

    results = {'A': 0, 'B': 0, 'C': 0}
    for i in range (10000):
        results [wr.random () ] += 1
    print (results)


class NextPose:
    def __init__(self, tilingsShapePure, qValue:qLearning.QValueFunction, gridInterpol:gridInterpolating.GridInterpolating, ergonomicMeasureObj:ergonomicMeasure.ErgonomicMeasure):
        self.tilingsShape = self.calculateTiling(tilingsShapePure)
        self.qValue = qValue
        self.gridInterpol = gridInterpol
        self.ergonomicMeasureObj = ergonomicMeasureObj
        self.tilings = tiling.create_tilings(self.tilingsShape)
        self.num_tilings = len(self.tilings)
        self.pose_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in
                            self.tilings]
        self.next_pose_tables = self.resetNextPoseTable()
        self.datastructure = []
        self.samplePoints = ergonomicMeasureObj.samplePoints #0: big Box, 1: small Box
        self.maxSampleBoxsize = [0 for el in self.samplePoints]
        self.boxType = 0
        self.radius = 0.25#TODO test this parameter
        self.estimateSampleBoxSize(self.radius)
        now = datetime.now()
        self.dateTime = now.strftime("%Y%m%d_%H_%M_%S")


    def calculateTiling(self, tilingsShapePure):
        tilingsShapePure["featureRange"] = []
        tilingsShapePure["bins"] = []
        factor = 1
        for box, sampling in zip(tilingsShapePure["boundingBox"], tilingsShapePure["sampling"]):
            tilingsShapePure["featureRange"].append([(box[0]+sampling/2), (box[1]-sampling/2)])
            tilingsShapePure["bins"].append(int(round((box[1]-box[0])*factor/sampling + 1)))
        return tilingsShapePure

    def resetNextPoseTable(self):
        return [np.zeros(shape=(pose_size)) for pose_size in self.pose_sizes]

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

    def getPseudoRandomNumber(self, randomSeed):
        if not(randomSeed is None):
            random.seed(randomSeed)
            print("random Seed is: ", randomSeed)
        else:
            #random.seed(30) # fixed order of the random numbers 
            pass
        # auswerten welche Posen schon probiert, in diesem Bereich keine Zufallsvariable mehr
        # es wird die Liste der gültigen Posen verwendet
        #Kugelmodel: durch die bereits vorhandenen Werte werden kugeln mit gleichem r aufgespannt, r reduziert bis Lücke innerhalb des Bereichs
        #r mit Value kombinieren, bei schlechtem Value größere Kugel #TODO: sinnvolle Wahl fpr value
        samplePoints = self.samplePoints[self.boxType]
        measuredPoints = []
        if len(self.datastructure) == 0:
            # startpunkt in einer Ecke (information der Mitte wird leicht über die inneren Informationen generiert)
            # daher hier Wahl des Mittelpunktes
            bb = np.array(self.tilingsShape["boundingBox"])
            measuredPoints.append({"pose": np.array([(bb[0][0] + bb[0][1])/2, (bb[1][0] + bb[1][1])/2, (bb[2][0] + bb[2][1])/2]), "value": 1})
        else:
            for measuredPoint in self.datastructure:
                measuredPoints.append({"pose": self.pose6Dto3D(measuredPoint["pose"]), "value": 1})
        #print("measuredPoints: ", measuredPoints)
        for r in np.linspace(0, 2, num=40):
            editedSamplePoints = samplePoints
            for measuredPoint in measuredPoints:
                newSamplePoints = [samplePoint for samplePoint in editedSamplePoints if np.linalg.norm(samplePoint["pose"]-measuredPoint["pose"]) > r/measuredPoint["value"]] #nur die Punkte notieren die außerhalb der Kugel liegen
                editedSamplePoints= newSamplePoints

            if len(editedSamplePoints) < 5: #beliebige Zahl > 0
                if len(editedSamplePoints) == 0:
                    editedSamplePoints =  samplePoints
                newPose = editedSamplePoints[random.randint(0, len(editedSamplePoints)-1)]
                print("len: ", len(editedSamplePoints), newPose)
                return newPose
            samplePoints = editedSamplePoints #iteration abschließen
        print("run failed....")
        return False

    def estimateSampleBoxSize(self, radius):
        # dichte der samplings im umfeld (feste boxen vs dynamische boxen(probleme an den ränder))
        #   Problem an den rändern, daher: samples in dyn box / samplePoints in dyn box ~> [0, 1]
        for samplePoint in self.samplePoints[self.boxType]:
            samplesInRange = sum([1 for neighborPoint in self.samplePoints[self.boxType] if np.linalg.norm(neighborPoint["pose"]-samplePoint["pose"]) <= radius])
            samplePoint["boxSize"] = samplesInRange
            if samplesInRange > self.maxSampleBoxsize[self.boxType]:
                self.maxSampleBoxsize[self.boxType] = samplesInRange
        for samplePoint in self.samplePoints[self.boxType]:
            samplePoint["counterCorrection"] = 1.0 * self.maxSampleBoxsize[self.boxType] / samplePoint["boxSize"]

    def calcExploration(self, radius):
        measuredPoints = []
        for measuredPoint in self.datastructure:
            measuredPoints.append({"pose": self.pose6Dto3D(measuredPoint["pose"]), "value": 1})
        
        maxExploration = 0
        for samplePoint in self.samplePoints[self.boxType]:
            counter = sum([1 for measuredPoint in measuredPoints if np.linalg.norm(measuredPoint["pose"]-samplePoint["pose"]) <= radius])  #TODO-timeoptimazation, indizes für die entspchenden Punkte notieren
            #TODO: weight with distance
            samplePoint["exploration"] = samplePoint["counterCorrection"] * counter  #wieviel Prozent abdeckung
            #print("samplePoint[exploration]", samplePoint["exploration"], " counter; ", counter)
            if samplePoint["exploration"] > maxExploration:
                maxExploration = samplePoint["exploration"]

                
        return maxExploration

    def estimateNextPose(self, randomSeed=None):
        plot3D = True
        #abhängig von der Anzahl der bisherigen Posen Zufallszahl oder mit den bisherigen Informationen eine neue Pose bestimmen
        if len(self.datastructure) < 4: # da keine Interpolation möglich ist quasi-Zufallszahlen, möglichst weit verteilt über den Sampling-raum
            nextPose = self.pose3Dto6D(self.getPseudoRandomNumber(randomSeed)["pose"])
            plt.close('all')
            self.plot3D("Total", None, block=True, ifAnimation=False, name=len(self.datastructure))
            return nextPose
        else:                                       #11, 13, 16
            evolutionary = {"iteration": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 
                            "radius": [0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                            "factor": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]}
            for i, iteration in enumerate(evolutionary["iteration"]):
                if len(self.datastructure) == iteration:
                    self.radius = evolutionary["radius"][i]
                    self.estimateSampleBoxSize(self.radius)
                    self.factor = evolutionary["factor"][i]
                    break
            print("ITERATION: ", len(self.datastructure), " radius: ", self.radius)
            maxExploration = self.calcExploration(self.radius)
            randomizerList = []
            explorationList = []
            exploitationList = []
            ergonomicList = []
            distList = []
            densityList = []
            results = {}
            totalDensity = 0
            #radius muss sich verändern; anfangs groß, damit alle regionen angetestet werden, später kleiner, damit in guten gebieten mehr getestet wird 
            # (hierbei muss exploitation in guten gebieten stärker sein, als ein noch nicht gewählter punkt im null gebiet)
            self.gridInterpol.calcDistanceTable3D( not not plot3D)
            #self.ergonomicMeasureObj.printErgonomicMeasure3D(radius=self.radius) 
            for i, samplePoint in enumerate(self.samplePoints[self.boxType]):
                #exploration = (maxExploration - samplePoint["exploration"])/maxExploration #0: für den Punkt der in seinem Umfeld am meisten Proben hat. 1 Falls noch keine Probe im Umfeld
                exploration = (np.log(len(self.datastructure)) / samplePoint["exploration"] ) #sqrt wurde entfernt
                explorationMaxValue = 2
                exploration = (exploration if np.isfinite(exploration) else explorationMaxValue)/explorationMaxValue
                explorationList.append(exploration)
                #print("len: ", len(self.datastructure), " ", len(self.datastructure)/self.maxSampleBoxsize[self.boxType])
                #exploration = np.sqrt(np.log(len(self.datastructure)/self.maxSampleBoxsize[self.boxType])/samplePoint["exploration"]) #add +1 to the counter in calcExploration
                #print("inside: ", np.log(len(self.datastructure)/self.maxSampleBoxsize[self.boxType]), " exploration ", exploration)

                ergonomic = self.ergonomicMeasureObj.extrapolateData(self.boxType, i, self.radius) #[0, 1] # 1 is the best
                ergonomic = 0 if np.isnan(ergonomic) else ergonomic
                ergonomicList.append(ergonomic)
                dist = self.gridInterpol.value(samplePoint["pose"]) #dist: [0, 1] # 1 is the best
                dist = 0 if np.isnan(dist) else dist
                distList.append(dist)
                exploitation = (ergonomic + dist)/2
                #exploitation = ergonomic * dist #TODO
                exploitationList.append(exploitation)

                density = (1-self.factor) * exploitation + self.factor * exploration
                #density = np.power(30, density) #TODO
                #density = density * density #TODO
                densityList.append(density)
                totalDensity += density
                #print("Expl: : ", "NaN" if exploitation == 0 else self.ergonomicMeasureObj.extrapolateData(self.boxType, i)/exploitation*100)
                #print("density: ", exploitation/density*100)
                randomizerList.append((i, density))
    #print("dens: ", density, " explor: ", exploration, " exploi: ", exploitation, " dist: ", dist, " erg: ", ergonomic)
                results[i] = 0
                #print("exploitation: ", (1-factor) *4* exploitation, " exploration: ", factor * np.sqrt(2) * exploration, self.ergonomicMeasureObj.extrapolateData(self.boxType, i))
            print("len: ", len(randomizerList), " to ", int(np.around(len(randomizerList)*self.factor)))
            randomizerList.sort(key=lambda tup: tup[1], reverse=True)
            randomizerList = randomizerList[:int(np.around(len(randomizerList)*self.factor))]
            wr = WeightedRandomizer(randomizerList)
            #for i in range (100000):
            #    results[wr.random()] += 1
            #for el in results:
             #   print(el, ":  ", results[el]*100.0/100000, " - ", randomizerList[el] * 100.0 / totalDensity)
            
            #verwenden des Distanzgütemaßes ~> ein 3D Skalarfeld für alle Punkte!
                # es fehlt noch die Extrapolierung; werden nur die besten 20% der Punkte weitergegeben oder alle? Histogrammvorgabe?
            #verwenden des Ergonomiemaßes ~> ein 3D Skalarfeld für alle Punkte!
                # ich habe: zu jedem der bisherigen Punkte 27 Actions
                # ich habe zu jeder Action eine Karte
                # currentState(pose) ~> bei der Pose die Actions einzeichnen ~> 27 Punkte pro Messung
                    # Punkte vervielfältigen zu 125 Punkten mit zugehöriger Gewichtung (z.B: zuerst 2 und dann 1)
                # alle Punkte in eine Karte einzeichnen ~> Tiling mit durchschnittsbildung (Beachtung der Gewichtung)
                # es fehlt noch die Extrapolierung; werden nur die besten 20% der Punkte weitergegeben oder alle? Histogrammvorgabe?

            #two main perspectives: 
            #   -gewinnrate, von den bisherigen samplings; gewinnen in der nähe ist besser als weiter weg
            #       -klingt nach IDW or RBF ~> Gesamtkarte erstellen ~> [0, 1] ~> entsprechenden Punkt auslesen
            #           -IDW und lin Interpol einfach addieren; oder 2* IDW.... 
            # verknüpfung dieser beiden werte mit einem Faktor c ~>Dichtefunktion über alle Punkte
            #   für die Verknüpfung siehe auch: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
            #      exploitation = wi/ni
            #      exploration = sqrt(ln(Ni)/ni) = sqrt(ln(Ni/maxBoxsize)/(ni/niBoxSize))
            #      fi = (1-a) * exploitation + a * c * exploration with c=sqrt(2), a is dynamic variable
            if plot3D:
                print("ergonomic: ", np.histogram(ergonomicList, bins=10, range=(0,1)))
                print("dist: ", np.histogram(distList, bins=10, range=(0,1)))
                print("Exploitation: ", np.histogram(exploitationList, bins=10, range=(0,1)))
                print("exploration: ", np.histogram(explorationList, bins=10, range=(0,1)))
                print("Density: ", np.histogram(densityList, bins=10, range=(1, 30)))
                plt.close('all')
                #self.plot3D("Exploration", explorationList, block=True)
               # self.plot3D("Exploitation", exploitationList, block=False)
                self.plot3D("Total", densityList, block=True, ifAnimation=False, name=len(self.datastructure))
            nextPose = self.pose3Dto6D(self.samplePoints[self.boxType][wr.random()]["pose"])
            return nextPose

    #TODO: ergonomic schärfen


    def plot3D(self, title, gridValRaw, boundingBox=None, block=True, ifAnimation=False, name=None):
        fig = plt.figure()
        w_in_inches = 5
        h_in_inches = 5
        fig.set_size_inches(w_in_inches, h_in_inches, True)
        boxMarkerN = []
        ax = []
        plotIndex = 0
        ax.append(fig.add_subplot(111, projection='3d'))
        if not(gridValRaw is None):
            colormap = plt.cm.RdYlGn
            norm = mpl.colors.Normalize(vmin=np.amin(gridValRaw), vmax=np.amax(gridValRaw))
            gridValMin = 1
            gridValMax = 0


            grid_x = []
            grid_y = []
            grid_z = []
            gridVal = []
            for i, samplePoint in enumerate(self.samplePoints[self.boxType]):
                grid_x.append(samplePoint["pose"][0])
                grid_y.append(samplePoint["pose"][1])
                grid_z.append(samplePoint["pose"][2])
                gridVal.append(gridValRaw[i])

            gridValMin = min([gridValMin, np.nanmin(gridVal)])
            gridValMax = max([gridValMax, np.nanmax(gridVal)])

            #print("No: ", plotIndex, ' m: ', np.nanmin(gridVal), ' M: ', np.nanmax(gridVal))
            boxMarkerN.append(ax[plotIndex].scatter(grid_x, grid_y, grid_z, marker='o', c=gridVal, s=30, norm=norm, cmap=colormap))

        printedPoints = []
        markerSize = []
        points = np.ndarray((len(self.datastructure), 3))
        print("lenght of datastructure: ", len(self.datastructure))
        for i, data in enumerate(self.datastructure):
            print("Pose: ", i , " : ", data["pose"])
            points[i, :] = self.pose6Dto3D(data["pose"])
            encrypt = points[i, 0] *10000 + points[i, 1] * 100 + points[i, 2]
            if encrypt in printedPoints:
                print("BEREITS VORHANDEN!")
                markerSize.append(120)
            else:
                printedPoints.append(encrypt)
                markerSize.append(60)
        ax[plotIndex].scatter(points[:,0], points[:,1], points[:,2], marker='x', color='r', s=markerSize)

        if not(boundingBox == None):
            ax[plotIndex].set_xlim(boundingBox[0][0], boundingBox[0][1])
            ax[plotIndex].set_ylim(boundingBox[1][0], boundingBox[1][1])
            ax[plotIndex].set_zlim(boundingBox[2][0], boundingBox[2][1])

        ax[plotIndex].set_xlabel('X')
        ax[plotIndex].set_ylabel('Y')
        ax[plotIndex].set_zlabel('Z')
        #plt.title(title)
        # Customize the view angle
        ax[plotIndex].view_init(elev=15., azim=-30) #azim um z, links händisch #-5/-30
        #fig.colorbar(boxMarkerN[0], shrink=0.5, aspect=5)
        ifPlot = True
        if len(str(name)) == 1:
            name = '0' + str(name)
        if ifPlot:
            csv_write_dir = '/home/oliver/DATA/Raphael/20200220'
            fileName = 'computePose_' + self.dateTime + '_' + str(name) + '.png'
            absFilePath = os.path.join(csv_write_dir, fileName)
            plt.savefig(absFilePath, bbox_inches='tight', dpi=300)
        else:
            plt.show(block=block)
        if ifAnimation:
            plt.pause(2)
            plt.close()
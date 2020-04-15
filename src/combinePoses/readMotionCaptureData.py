import os
import numpy as np
import csv
import visualization
import workerStillStanding
import ergonomicRating
import qLearning
import poseScalarField
import tiling
import gridInterpolating
import computeNextPose
import ergonomicMeasure
import UR_communication as xml_rpc_communication
import codecs, json 
import matplotlib.pyplot as plt
import translationIndex
import pickle
from datetime import datetime

class CSVFile:
    def __init__(self, input_dir=None):
        if input_dir == None:
            #self.input_dir = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191204'
            self.input_dir = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191219'
        else:
            self.input_dir = input_dir
        self.file_paths = []
        self.updateFilePaths()

    def updateFilePaths(self):
        self.file_paths = []
        self.file_paths.extend([os.path.join(self.input_dir, file)
                            for file in os.listdir(self.input_dir)
                            if file.endswith('csv')])
        self.file_paths.sort()

    def parseFloat(self, value):
        try:
            return float(value)
        except:
            return float('nan')

    def readCSVFile(self, fileName, newFormat=False):
        data = []
        if newFormat:
            subHeader = ["rx", "ry", "rz", "rw", "tx", "ty", "tz", "t"]
        else:
            if self.input_dir == 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191204':
                subHeader = ["rx", "ry", "rz", "tx", "ty", "tz"]
            else:
                subHeader = ["rx", "ry", "rz", "rw", "tx", "ty", "tz"]
        with open(fileName, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            line_count = -1
            header = []
            for rowKomma in csv_reader:
                line_count += 1
                if newFormat:
                    if line_count == 0:
                        for elNr, el in enumerate(rowKomma):
                            #print("el: ", el)
                            if elNr % 8 == 0:
                                if el[:el.find("Q")] == "Thorax": #TODO namen des Thorax einfügen
                                    header.append("ThoraxRaphael")
                                else:
                                    header.append(el[:el.find("Q")])
                    else:
                        #print("old: ", rowKomma)
                        row = [float(str(el).replace(',', '.')) for el in rowKomma]
                        #print("new: ", row)
                        if len(row) >= len(header) * len(subHeader):
                            newLine = {}
                            for i in range(len(header)):
                                #if i < 2:
                                #    newLine[header[i]] = row[i] 
                                #else:
                                newLine[header[i]] = {}
                                if i == 0:
                                    newLine["timestamp"] = row[7] #kann aber notfalls auch ignoriert werden
                                for j in range(len(subHeader)-1): #Time wird ignoriert
                                    newLine[header[i]][subHeader[j]] = self.parseFloat(row[ i * len(subHeader) + j])
                            data.append(newLine)
                        else:
                            print("POTENTIAL FAILURE: ROW is in ReadCSVFile ignored")
                            pass
                            #line is ignored ~> time jump
                else: #oldDataForma
                    header = ["Frame", "Subframe"]
                    if line_count < 2 or line_count == 3 or line_count == 4 or len(row) == 0:
                        continue
                    elif line_count == 2:
                        for el in row:
                            if el != '':
                                if el[el.find(":")+1:] == "RScapula":
                                    header.append("RightScapula")
                                elif el[el.find(":")+1:] == "ThoraxRaphael_1": 
                                    header.append("ThoraxRaphael")
                                else:
                                    header.append(el[el.find(":")+1:])
                    else:
                        if len(row) >= 2 + (len(header)-2) * len(subHeader):
                            newLine = {}
                            for i in range(len(header)):
                                if i < 2:
                                    newLine[header[i]] = row[i]
                                else:
                                    newLine[header[i]] = {}
                                    for j in range(len(subHeader)):
                                        newLine[header[i]][subHeader[j]] = self.parseFloat(row[ 2 + (i-2) * len(subHeader) + j])
                            data.append(newLine)
                        else:
                            pass
                            #line is ignored ~> time jump
            print("header: ", header)
        return data

class AnnotationFile:
    def __init__(self, dir):
        self.input_dir = dir
        self.fileName = "annotations.txt"
        self.poseList = []
        if dir == 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191204':
            self.translateionIndex = translationIndex.translation_data_20191204
            self.annotationOffset = translationIndex.offset_20191204
        elif dir == 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191219':
            self.translateionIndex = translationIndex.translation_data_20191219()
            self.annotationOffset = []
        else:
            self.translateionIndex = {}
            self.annotationOffset = []

    def readFile(self):
        with open(os.path.join(self.input_dir, self.fileName), mode='r') as f:
            self.poseList = json.load(f)
        #print("self.poseList: ", self.poseList) #TODO stimmt das hier????
        return self.poseList
    
    def getPose(self, name):
        name = name.replace(" ", "")
        if self.input_dir == 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191204':
            if name.find('B') > -1:
                nameIndex = name[name.find('B'):name.find('B')+3]
            else:
                nameIndex = name[3:5] #[i for i in name.split() if str(i).isnumeric()]
        else:
            nameIndex = name[name.find('_')+1:name.find('.')]
        poseIndex = self.translateionIndex[nameIndex]
        pose = np.array(self.poseList[poseIndex]["pose"])
        for offset in self.annotationOffset:
            if poseIndex >= offset["start"] and poseIndex <= offset["end"]:
                pose += np.array(offset["displacement"])
        data = {"nameIndex": nameIndex, "poseIndex": poseIndex, "pose": pose.tolist()}
        return data


def printJoints(dataStructure):
    lineData = {}
    for data in dataStructure:
        for key in data["jointAvg"]["getBox"].keys():
            if not (key in lineData):
                lineData[key] = []
            lineData[key].append(data["jointAvg"]["getBox"][key])

    if False: #depositBox
        for data in dataStructure:
            for key in data["jointAvg"]["depositBox"].keys():
                if not ('d_' + key in lineData):
                    lineData['d_' + key] = []
                lineData['d_' + key].append(data["jointAvg"]["depositBox"][key])
    
    if True: #Action
        for data in dataStructure:
            for key in data["action"].keys():
                if not (key in lineData):
                    lineData[key] = []
                lineData[key].append(data["action"][key])

    fig, ax = plt.subplots()
    line = {}
    x = np.linspace(1, len(dataStructure), len(dataStructure))
    if "OUTOFMIDDLEDISTANCE" in lineData:
        lineData["OUTOFMIDDLEDISTANCE"] = np.array(lineData["OUTOFMIDDLEDISTANCE"])/10
    for key in lineData:
        line[key], = ax.plot(x, lineData[key], '-', linewidth=2,
                    label=key)

    ax.legend(loc='upper right')
    ax.grid(True)
    plt.show()


def readAllCSVFiles():
    csvFile = CSVFile()
    annotations = AnnotationFile(csvFile.input_dir)
    annotations.readFile()
    dataStructure = []

    stillStanding = workerStillStanding.WorkerStillStanding()
    for i, fileName in enumerate(csvFile.file_paths):
        posesDict = csvFile.readCSVFile(fileName)
        data = annotations.getPose(fileName[-10:]) # {"nameIndex", "poseIndex", "pose"}
        print("data: ", data["nameIndex"])
        #if i > 39 and i < 90: #Bereich für 1204_TestB
        #if data["nameIndex"] == "002" or data["nameIndex"] == "019" or data["nameIndex"] == "020":
        if True: #i < 20:
            print("data: ", data)
            poseLists = visualization.createPoseLists(posesDict)
            stepSize = 5
            displacementVectorNorm = stillStanding.calculate(poseLists, stepSize)
            displacementVectorNormMean = stillStanding.calcMin(displacementVectorNorm)
            [success, timeSequences, minArray] = stillStanding.findMinSection(displacementVectorNormMean, stepSize)
            if success:
                sequences = np.zeros(np.shape(displacementVectorNorm[0])[0])
                for sequence in timeSequences:
                    #if sequence["name"] == "getBox":
                    sequences[sequence["start"]:sequence["end"]+1] = np.ones(sequence["length"]+1)

                [score, additionalLines, jointAvg, handCenterAvg, path] = ergonomicRating.RatingClass.calculateScore(poseLists, timeSequences)
                data["score"] = score
                data["jointAvg"] = jointAvg
                data["handCenterAvg"] = handCenterAvg
                data["path"] = path
                print("data: ", data)
                dataStructure.append(data)
                
                #stillStanding.visual(displacementVectorNorm, displacementVectorNormMean, minArray, sequences, additionalLines, name=i)
                
                poseSeries = visualization.createPosSeries(posesDict)
                #visualization.animate(poseSeries, data["pose"], name=i) #sequences
        
    relativePath = "data_structure_20200115_test.json"
    absFilePath = os.path.join(csvFile.input_dir, relativePath)
    json.dump(dataStructure, codecs.open(absFilePath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
    return dataStructure

def runGeneratedFile(relativePath):
    #data = {"generation": {"RShoulder2D": RShoulder2D[idx], "RSHOULDEROPEN": RShoulderOpen[idx], "RELBW": RElbw[idx], "RHAND": RHand[idx], "LSHOULDEROPEN": LShoulderOpen[idx]}, "points": {}, "pose": []
    #input_dir = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191204'
    input_dir = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191219'
    fileName = "forward_data_20200115.json"
    fileName = "forward_data_20200115_short.json"
    fileName = "forward_data_20200117_short_120_140.json"
    fileName = "forward_data_20200120_5steps_120_140.json"
    #fileName = "forward_data_20200116.json"
    #fileName = "forward_data_20200115_very_short.json"
    obj_text = codecs.open(os.path.join(input_dir, fileName), 'r', encoding='utf-8').read()
    rawData = json.loads(obj_text)

    dataStructure = []

    extreme = {"min": 1000, "max": -1000}
    scoreExtreme = {}
    for i, el in enumerate(rawData):
        if i%100==0:
            print("dataStructure: ",i)
        data = {"nameIndex": str(i), "poseIndex": i, "pose": el["pose"]}
        if True: #i < 20:
            [score, jointAvg, handCenterAvg, appraiseAvg] = ergonomicRating.RatingClass.calculateScoreForward(el["points"])
            for sequence in score:
                if not(sequence in scoreExtreme):
                    scoreExtreme[sequence] = extreme
            if score[sequence] < scoreExtreme[sequence]["min"]:
                scoreExtreme[sequence]["min"] = score[sequence]
            if score[sequence] > scoreExtreme[sequence]["max"]:
                scoreExtreme[sequence]["max"] = score[sequence]
            data["score"] = score
            data["jointAvg"] = jointAvg
            data["handCenterAvg"] = handCenterAvg
            data["appraiseAvg"] = appraiseAvg
            dataStructure.append(data)
    scoreList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for sequence in scoreExtreme:
        difference = (scoreExtreme[sequence]["max"] - scoreExtreme[sequence]["min"])
        minValue = scoreExtreme[sequence]["min"]
        for data in dataStructure:
            data["score"][sequence] = (data["score"][sequence]-minValue) / difference
            if data["score"][sequence] < 0:
                print("Failer0000!")
            elif data["score"][sequence] < 0.1:
                scoreList[0] += 1
            elif data["score"][sequence] < 0.2:
                scoreList[1] += 1
            elif data["score"][sequence] < 0.3:
                scoreList[2] += 1
            elif data["score"][sequence] < 0.4:
                scoreList[3] += 1
            elif data["score"][sequence] < 0.5:
                scoreList[4] += 1
            elif data["score"][sequence] < 0.6:
                scoreList[5] += 1
            elif data["score"][sequence] < 0.7:
                scoreList[6] += 1
            elif data["score"][sequence] < 0.8:
                scoreList[7] += 1
            elif data["score"][sequence] < 0.9:
                scoreList[8] += 1
            elif data["score"][sequence] <= 1:
                scoreList[9] += 1
            elif data["score"][sequence] > 1:
                print("Failer111111!")
    print("scoreList: ", scoreList) #  [60, 342, 538, 757, 759, 812, 912, 849, 669, 252] ~> keine Spreizung notwendig...
            

    absFilePath = os.path.join(input_dir, relativePath)
    json.dump(dataStructure, codecs.open(absFilePath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
    return dataStructure


    #ToDos:
    #paket posen anzeigen
    #   state: 4 Gelenkwinkel
    #   action: für alle verschiebungsvektor(poseNeu-posealt) 
    #   [new State: eigentlich aus poseNeu ~> ignorieren]
    #   reward ist das Gütemaß(poseNeu)
    # To discuss:
    #   diskretisieren / kontinuiesierung des Raumes?
    #       Forderung: nur kleine Bewgungen ~> quasikonstante Winkel ~> für jede Pose wird ein konstanter Winkel berechnet:
    #       Ansatz: Arbeiter tritt näher heran, arme werden angezogen ~> Roboter geht zurück oder macht neue Bewegung um Situation aufzubrechen ~> absolute Position ist irrelevant
    #       Annahme: zur Datengenierung jedoch muss die Position absolut konstant bleiben, da die Positionen mit einander verglichen werden!
    #       III) neuer Zustand: Winkel spannen einen 4D Raum auf ~> suche nach pose mit minimalem Abstand ~> wähle Aktion entsprechend Training bzw. zeichne Aktivierungskarte
    # jeder mit jedem trainieren 40² = 1600 Optionen

    #getrennte Datensätze verknüpfen

def findNextPoseInDataStructure(dataStructure, pose):
    for index, el in enumerate(dataStructure):
        if np.all(el["pose"] == pose):
            return dataStructure[index]
    print("pose: ", pose)
    return {"Failed": True}

def pose6Dto3D(pose):
    return np.array(pose[:3])

def evalPoses(dataStructure):
   # for data in dataStructure:
   #     print("name: ", data["nameIndex"], " ang1: ", data["jointAvg"]["getBox"]["HANDDISTANCE"], " - ", data["jointAvg"]["getBox"]["SHOULDERDISTANCE"], " - ", data["jointAvg"]["getBox"]["LUPPERARM"] - data["jointAvg"]["getBox"]["RUPPERARM"])
   # for data in dataStructure:
   #    print("name: ", data["nameIndex"], " ang1: ", data["jointAvg"]["getBox"]["LSHOULDER2D"], " - ", data["jointAvg"]["getBox"]["RSHOULDER2D"], " - ", data["jointAvg"]["getBox"]["LSHOULDER2D"] - data["jointAvg"]["getBox"]["RSHOULDER2D"])

    evalList = {
      "HANDDISTANCE": [],
      "SHOULDERDISTANCE": [],
      "LUPPERARM": [],
      "LFOREARM": [],
      "RUPPERARM": [],
      "RFOREARM": [],
      "upperDiff": [],
      "foreDiff": []
    }
    minPose = np.zeros(6)
    maxPose = np.zeros(6)
    minPose[0] = 2
    minPose[2] = 2
    minPoses = [1, 2, 3]
    maxPoses = [1, 2, 3]
    score = {}
    for i, data in enumerate(dataStructure):
        for key in evalList.keys():
            if key == "upperDiff":
                evalList[key].append(data["jointAvg"]["getBox"]["LUPPERARM"]-data["jointAvg"]["getBox"]["RUPPERARM"])
            elif key == "foreDiff":
                evalList[key].append(data["jointAvg"]["getBox"]["LFOREARM"]-data["jointAvg"]["getBox"]["RFOREARM"])
            else:
                evalList[key].append(data["jointAvg"]["getBox"][key])
        for i, el in enumerate(data["pose"]):
            if el < minPose[i]:
                minPose[i] = el
                minPoses[i] = data["pose"]
            elif el > maxPose[i]:
                maxPose[i] = el
                maxPoses[i] = data["pose"]
        if "appraiseAvg" in data:
            for key in data["appraiseAvg"]["getBox"].keys():
                if not(key in score):
                    score[key] = []
                score[key].append(data["appraiseAvg"]["getBox"][key])
    print("data: ", dataStructure[0])
        



    print("min: ", minPose)
    print("max: ", maxPose)
    print("dif: ", maxPose-minPose)
    print("min: ", minPoses)
    print("max: ", maxPoses)
    for key in score.keys():
        pass
        #print("KEY: ", key)
        #print(score[key])
    #for key in evalList.keys():
    #    print("LIST: ", key)
    #    print(evalList[key])
    #for key in evalList.keys():
    #    print(key, ": ", np.nanmedian(evalList[key]))
        #forearm: 225
        #upperarm: 400
        #shoulderdistance: 286
        #Handdistance: 475


class DataProcessing():
    def __init__(self, testMode=True, poses=None):
        print("dataProcessing")
        self.testMode = testMode
        if testMode:
            newDataStructure = False #read the real Data of the csv files
            if newDataStructure:
                self.testDataStructure = readAllCSVFiles()
            else:
                #input_dir = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191204'
                input_dir = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191219'
                fileName = "data_structure_20200109_widerRating.json"
                fileName = "data_structure_20200109.json"
                fileName = "data_structure_20200113.json"
                fileName = "data_structure_20200115_test.json"
                #fileName = "data_structure_20200109_short.json"
                obj_text = codecs.open(os.path.join(input_dir, fileName), 'r', encoding='utf-8').read()
                self.testDataStructure = np.array(json.loads(obj_text)) #TODO remove np.array
                for el in self.testDataStructure:
                    el["pose"] = np.array(el["pose"])
                    for elPath in el["path"]:
                            elPath["path"] = np.array(elPath["path"])
        
        # computeGeneratedTrainingData==False:
        #input_dir_qFile = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191219'
        #fileName = 'qValue_20200120_10000.pkl'
        #fileName = "qValue_20200120_300_0-1.pkl"
        fileName = 'qValue_20200117_forward_short_120_140_50000_0-1.pkl'
        #with open(os.path.join(input_dir_qFile, fileName), 'rb') as input:
        with open(fileName, 'rb') as input: #TODO change the directory
            qValueFunction = pickle.load(input)
        #qValueFunction.printQTable3D() #Plots für jede Action des Gelenkwinkelraum des Arbeiters mit dem Gütemaß als Skalar
        print("QModel ist geladen!")
        poseTilingsShape = {"numberTilings": 1, "boundingBox": [[-0.2, 0.3], [-0.4, 0.4], [-0.3, 0.5]], "sampling": [0.100, 0.100, 0.100]}
        if poses == None:
            poses = xml_rpc_communication.Pose()
        samplePoints = [([{"pose": pose6Dto3D(el)} for el in boxList.correctedList]) for boxList in poses.box] #0: big Box, 1: small Box
        self.stillStanding = workerStillStanding.WorkerStillStanding()
        self.poseScalar = poseScalarField.PoseScalarField(poseTilingsShape, qValueFunction) # brauch ich nicht
        self.gridInterpol = gridInterpolating.GridInterpolating(poseTilingsShape, samplePoints)
        self.ergonomicMeasureObj = ergonomicMeasure.ErgonomicMeasure(poseTilingsShape, qValueFunction, samplePoints)
        self.nextPoseObj = computeNextPose.NextPose(poseTilingsShape, qValueFunction, self.gridInterpol, self.ergonomicMeasureObj)
        #self.poseScalar.printAllPosesForEachAction3D(dataStructure, poseTilingsShape["boundingBox"]) #PLOT POSE FOR EACH ACTION
        self.randomSeed = 7 #TODO change seed for each iteration!!!
        self.runIterator = 0
        csv_input_dir = '/home/oliver/DATA/Raphael/20200220' #TODO change the directory
        csv_write_dir = '/home/oliver/DATA/Raphael/20200220' #TODO change the directory
        self.csvFile = CSVFile(csv_input_dir)
        now = datetime.now()
        d = now.strftime("%Y%m%d_%H_%M_%S")
        relativePath = "data_structure_experiment_" + d + "_test.json"
        self.absFilePath_dataStructure = os.path.join(csv_write_dir, relativePath)
        with open(self.absFilePath_dataStructure, 'a') as file:
            file.write("[\n")
        self.nextPose = np.zeros(6)
        print("class Dataprocessing init is finished")

    def getNextPose(self): #auch add nextPose
        print("getNextPose: ", self.runIterator)
        if self.runIterator == 0:
            self.nextPose = self.nextPoseObj.estimateNextPose() # Estimate 1. Pose
        else:
            print("last nextPose was: ", self.nextPose)
            #self.poseScalar.getScalarField(dataStructure[i]["jointAvg"], dataStructure[i]["pose"])
            #self.poseScalar.printPoseTable3D()

            ifNewDataStructure = True
            if self.testMode:
                newDataStructure = findNextPoseInDataStructure(self.testDataStructure, self.nextPose)
                if "Failed" in newDataStructure:
                    print("Zielpose konnte in Datastructure nicht gefunden werden!")
                    ifNewDataStructure =  False
                newDataStructure["pose"] = newDataStructure["pose"].tolist()
                for el in newDataStructure["path"]:
                    el["path"] = el["path"].tolist()
            else: #wir sind hier...
                newDataStructure = self.readNewestCSVFile(self.nextPose, self.runIterator)
                if "runAnalysingFailed" in newDataStructure:
                    print("Datastructure konnte nicht ausgewertet werden!")
                    ifNewDataStructure = False
                else:
                    print("hasn't failed...")
            #TODO: testmode um abgelaufenes Programm zu wiederholen
            newDataStructure["seed"] = self.randomSeed
            with open(self.absFilePath_dataStructure, 'a') as file:
                json.dump(newDataStructure, file, separators=(',', ':'), sort_keys=True, indent=4)
                file.write(",\n")

            if ifNewDataStructure:
                self.gridInterpol.addRun(newDataStructure)
                self.ergonomicMeasureObj.addRun(newDataStructure)
                self.nextPoseObj.addRun(newDataStructure)
            else:
                self.randomSeed += 1
            self.nextPose = self.nextPoseObj.estimateNextPose(self.randomSeed)
            
            ifPrint = False
            if ifPrint:
                ergonomicMeasureObj.printErgonomicMeasure3D()
                gridInterpol.calcDistanceTable3D(False)

        self.runIterator +=1
        return self.nextPose

    def readNewestCSVFile(self, nextPose, iteration, ):
        #print("in read File")
        self.csvFile.updateFilePaths()
        #print("list: ", self.csvFile.file_paths)
        print("found data: ", len(self.csvFile.file_paths), " iteration: ", iteration)
        if not(iteration == len(self.csvFile.file_paths)):
            print("ERROR their are to less data")
            return None
        filename = self.csvFile.file_paths[-1]#TODO unklar wie die aktuellste Datei ausgewählt wird
        print("filename: ", filename)
        posesDict = self.csvFile.readCSVFile(filename, True) 
        data = {"nameIndex": filename[-7:-4], "poseIndex": iteration, "pose": nextPose.tolist()} #TODO: Namensbereich der Datei anpassen
        print("data: ", data["nameIndex"])

        poseLists = visualization.createPoseLists(posesDict)
        #print("poseLists: ", poseLists)
        stepSize = 5
        displacementVectorNorm = self.stillStanding.calculate(poseLists, stepSize)
        #print("displacementVectorNorm: ", displacementVectorNorm)
        displacementVectorNormMean = self.stillStanding.calcMin(displacementVectorNorm)
        #print("displacementVectorNormMean: ", displacementVectorNormMean)
        [success, timeSequences, minArray] = self.stillStanding.findMinSection(displacementVectorNormMean, stepSize)
        print("Success: ", success, " time: ", timeSequences, )
        if success:
            print("timesequence: success")
            sequences = np.zeros(np.shape(displacementVectorNorm[0])[0])
            for sequence in timeSequences:
                #if sequence["name"] == "getBox":
                sequences[sequence["start"]:sequence["end"]+1] = np.ones(sequence["length"]+1)

            [score, additionalLines, jointAvg, handCenterAvg, path] = ergonomicRating.RatingClass.calculateScore(poseLists, timeSequences)
            data["score"] = score
            data["jointAvg"] = jointAvg
            data["handCenterAvg"] = handCenterAvg
            data["path"] = path
            print("data: ", data)
            #self.stillStanding.visual(displacementVectorNorm, displacementVectorNormMean, minArray, sequences, additionalLines, name=i)
        else:
            print("timesequence: failed")
            data["runAnalysingFailed"] = True

        return data




if __name__ == "__main__":
    test = False
    if test:
        dataProcessing = DataProcessing()
        for i in range(20):
            dataProcessing.getNextPose()
    else:
        newDataStructure = False #read the real Data of the csv files
        useGeneratedTrainingData = True #use the Model as training-Data-set
        computeGeneratedTrainingData = False #save new Object
        if newDataStructure:
            dataStructure = readAllCSVFiles()
        else:
            #input_dir = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191204'
            input_dir = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191219'
            fileName = "data_structure_20200109_widerRating.json"
            fileName = "data_structure_20200109.json"
            fileName = "data_structure_20200113.json"
            fileName = "data_structure_20200115_test.json"
            #fileName = "data_structure_20200109_short.json"
            obj_text = codecs.open(os.path.join(input_dir, fileName), 'r', encoding='utf-8').read()
            dataStructure = np.array(json.loads(obj_text)) #TODO remove np.array
            for el in dataStructure:
                el["pose"] = np.array(el["pose"])
                for elPath in el["path"]:
                        elPath["path"] = np.array(elPath["path"])
        
        q = True
        if q: ###trainiert das q-File###
            input_dir_qFile = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191219'

            if computeGeneratedTrainingData:
                if useGeneratedTrainingData:
                    relativePath = "data_structure_forward_20200117_short_120_140_test.json"
                    dataStructureTraining = runGeneratedFile(relativePath)
                else:
                    dataStructureTraining = dataStructure
                #print("Eval DataStructureTraining: ")
                #evalPoses(dataStructureTraining)
                actions = qLearning.Actions()
                dimensionsReduction = True
                outOfMiddleDistance = True
                #featureRange: Elbow, shoulder2D, Z-Center
                #stateTilingsShape = {"numberTilings": 1, "featureRange": [[65, 115], [0, 30], [-150, 150]], "bins": [7, 5, 5]} # einfacher Fall
                #stateTilingsShape = {"numberTilings": 2, "featureRange": [[65, 115], [0, 30], [-150, 150]], "bins": [7, 5, 5]} # sehr feine auflösung eher uninteressant
                #stateTilingsShape = {"numberTilings": 2, "featureRange": [[65, 115], [0, 30], [-150, 150]], "bins": [6, 4, 4]} # kompromis
                #stateTilingsShape = {"numberTilings": 3, "featureRange": [[65, 115], [0, 30], [-150, 150]], "bins": [6, 4, 4]}
                #stateTilingsShape = {"numberTilings": 3, "featureRange": [[65, 115], [0, 30], [-150, 150]], "bins": [5, 3, 3]} # reduzierte Daten
                #stateTilingsShape = {"numberTilings": 3, "featureRange": [[65, 115], [0, 60], [-150, 150]], "bins": [5, 5, 3]} # reduzierte Daten
                stateTilingsShape = {"numberTilings": 3, "featureRange": [[65, 135], [0, 110], [-150, 150]], "bins": [7, 9, 3]} # reduzierte Daten
                #stateTilingsShape = {"numberTilings": 4, "featureRange": [[65, 115], [0, 30], [-150, 150]], "bins": [5, 3, 3]}
                qValueFunction = qLearning.QValueFunction(stateTilingsShape, actions, 0.5, dataStructureTraining, dimensionsReduction=dimensionsReduction, outOfMiddleDistance=outOfMiddleDistance)
                qValueFunction.trainWithAllData(useGeneratedTrainingData)
                fileName = 'qValue_20200117_forward_short_120_140_50000_0-1_test.pkl'

                with open(os.path.join(input_dir_qFile, fileName), 'wb') as output:
                    pickle.dump(qValueFunction, output, pickle.HIGHEST_PROTOCOL)
            else:
                fileName = 'qValue_20200120_10000.pkl'
                #fileName = "qValue_20200120_300_0-1.pkl"
                fileName = 'qValue_20200117_forward_short_120_140_50000_0-1.pkl'
                with open(os.path.join(input_dir_qFile, fileName), 'rb') as input:
                    qValueFunction = pickle.load(input)

            #print("testValu: ", qValueFunction.value([100, 17, 100], "halt")) # test Value
            # print("sruct; ", qValueFunction.dataStructure)
            # print("q_table: ", qValueFunction.q_tables)
            #print("q_table: ", np.shape(qValueFunction.q_tables))
            #qValueFunction.printQTable3D() #Plots für jede Action des Gelenkwinkelraum des Arbeiters mit dem Gütemaß als Skalar
            #printJoints(qValueFunction.dataStructure)
            # print("dataStructure: ", dataStructure)
        pose = True
        if pose: ###wertet die einzelnen Posen nacheinander aus###
            poseTilingsShape = {"numberTilings": 1, "boundingBox": [[-0.2, 0.3], [-0.4, 0.4], [-0.3, 0.5]], "sampling": [0.100, 0.100, 0.100]}
            poses = xml_rpc_communication.Pose()
            samplePoints = [([{"pose": pose6Dto3D(el)} for el in boxList.correctedList]) for boxList in poses.box] #0: big Box, 1: small Box
            qValueFunction.printStates3D(dataStructure) #Only for testing purpose, to see where are the training data in the state-room #TODO
            poseScalar = poseScalarField.PoseScalarField(poseTilingsShape, qValueFunction) # brauch ich nicht
            gridInterpol = gridInterpolating.GridInterpolating(poseTilingsShape, samplePoints)
            ergonomicMeasureObj = ergonomicMeasure.ErgonomicMeasure(poseTilingsShape, qValueFunction, samplePoints)
            nextPoseObj = computeNextPose.NextPose(poseTilingsShape, qValueFunction, gridInterpol, ergonomicMeasureObj)
            #PLOT POSE FOR EACH ACTION 
            #poseScalar.printAllPosesForEachAction3D(dataStructure, poseTilingsShape["boundingBox"])


            if False: # alle runs durchgehen
                for i in range(len(dataStructure)):
                    poseScalar.getScalarField(dataStructure[i]["jointAvg"], dataStructure[i]["pose"])
                    gridInterpol.addRun(dataStructure[i])
                    nextPoseObj.addRun(dataStructure[i])
                    nextPoseObj.estimateNextPose()
                    if i % 10 == 0:
                        gridInterpol.calcDistanceTable3D(False)
                        poseScalar.printPoseTable3D()
            else:
                nextPose = nextPoseObj.estimateNextPose() # Estimate 1. Pose
                for i in range(20):
                    #poseScalar.getScalarField(dataStructure[i]["jointAvg"], dataStructure[i]["pose"])
                    #poseScalar.printPoseTable3D()
                    testingInLoop = True
                    print("nextPose: ", nextPose)
                    if testingInLoop:
                        newDataStructure = findNextPoseInDataStructure(dataStructure, nextPose)
                        if "Failed" in newDataStructure:
                            print("Zielpose konnte in Datastructure nicht gefunden werden!")
                            break
                    gridInterpol.addRun(newDataStructure)
                    ergonomicMeasureObj.addRun(newDataStructure)
                    nextPoseObj.addRun(newDataStructure)
                    nextPose = nextPoseObj.estimateNextPose() 
                    if i % 1 == 0:
                        #ergonomicMeasureObj.printErgonomicMeasure3D()
                        # gridInterpol.calcDistanceTable3D(False)
                        pass
                        

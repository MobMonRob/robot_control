import numpy as np

class Evaluation:
  acceptableMin = 0
  fitMin = 0
  fitMax = 0
  acceptableMax = 0

  def eval2(self, value): # 1 is best rating, 0 is worst
    if np.isnan(value):
      return np.NaN
    elif (value > self.acceptableMax):
      return 0
    elif (value > self.fitMax):
      return (self.acceptableMax-value)/(self.acceptableMax-self.fitMax)
    elif (value > self.fitMin):
      return 1
    elif (value > self.acceptableMin):
      return (value-self.acceptableMin)/(self.fitMin-self.acceptableMin)
    else:
      return 0

  def eval(self, value): # 1 is best rating, ?? is worst #TODO: welches eval?
    if np.isnan(value):
      return np.NaN
    elif (value > self.fitMax):
      return (self.acceptableMax-value)/(self.acceptableMax-self.fitMax)
    elif (value > self.fitMin):
      return 1
    else:
      return (value-self.acceptableMin)/(self.fitMin-self.acceptableMin)

class Timestamp():
  def eval(self, value): 
    return value

class ElbW(Evaluation): #symmetrisch um 90°
  acceptableMin = 65  
  fitMin = 75
  fitMax = 105
  acceptableMax = 115
  weight = 1
  # :65:75:85:95:105:115:

class Shoulder2D(Evaluation): # in degree
  acceptableMin = -5
  fitMin = 0
  fitMax = 20
  acceptableMax = 30
  weight = 1
  # :0:10:20:30:  #vielleicht noch datenpunkte über 50°

class outOfMiddleDistance(Evaluation): # in mm
  acceptableMin = -150
  fitMin = -50
  fitMax = 50
  acceptableMax = 150
  weight = 2
  # :-150:-50:50:150:

class ElbW2(Evaluation): #symmetrisch um 90°
  acceptableMin = 55  
  fitMin = 85
  fitMax = 95
  acceptableMax = 125
  weight = 1
  # :65:75:85:95:105:115:

class Shoulder2D2(Evaluation): # in degree
  acceptableMin = -5
  fitMin = 10
  fitMax = 20
  acceptableMax = 50
  weight = 1
  # :0:10:20:30:  #vielleicht noch datenpunkte über 50°

class TranslationTable:
  normalVersion = False #TODO: evaluate
  if normalVersion:
    LELBW = ElbW()
    RELBW = ElbW()
    LSHOULDER2D = Shoulder2D()
    RSHOULDER2D = Shoulder2D()
  else:
    LELBW = ElbW2()
    RELBW = ElbW2()
    LSHOULDER2D = Shoulder2D2()
    RSHOULDER2D = Shoulder2D2()
  OUTOFMIDDLEDISTANCE = outOfMiddleDistance()
  TIME = Timestamp()

class RatingClass:
  @staticmethod
  def _appraiseAvgPoints(jointAvg):
    appraiseAvg = {}
    for key in jointAvg.keys():
      appraiseAvg[key] = {}
      for key2 in jointAvg[key].keys():
        if key2 == "timestamp":
          appraiseAvg[key][key2] = jointAvg[key][key2]
        elif key2 == "HANDDISTANCE" or key2 == "SHOULDERDISTANCE" or key2 == "LUPPERARM" or key2 == "LFOREARM" or key2 == "RUPPERARM" or key2 == "RFOREARM":
          continue
        else:
          appraiseAvg[key][key2] = getattr(TranslationTable, key2).eval(jointAvg[key][key2])
    return appraiseAvg
 
  @staticmethod
  def _appraiseDatapoints(jointStream):
    appraiseStream = {}
    for key in jointStream.keys():
      appraiseStream[key] = []
    for key in jointStream.keys():
      for data in jointStream[key]:
        value = getattr(TranslationTable, key).eval(data)
        appraiseStream[key].append(value)
    return appraiseStream

  @staticmethod
  def _cumulateTimeline(appraiseStream, timeSequences=None): #ToDo sequences
    averageValues = {}
    if timeSequences == None:
      averageValues["wholeSequence"] = {}
      for key in appraiseStream.keys():
        averageValues["wholeSequence"][key] = np.nanmedian(np.array(appraiseStream[key]))  #mean oder median, anderes quantil?
    else:
      for timeSequence in timeSequences:
        averageValues[timeSequence["name"]] = {}
        averageValues[timeSequence["name"]]["timestamp"] = (timeSequence["start"]+timeSequence["end"])/2+timeSequence["halfStepSize"]
        for key in appraiseStream.keys():
          averageValues[timeSequence["name"]][key] = np.nanmedian(np.array(appraiseStream[key][timeSequence["start"]+timeSequence["halfStepSize"]:timeSequence["end"]+timeSequence["halfStepSize"]]))  #mean oder median, anderes quantil?
          #print("avg: ", averageValues[timeSequence["name"]][key], " stream: ", np.array(appraiseStream[key][timeSequence["start"]+timeSequence["halfStepSize"]:timeSequence["end"]+timeSequence["halfStepSize"]]))
    return averageValues

  @staticmethod
  def _weightedScore(averageValues):
    weightedScore = {}
    for sequence in averageValues.keys():
      counter = 0
      score = 0
      #print("!!!sequence: ", sequence)
      for key in averageValues[sequence].keys(): #mean oder median, anderes quantil?
        #if key == "OUTOFMIDDLEDISTANCE": #TODO: Gewicht zu ordnen
        #  continue
        if key == "timestamp":
          continue
        weight = getattr(TranslationTable, key).weight
        counter += 1 * weight
        score += averageValues[sequence][key] * weight
        #print(key, ": ", round(averageValues[sequence][key], 1))
      weightedScore[sequence] = score/counter
    # print('weightedScore: ', weightedScore)  
    return weightedScore

  @staticmethod
  def _weightedScoreMin(averageValues):
    weightedScore = {}
    for sequence in averageValues.keys():
      score = 1
      for key in averageValues[sequence].keys(): #mean oder median, anderes quantil?
        if key == "timestamp":
          continue
        if averageValues[sequence][key] < score:
          score = averageValues[sequence][key]
          #print("key: ", key, " ", score)
      weightedScore[sequence] = score
    return weightedScore

  @staticmethod
  def _weightedScoreMin2(averageValues):
    weightedScore = {}
    for sequence in averageValues.keys():
      score = 0
      #print("!!!sequence: ", sequence)
      scores = np.zeros(3)
      weights = np.zeros(3)
      for key in averageValues[sequence].keys(): #mean oder median, anderes quantil?
        if key == "RELBW" or key == "LELBW":
          weight = getattr(TranslationTable, key).weight
          weights[0] += weight
          scores[0] += averageValues[sequence][key] * weight
        elif key == "LSHOULDER2D" or key == "RSHOULDER2D":
          weight = getattr(TranslationTable, key).weight
          weights[1] += weight
          scores[1] += averageValues[sequence][key] * weight
        elif key == "OUTOFMIDDLEDISTANCE":
          weight = getattr(TranslationTable, key).weight
          weights[2] += weight
          scores[2] += averageValues[sequence][key] * weight
      scores = np.sort(scores/weights)
      weightedScore[sequence] = np.mean(scores[:2])
    #print('weightedScore: ', weightedScore)  
    return weightedScore

  @staticmethod
  def _weightedPath(averageValues):
    paths = []
    sequences = []
    for sequence in averageValues.keys():
      sequences.append({'pos': np.array([averageValues[sequence]['x'], averageValues[sequence]['y'], averageValues[sequence]['z']]), 'timestamp': averageValues[sequence]['timestamp'], "name": sequence})
    for i, sequence1 in enumerate(sequences):
      for sequence2 in sequences[i+1:]:
          path = {}
          path["start"] = sequence1["name"]
          path["end"] = sequence2["name"]
          path["duration"] = sequence2["timestamp"]-sequence1["timestamp"]
          path["path"] = (sequence2["pos"]-sequence1["pos"]).tolist()
          path["distance"] = np.linalg.norm(path["path"])
          paths.append(path)
    return paths

  @staticmethod
  def _calculateAngle3D(point1, pointCenter, point2):
    joint = np.zeros(len(point1[0]))
    for i in range(len(point1[0])):
      v1 = point1[:, i] - pointCenter[:, i]
      v2 = point2[:, i] - pointCenter[:, i]
      if np.any(np.isnan([v1, v2])):
        joint[i] = np.NaN
        continue
      # dot = x1*x2 + y1*y2 + z1*z2    #between [x1, y1, z1] and [x2, y2, z2]
      # lenSq1 = x1*x1 + y1*y1 + z1*z1
      # lenSq2 = x2*x2 + y2*y2 + z2*z2
      dot = np.inner(v1, v2)
      lenSq1 = np.inner(v1, v1)
      lenSq2 = np.inner(v2, v2)
      joint[i] = np.arccos(dot/np.sqrt(lenSq1 * lenSq2))*180/3.14159
    return joint

  @staticmethod
  def _calculateAngleProjection2D(point1, point2, pointN1, pointN2):
    # Ebene bestimmt durch den Schulter-Schulter-Vektor
    # Schulter-Ellenbogen auf die Ebene projezieren ~> 2D
    # Torso-Schultermittelpunkt auf die Ebene projezieren ~> 2D
    # Winkel in der 2D Ebene bestimmen
    # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors/16544330#16544330
    joint = np.zeros(len(point1[0]))
    for i in range(len(point1[0])):
      m = (pointN1[:, i] + pointN2[:, i])/2
      v1 = point1[:, i] - m
      v2 = point2[:, i] - m
      n = pointN1[:, i] - m
      n = n/np.sqrt(np.inner(n, n))
      if np.any(np.isnan([v1, v2, n])):
        joint[i] = np.NaN
        continue
      # dot = x1*x2 + y1*y2 + z1*z2
      # det = x1*y2*zn + x2*yn*z1 + xn*y1*z2 - z1*y2*xn - z2*yn*x1 - zn*y1*x2
      # angle = atan2(det, dot)
      dot = np.inner(v1, v2)
      det = np.linalg.det(np.array([v1, v2, n]))
      joint[i] = np.arctan2(det, dot)*180/3.14159
    return joint

  @staticmethod
  def _calculateCenter(point1, point2):
    center = np.zeros([3, len(point1[0])])
    for i in range(len(point1[0])):
      center[:, i] = (point1[:, i] + point2[:, i])/2
    return {"x": center[0,:], "y": center[1,:], "z": center[2,:]}

  @staticmethod
  def _calculateOutOfMiddleDinstance(pointVector1A, pointVector2A, pointVector2B, pointVector1B): #0358; 1:Handgelenk 2:SChulter
    # durch den Mittelpunkt der Schultern und dem Schulternvektor wird eine Ebene aufgespannt
    # Abstand zwischen dieser Ebene und dem Mittelpunkt der Handgelenke wird bestimmt
    # weitere Ootionen:
    # -2.Winkel der Schultergelenke: Winkel zwischen (Handgelenk-Ellenbogen/Schulter) zu (Schulter-Schulter)-Bewegung, in der horizontalen Ebene bestimmt durch Schultermitte und Torso
    # -Oberarmgelenke:
    distance = np.zeros(len(pointVector1A[0]))
    for i in range(len(pointVector1A[0])):
      pointMiddle1 = (pointVector1A[:, i] + pointVector1B[:, i])/2
      pointMiddle2 = (pointVector2A[:, i] + pointVector2B[:, i])/2
      vector2 = (pointVector2B[:, i] - pointVector2A[:, i])

      if np.any(np.isnan([pointMiddle1, pointMiddle2, vector2])):
        distance[i] = np.NaN
        continue

      n0 = vector2/np.sqrt(np.inner(vector2, vector2)) # normierter Normalenvektor der Ebene
      distance[i] = np.inner((pointMiddle1 - pointMiddle2), n0)
      # print("pm1: ", pointMiddle1, " pm2: ", pointMiddle2, " n0: ", n0, "dist: ", pointMiddle1 - pointMiddle2 ," d: ", distance[i]) # distance springt wenn leftForehand spring
    return distance
  
  @staticmethod
  def _calculateDistance(pointVectorA, pointVectorB):
    #distance = np.zeros(len(pointVectorA[0]))
    #for i in range(len(pointVectorA[0])):
    #  distance[i] = np.linalg.norm(pointVectorB[:,i]-pointVectorA[:,i])
    distance = np.linalg.norm(pointVectorB-pointVectorA, axis=0)
    return distance

       

  @staticmethod
  def calculateScore(dataStream, timeSequences): # 9, 3, 457
    dataStream = np.array(dataStream)
    jointStream = {
      "LELBW": RatingClass._calculateAngle3D(dataStream[0], dataStream[1], dataStream[3]),
      "RELBW": RatingClass._calculateAngle3D(dataStream[8], dataStream[7], dataStream[5]),
      "LSHOULDER2D": RatingClass._calculateAngleProjection2D(dataStream[1], dataStream[4], dataStream[3], dataStream[5]), #durch Vektor der Schultern bestimmt
      "RSHOULDER2D": RatingClass._calculateAngleProjection2D(dataStream[7], dataStream[4], dataStream[3], dataStream[5]),
      "OUTOFMIDDLEDISTANCE": RatingClass._calculateOutOfMiddleDinstance(dataStream[0], dataStream[3], dataStream[5], dataStream[8]),
      "HANDDISTANCE": RatingClass._calculateDistance(dataStream[0], dataStream[8]),
      "SHOULDERDISTANCE": RatingClass._calculateDistance(dataStream[3], dataStream[5]),
      "LUPPERARM": RatingClass._calculateDistance(dataStream[1], dataStream[3]),
      "LFOREARM": RatingClass._calculateDistance(dataStream[0], dataStream[1]),
      "RUPPERARM": RatingClass._calculateDistance(dataStream[7], dataStream[5]),
      "RFOREARM": RatingClass._calculateDistance(dataStream[8], dataStream[7]),
      }

    jointAvg = RatingClass._cumulateTimeline(jointStream, timeSequences)
    oldWay = False
    if oldWay:
      appraiseStream = RatingClass._appraiseDatapoints(jointStream)
      appraiseAvg = RatingClass._cumulateTimeline(appraiseStream, timeSequences)
    else:
      appraiseAvg = RatingClass._appraiseAvgPoints(jointAvg)
    score = RatingClass._weightedScore(appraiseAvg)

    handCenterStream = RatingClass._calculateCenter(dataStream[0], dataStream[8])
    handCenterAvg = RatingClass._cumulateTimeline(handCenterStream, timeSequences)
    path = RatingClass._weightedPath(handCenterAvg)

    # print("jointStream: ", jointStream)
    # print("jointAvg: ", jointAvg)
    # print("appraiseAvg: ", appraiseAvg)
    # print("score: ", score)

    return score, jointStream, jointAvg, handCenterAvg, path


  @staticmethod
  def calculateScoreForward(inputDataPoints): # 9, 3, 45
    #   dataPoints = {"RSHOULDER2D", "RELBW", "RHand", "LSHOULDER2D", "LELBW", "LHand", "TORSODOWN"}
    timeSequences = [{'name': 'getBox', 'start': 0, 'end': 1, 'length': 1, 'halfStepSize': 0}]
    dataPoints = {}
    for key in inputDataPoints.keys():
      dataPoints[key] = np.array([[inputDataPoints[key][0]], [inputDataPoints[key][1]], [inputDataPoints[key][2]]])
    jointStream = {
      "LELBW": RatingClass._calculateAngle3D(dataPoints["LHand"], dataPoints["LELBW"], dataPoints["LSHOULDER2D"]),
      "RELBW": RatingClass._calculateAngle3D(dataPoints["RHand"], dataPoints["RELBW"], dataPoints["RSHOULDER2D"]),
      "LSHOULDER2D": RatingClass._calculateAngleProjection2D(dataPoints["LELBW"], dataPoints["TORSODOWN"], dataPoints["LSHOULDER2D"], dataPoints["RSHOULDER2D"]), #durch Vektor der Schultern bestimmt
      "RSHOULDER2D": RatingClass._calculateAngleProjection2D(dataPoints["RELBW"], dataPoints["TORSODOWN"], dataPoints["LSHOULDER2D"], dataPoints["RSHOULDER2D"]),
      "OUTOFMIDDLEDISTANCE": RatingClass._calculateOutOfMiddleDinstance(dataPoints["LHand"], dataPoints["LSHOULDER2D"], dataPoints["RSHOULDER2D"], dataPoints["RHand"]),
      "HANDDISTANCE": RatingClass._calculateDistance(dataPoints["LHand"], dataPoints["RHand"]),
      "SHOULDERDISTANCE": RatingClass._calculateDistance(dataPoints["LSHOULDER2D"], dataPoints["RSHOULDER2D"]),
      "LUPPERARM": RatingClass._calculateDistance(dataPoints["LELBW"], dataPoints["LSHOULDER2D"]),
      "LFOREARM": RatingClass._calculateDistance(dataPoints["LHand"], dataPoints["LELBW"]),
      "RUPPERARM": RatingClass._calculateDistance(dataPoints["RELBW"], dataPoints["RSHOULDER2D"]),
      "RFOREARM": RatingClass._calculateDistance(dataPoints["RHand"], dataPoints["RELBW"]),
      }
    jointAvg = RatingClass._cumulateTimeline(jointStream, timeSequences)
    appraiseAvg = RatingClass._appraiseAvgPoints(jointAvg)
    score = RatingClass._weightedScoreMin2(appraiseAvg)
    #score = RatingClass._weightedScore(appraiseAvg)

    handCenterStream = RatingClass._calculateCenter(dataPoints["LHand"], dataPoints["RHand"])
    handCenterAvg = RatingClass._cumulateTimeline(handCenterStream, timeSequences)
    return score, jointAvg, handCenterAvg, appraiseAvg

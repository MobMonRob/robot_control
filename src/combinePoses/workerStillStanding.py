import numpy as np
#from skimage.filters import rank
#from skimage.morphology import disk
import matplotlib.pyplot as plt


def smooth(x,window_len=11,window='hanning'):
   # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #    raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[int((window_len-1)/2):0:-1],x,x[-2:int(-(window_len-1)/2-2):-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

class WorkerStillStanding:
    def __init__(self):
        pass

    def calculate(self, pose_series, stepSize):
        # stepSize = 5
        filtered = [[], [], []]
        displacementVectorNorm = [[], [], [], [], [], [], [], [], []]
        for series_idx in [0, 1, 7, 8]:
            for i in range(3):
                filtered[i] = smooth(pose_series[series_idx][i], window_len=stepSize, window='flat')
            displacementVectorNorm[series_idx] = ((filtered[0][stepSize:] - filtered[0][:-stepSize])**2 + (filtered[1][stepSize:] - filtered[1][:-stepSize])**2 + (filtered[2][stepSize:] - filtered[2][:-stepSize])**2)**0.5
            # länge wird um stepSize kürzer, d.h. der erste Wert wird für den Punkt +stepSize/2 berechnet
        return displacementVectorNorm

    def calcMin(self, displacementVectorNorm):
        displacementVectorNormMean = []
        for j in range(np.shape(displacementVectorNorm[0])[0]):
            counter = 4
            maxDiff = 0
            meanBuffer = 0
            for i in [0, 1, 7, 8]:
                if np.isnan(displacementVectorNorm[i][j]):
                    counter -= 1
                else:
                    meanBuffer += displacementVectorNorm[i][j]
                    if displacementVectorNorm[i][j] > maxDiff:
                        maxDiff = displacementVectorNorm[i][j]
            if counter > 1:
                displacementVectorNormMean.append((meanBuffer - maxDiff)/(counter-1))
            elif counter > 0:
                displacementVectorNormMean.append(meanBuffer)
            else:
                displacementVectorNormMean.append(np.NaN)
        return displacementVectorNormMean

    def findMinSection(self, displacementVectorNormMean, stepSize):
        displacementVectorNormMean = np.array(displacementVectorNormMean)
        if np.any(np.isnan(displacementVectorNormMean)):
            print("Nan-Fehler")
            return False, False, False
        else:
            print("nicht im nan fehler")
        minArray =  np.where(displacementVectorNormMean < 10, 1, 0)
    #    print("minArray: ", minArray)
        startMin = []
        endMin = []
        firstOrLast = []
        smallValue = False
        for i in range(np.shape(displacementVectorNormMean)[0]):
            if not smallValue:  #falls noch kein kleiner Bereich gestartet
                if minArray[i] == 1: #falls der Wert aber klein ist
                    startMin.append(i) #starte einen kleinen Bereich
                    smallValue = True
                    if i == 0:
                        firstOrLast.append(1)
            else:   #falls ein kleiner Bereich gestartet wurde
                if minArray[i] == 0: #aber ein nicht kleiner Wert kommt
                    endMin.append(i) #beende Bereich
                    smallValue = False
                    if len(firstOrLast) < len(startMin):
                        firstOrLast.append(0)
      #  print("Nach dem ersten for loop")
        if len(endMin) < len(startMin): #falls letzter Bereich nicht beendet wurde
            endMin.append(np.shape(displacementVectorNormMean)[0]-1)
            smallValue = False
            if len(firstOrLast) < len(startMin):
                firstOrLast.append(1)
     #   print("Nach dem ersten if")
        for i in reversed(range(len(startMin)-1)):
            if (startMin[i + 1]-endMin[i]) < 15: # 300ms
                # print("endMin[sequenceArg[0]]: ", endMin[i], " startMin[sequenceArg[1]]: ", startMin[i+1], "distance: ", startMin[i +1]-endMin[i])
                del startMin[i + 1]
                del endMin[i]
                if firstOrLast[i] == 1 or firstOrLast[i + 1] == 1:
                    firstOrLast[i] =1
                del firstOrLast[i+1]    
     #   print("startMin: ", startMin)
      #  print("endMin: ", endMin)
       # print("firstOrLast: ", firstOrLast)
        if len(startMin) < 2: #falls zu wenige Bereiche gefunden wurden.
        #    print("Only ", len(startMin), "possible rest-sections is found!")
         #   print("Worker need to stop a small time at the Get and Set Position")
            evaluateFailer = False #to see the failed runs
            if evaluateFailer:
                timeSequences = [
                    {"name": "getBox", "start": startMin[0], "end": endMin[0], "length": endMin[0]-startMin[0], "halfStepSize": int((stepSize-1)/2)},
                    {"name": "depositBox", "start": startMin[0], "end": endMin[0], "length": endMin[0]-startMin[0], "halfStepSize": int((stepSize-1)/2)},
                ]
                return True, timeSequences, minArray
            return False, False, False

        lengthMin = np.array(endMin) - np.array(startMin)
      #  print("lengthMin: ", lengthMin)
        # removing the largest element (and the first or last element) from temp list 
        new_list = np.multiply((np.ones(np.shape(firstOrLast)[0]) - np.array(firstOrLast)), lengthMin)
        argMax1 = np.argmax(new_list)
        #print("argMax1: ", argMax1)
        new_list = np.delete(new_list, np.argmax(new_list))
        #print("new_list: ", new_list)
        if np.amax(new_list) > 15: # zweit größter Wert hat mindestgröße
            #print("in if...")
            argMax2 = int(np.argwhere(lengthMin == np.amax(new_list))[0])
            #print("argMax2: ", argMax2)
            new_list = np.delete(new_list, np.argmax(new_list)) # löschen des zweiten Argmax
            #print("new_list2: ", new_list)
            if np.shape(new_list)[0] > 0 and np.amax(new_list) > 0:  # Sonderfall falls Daten schlecht aufgenommen werden, gibt es einen dritten Punkt
                #print("if dritter punkt")
                argMax3 = int(np.argwhere(lengthMin == np.amax(new_list))[0])
                #print("argMax3: ", argMax3)
                lengthMinSelection = np.array([lengthMin[argMax1],  lengthMin[argMax2], lengthMin[argMax3]])
                sequenceArg = np.array(np.array([argMax1, argMax2, argMax3]))
                #print("vor dem if")
                if np.amin(lengthMinSelection) < 0.7 * np.median(lengthMinSelection): #falls eine der drei Phasen sehr klein ist
                    #print("im if...")
                    #print("argmin: ", np.argmin(lengthMinSelection))
                    sequenceArg = np.delete(sequenceArg, np.argmin(lengthMinSelection))
                    sequenceArg = np.sort(sequenceArg)
                else: 
                    sequenceArg = np.sort(sequenceArg)
                    sequenceArg = np.delete(sequenceArg, 2) # nimm die ersten Werte - Daten werden erst aufgenommen, wenn das Paket kommt
            else:
                #print("erlse....")
                sequenceArg = np.sort(np.array([argMax1, argMax2]))

        elif lengthMin[-1] > 15 and firstOrLast[-1] == 1: # falls Ende
            sequenceArg = np.array([argMax1, np.shape(lengthMin)[0]-1]) # argMax2 = np.shape(lengthMin)[0]-1
            #print("end")
        elif lengthMin[0] > 15 and firstOrLast[0] == 1: # falls Anfang
            sequenceArg = np.array([0, argMax1]) #argMax2 = 0
            #print("start")
        else:
            #print("No timesequece")
            return False, False, False
        #print("sequenceArgOrg: ", sequenceArg)
        if sequenceArg[0] == sequenceArg[1]: # falls 2 mal die selbe Länge vor kommt
            sequenceArg[1] += np.argwhere(lengthMin[sequenceArg[1] + 1:] == lengthMin[sequenceArg[1]])[0] + 1
        timeSequences = [
            {"name": "getBox", "start": startMin[sequenceArg[0]], "end": endMin[sequenceArg[0]], "length": lengthMin[sequenceArg[0]], "halfStepSize": int(round((stepSize-1)/2))},
            {"name": "depositBox", "start": startMin[sequenceArg[1]], "end": endMin[sequenceArg[1]], "length": lengthMin[sequenceArg[1]], "halfStepSize": int(round((stepSize-1)/2))},
            ]
     #   print("endMin[sequenceArg[0]]: ", endMin[sequenceArg[0]], " startMin[sequenceArg[1]]: ", startMin[sequenceArg[1]], "distance: ", startMin[sequenceArg[1]]-endMin[sequenceArg[0]])
        #print("startMin: ", startMin)
        #print("timeSequences: ", timeSequences)

        return True, timeSequences, minArray

    def visual(self, displacementVectorNorm, displacementVectorNormMean, minArray, sequences, additionalLines=None, name=None):
        x = np.linspace(0, np.shape(displacementVectorNorm[0])[0]/96,  np.shape(displacementVectorNorm[0])[0])

        #fig, ax = plt.subplots()
        fig = plt.figure()
        w_in_inches = 5
        h_in_inches = 5
        fig.set_size_inches(w_in_inches, h_in_inches, True)
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.75])

        ax.set_ylim(-20, 100)
        line1, = ax.plot(x, displacementVectorNorm[0], '--', linewidth=2,
                        label='linke Hand')
        line2, = ax.plot(x, displacementVectorNorm[1], '--', linewidth=2,
                        label='linker Ellenbogen')
        line4, = ax.plot(x, displacementVectorNorm[8], '--', linewidth=2,
                        label='rechte Hand')
        line3, = ax.plot(x, displacementVectorNorm[7], '--', linewidth=2,
                        label='rechter Ellenbogen')
        line5, = ax.plot(x, displacementVectorNormMean, '-', linewidth=2,
                        color='k', label='Aufbereiteter Durchschnitt')
        #line6, = ax.plot(x, np.ones(np.shape(displacementVectorNorm[0])[0])*10, '-', linewidth=2,
        #                color='k', label='10-boundary')
        #line7, = ax.plot(x, minArray*30, '-', linewidth=2, color='g', label='candidates')
        line8, = ax.plot(x, (40-sequences*40)/4, '-', linewidth=3,
                color='r', label='Bewegungs- & Stillstandsphasen')
        if False: #not(additionalLines == None):
            x2 = np.linspace(0, np.shape(additionalLines["LSHOULDER2D"])[0]/96,  np.shape(additionalLines["LSHOULDER2D"])[0])
            line9, = ax.plot(x2, additionalLines["LSHOULDER2D"], '-', linewidth=2,
                color='g', label='LSHOULDER2D')
            line10, = ax.plot(x2, additionalLines["RSHOULDER2D"], '-', linewidth=2,
                color='r', label='RSHOULDER2D')
            line11, = ax.plot(x2, additionalLines["OUTOFMIDDLEDISTANCE"]/10, '-', linewidth=2,
                color='k', label='Middle Distance in cm')

        ax.set_xlabel('Zeit in Sekunden')
        ax.set_ylabel('Geschwindigkeit')
        #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=3, fancybox=True, shadow=True)
        plt.title('Geschwindigkeitsverlauf ausgewählter Marker', fontdict={'size': 16})
        ifPlot = True
        if ifPlot:
            plt.savefig('velocitiys_' + str(name) + '.png', bbox_inches='tight', dpi=300)
        else:
            plt.show()

        

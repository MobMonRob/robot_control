# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import json, os, codecs
np.set_printoptions(suppress=True, precision=3) # to avoid 1e-17 expressions in rotation matrix

def DH(conf):
    '''
    Calculates the Denavit-Hartenberg Matrix
    where
    d: offset along previous z to the common normal
    θ: angle about previous z, from old x to new x
    r: length of the common normal (aka a, but if using this notation, do not confuse with α). Assuming a revolute joint, this is the radius about previous z.
    α: angle about common normal, from old z axis to new z axis
    '''
    if len(conf) != 4:
        raise Exception('Need exactly 4 Denavit-Hartenberg parameters, you provided %i.' % len(conf))
        
    #    Z  |   X    
    Theta, d, r, alpha = conf
    
    T = np.eye(4, dtype=np.float32)

    cTheta = np.cos(Theta/180.0*np.pi)
    sTheta = np.sin(Theta/180.0*np.pi)
    calpha = np.cos(alpha/180.0*np.pi)
    salpha = np.sin(alpha/180.0*np.pi)
    
    T[np.ix_([0],[0])] = cTheta
    T[np.ix_([0],[1])] = -sTheta * calpha
    T[np.ix_([0],[2])] = sTheta * salpha
    T[np.ix_([0],[3])] = r * cTheta
    
    T[np.ix_([1],[0])] = sTheta
    T[np.ix_([1],[1])] = cTheta * calpha
    T[np.ix_([1],[2])] = -cTheta * salpha
    T[np.ix_([1],[3])] = r * sTheta

    T[np.ix_([2],[1])] = salpha
    T[np.ix_([2],[2])] = calpha
    T[np.ix_([2],[3])] = d

    return T

# [Denavit-Hartenberg](http://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters) Matrix
# 
# $$T=\left[\begin{array}{ccc|c}\cos\theta_n & -\sin\theta_n \cos\alpha_n & \sin\theta_n \sin\alpha_n & r_n \cos\theta_n \\    \sin\theta_n & \cos\theta_n \cos\alpha_n & -\cos\theta_n \sin\alpha_n & r_n \sin\theta_n \\    0 & \sin\alpha_n & \cos\alpha_n & d_n \\    \hline    0 & 0 & 0 & 1
#   \end{array}\right]=\left[\begin{array}{ccc|c}     &  &  &  \\     & R &  & T \\     & &  &  \\    \hline    0 & 0 & 0 & 1  \end{array}\right]$$
#   
# where
# 
# * $\theta\,$: angle about previous $z$, from old $x$ to new $x$
# * $d\,$: offset along previous $z$ to the common normal
# * $r\,$: length of the common normal (aka $a$, but if using this notation, do not confuse with $\alpha$).  Assuming a revolute joint, this is the radius about previous <math>z</math>.
# * $\alpha\,$: angle about common normal, from old $z$ axis to new $z$ axis
# 
# ![Image](http://upload.wikimedia.org/wikipedia/commons/3/3f/Sample_Denavit-Hartenberg_Diagram.png)
# 
# Or see a [Video on Youtube](https://www.youtube.com/watch?v=rA9tm0gTln8)

showPlot = True
testDataset = True
if testDataset:
    nxj = 10j # 11 #Shoulder2D ~> von 40° auf 80° erweitert
    nyj = 10j # 15 #Elbw ~> von 125° auf 170° erweitert
    nzj = 5j #10 #ShoulderOpen
    nuj = 5j #19 #Hand
    handDistance = [400.0, 550.0] #Handdistance: 475
    zDist = 50.0
else:
    nxj = 27j # 11 #Shoulder2D
    nyj = 18j # 15 #Elbw
    nzj = 10j #10 #ShoulderOpen
    nuj = 19j #19 #Hand
    handDistance = [450.0, 500.0] #Handdistance: 475
    zDist = 50.0
RShoulder2D, RElbw, RShoulderOpen, RHand, LShoulderOpen = np.mgrid[-10:120:nxj, 55:140:nyj, 0:45:nzj, -45:45:nuj, 0:45:nzj]
RShoulder2D = RShoulder2D.flatten()
RElbw = RElbw.flatten()
RShoulderOpen = RShoulderOpen.flatten()
RHand = RHand.flatten()
LShoulderOpen = LShoulderOpen.flatten()
LHand = np.mgrid[-45:45:nuj]

TLHand ={}
for tLShoulderOpen in np.mgrid[0:45:nzj]:
    TLHand[tLShoulderOpen] = {}
    for tLHand in np.mgrid[-45:45:nuj]:
        TLHand[tLShoulderOpen][tLHand] = DH([tLShoulderOpen-90.0, 0.0, 400.0, 90-tLHand])

print("RHand: ", np.shape(RHand))


dataStructure = []
counter = [0, 0]
for idx in range(len(RElbw)):
    if idx%1000 == 0:
        print(idx)
    #idx = random.randint(0, len(RElbw)-1)
    counter[0] += 1
    #        X,   Y,   Z, ...
    TCP = [0.0, 0.0, 0.0, 1.0] # starting from middle of rear axis of the vehicle

    #Verschibung des Torsos
    z = 2000.0
    Th = 0 #0° r in +x, 90° r in +y
    r = 2000.0
    data = {"generation": {"RShoulder2D": RShoulder2D[idx], "RSHOULDEROPEN": RShoulderOpen[idx], "RELBW": RElbw[idx], "RHAND": RHand[idx], "LSHOULDEROPEN": LShoulderOpen[idx]}, "points": {}, "pose": []}
    RSHOULDER2D = RShoulder2D[idx] #0° senkrecht nach unten, positiv nach vorne # :0:10:20:30: 
    RSHOULDEROPEN = RShoulderOpen[idx] #0° am Körper, 90° Horziontal #0:45
    RELBW = RElbw[idx] #90° rechtwinklig, 180° ausgestreckt # :65:75:85:95:105:115:
    RHAND = RHand[idx] #0° senkrecht nach vorne, positiv nach außen

    LSHOULDER2D = RShoulder2D[idx] #0° senkrecht nach unten, positiv nach vorne
    LSHOULDEROPEN = LShoulderOpen[idx] #0° am Körper, 90° Horziontal
    LELBW = RElbw[idx] #90° rechtwinklig, 180° ausgestreckt
    LHAND = 0 #0° senkrecht nach vorne, positiv nach außen

            #forearm: 225
            #upperarm: 400
            #shoulderdistance: 286
            #Handdistance: 475
    #                                          Z    |     X
    #                                        Th,   d,    r,    al
    konfiguration = OrderedDict([
                    ('TORSO', [Th, z, r, 00]),
                    ('TORSODOWN', [90.0-Th, -100.0, 0.0, 0.0]),
                    ('RSHOULDER2D', [90.0-Th, 0.0, 143.0, RSHOULDER2D + 90]),
                    ('RELBW', [RSHOULDEROPEN-90.0, 0.0, 400.0, -90+RHAND]),
                    ('RHand', [180.0-RELBW, 0.0, 225.0, 0.0]),
                    ('LSHOULDER2D', [-90.0-Th, 0.0, 143.0, -LSHOULDER2D + 90]),
                    ('LELBW', [LSHOULDEROPEN-90.0, 0.0, 400.0, 90-LHAND]),
                    ('LHand', [180.0-LELBW, 0.0, 225.0, 0.0])
                    ])

    T = {}
    for name, conf in konfiguration.items():
        T[name] = DH(conf)
    if showPlot:
        plt.close('all')
        fig = plt.figure()
        ax = Axes3D(fig)

    arms = [["RSHOULDER2D", "RELBW", "RHand"], ["LSHOULDER2D", "LELBW", "LHand"], ["TORSODOWN"]]
    color = ['r', 'g', 'k']
    validConfig = False
    for i in range(len(arms)):
        M = np.eye(4, dtype=np.float32)
        M = np.dot(M, T["TORSO"])
        oldp = np.dot(M,TCP)
        if showPlot:
            ax.scatter(oldp[0], oldp[1], oldp[2], s=70, c='k')
        for j in range(len(arms[i])):
            if i == 1 and j == 1:
                for tLHand in np.mgrid[-45:45:nuj]:
                    MBuffer1 = np.dot(M, TLHand[LSHOULDEROPEN][tLHand])
                    MBuffer2 = np.dot(MBuffer1, T[arms[1][2]])
                    dist = np.linalg.norm(np.array(data["points"]["RHand"])-np.dot(MBuffer2,TCP)[:3])
                    distZ = np.abs(np.array(data["points"]["RHand"][2])-np.dot(MBuffer2,TCP)[2])
                    #print("tLHand: ", tLHand, " dist: ", dist)
                    if dist > handDistance[0] and dist < handDistance[1] and distZ < zDist:
                        M = MBuffer1
                        data["generation"]["LHAND"] = tLHand
                        validConfig = True
                        break
            else:
                M = np.dot(M, T[arms[i][j]])

            p = np.dot(M,TCP)
            data["points"][arms[i][j]] = list(p[:3])

            if showPlot:
                x = float(p[0])
                y = float(p[1])
                z = float(p[2])
                ax.plot([oldp[0], x], [oldp[1], y], [oldp[2], z], c='k', alpha=.5)
                ax.scatter(x, y, z, s=50, c=color[i])
            
            oldp = p
    if validConfig == True:
        handCenter = np.zeros(6)
        handCenter[:3] = (np.array(data["points"]["RHand"])+np.array(data["points"]["LHand"]))/2
        data["pose"] = list(handCenter/1000)
        counter[1] += 1
        if testDataset:
            print("HandDist: ", np.linalg.norm(np.array(data["points"]["RHand"])-np.array(data["points"]["LHand"])))
        dataStructure.append(data)
        if showPlot:
            ax.scatter(handCenter[0], handCenter[1], handCenter[2], s=70, c='b')
            #plt.legend(bbox_to_anchor=(0, 0))
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            ax.view_init(0, -135)

            plt.show(block=False)
            plt.pause(2)
            plt.close()
    else:
        if testDataset:
            print("Configuration isn't valid")

relativePath = "forward_data_20200120_5steps_120_140_test.json"
input_dir = 'C:\\Users\\gruseck\\Data\\70_Master_Thesis_C\\DATA_20191219'
absFilePath = os.path.join(input_dir, relativePath)
json.dump(dataStructure, codecs.open(absFilePath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
print("counter: ", counter)


#TODO: nxj korrigieren; 10j bedeutet 10 abschnitte und nicht abschnitt = 10...
#TODO: schulterWinkel auf Vertikale beziehen und nicht mehr auf den Torso...
#TODO: immer best bewertesten value in einem bereich annehmen ~> der Mensch sucht sich eine sinnvolle Konfiguration :)

#TODO: testen mit anderer Dimensionalität
#TODO: Gütemaß trennen? Überarbeiten? min anstelle von Multiplikation? aktuell können gute Gelenkwinkel schlechte Mittellage überdecken...

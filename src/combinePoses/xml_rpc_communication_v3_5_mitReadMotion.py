  #!/usr/bin/ python3
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import numpy as np
import time
import json
import os
import threading
import random 
import rospy
from robot_control.srv import nextcycle
#from robotCommand.srv import nextCycle
#from robotCommand import nextCycle
import std_msgs.msg 
import csv
import readMotionCaptureData

global asked
global ready

class Box:
    def __init__(self, minimumPose, maximumPose, sampling, removeElements):
        self.list = self.calcPoseList(minimumPose, maximumPose, sampling)
        self.correctedList = self.list.copy()
        removeElements = list(np.sort(np.array(removeElements)))
        # print("removeElements: ", removeElements)
        for idx in reversed(list(removeElements)):
            # print("del: ", idx, " self.correctedList[idx]: ", self.correctedList[idx])
            del self.correctedList[idx]
        self.index = 0
        self.randomIndex = []
        random.seed(27) # fixed order of the random numbers
        while len(self.randomIndex) < len(self.correctedList):
            newInt = random.randint(0, len(self.correctedList)-1)
            if not( newInt in self.randomIndex):
                self.randomIndex.append(newInt)
        # print("List of random index: ", self.randomIndex)
        print("Length of the unedited list: ", len(self.list))
        print("Corrected list: ", len(self.correctedList))
        #for el, liste in enumerate(self.correctedList):
        #    print("el: ", el, "pose: ", liste)

    def calcPoseList(self, minimumPose, maximumPose, sampling):
        minimumPose = np.array(minimumPose)
        maximumPose = np.array(maximumPose)
        nj = np.around((maximumPose - minimumPose)/sampling +1) * 1j
        grid_x, grid_y, grid_z = np.mgrid[minimumPose[0]:maximumPose[0]:nj[0], minimumPose[1]:maximumPose[1]:nj[1], minimumPose[2]:maximumPose[2]:nj[2] ]
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        grid_z = grid_z.flatten()
        grid_zeros = np.zeros(np.shape(grid_z)[0])
        poseList = np.around(np.transpose(np.array([grid_x, grid_y, grid_z, grid_zeros, grid_zeros, grid_zeros])), 6)
        return poseList.tolist()


class Pose:
    def __init__(self):
        self.calibrationIndex = 0
        self.box = []
        removeListBig = [359, 438, 446, 447, 448, 451, 457, 517, 521, 522, 523, 524, 525, 526, 527, 531, 532, 533, 534, 535, 536, 537]
        for i in range(5): # vertikale ebene 1 von links nach rechts (y-z)
            removeListBig += range(0+i, 90, 10) # erste 5 Zeilen
        for i in range(4): # ebene 2
            removeListBig += range(90+i, 180, 10) # erste 4 Zeilen
        for i in range(3): # ebene 3
            removeListBig += range(180+i, 270, 10) # erste 3 Zeilen
        for i in range(2): # ebene 4
            removeListBig += range(270+i, 360, 10) # erste 2 Zeilen    
        for i in [0, 9]: # ebene 5
            removeListBig += range(360+i, 450, 10) # zehnte Zeilen
        for i in [0, 8, 9]: # ebene 6
            removeListBig += range(450+i, 540, 10) # zehnte Zeilen
        self.box.append( Box([-0.2, -0.4, -0.4], [0.3, 0.4, 0.5], [0.1, 0.1, 0.1], removeListBig)) # poses of the big box

        removeListSmall = [54, 63, 144, 150, 151, 152, 161,  225, 231, 232, 233, 315, 322, 323, 331, 332, 340, 341, 349, 350, 358, 359]#[8, 80]# jetzt dritte Ebene #80 so semi nicht...
        for i in [0, 8, 17, 72, 216, 288, 297, 306]: # erste drei Spalten erste Ebene, 1. Spalte 4. Ebene, erste drei Spalten 5. Ebene
                    removeListSmall += range(0+i, 9+i, 1) # 

                #72 pr Ebene

        self.box.append( Box([-0.2, -0.3, -0.4], [0.2, 0.4, 0.4], [0.1, 0.1, 0.1], removeListSmall)) # poses of the small box
        self.pause = False

def robot_nextcycle_server():
	rospy.init_node('robot_nextcycle',log_level=rospy.DEBUG)
	service_nextcycle = rospy.Service('nextCycle', nextcycle, handle_nextcycle)
	cycle_str = "ROS: nextCycle-Server started"
	rospy.loginfo(cycle_str)
	rospy.loginfo("ROS: Entering spin")
	rospy.spin()
    
#def robot_calibration_server():
#    service_calibration = rospy.Service('robotCalibration', roborCalibration, handle_calibration)
#    rospy.loginfo('ROS: robotCalibration-Server started')
#    rospy.spin()
        
def handle_nextcycle(req):
    print(req)

    global ready
    global asked 
	
    asked = 1
    header = std_msgs.msg.Header()
    if ready == 0:
        
        
        cycleinfo = "WAITING"
        time.sleep (1)
        
        cycle_nr=0
    
        pose_x=0
        pose_y=0
        pose_z=0
    
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = 'none'
        print(cycleinfo)
        return header, cycleinfo, pose_x, pose_y, pose_z, cycle_nr
    
    else:
        
        cycleinfo = "REACHED"
        
        cycle_nr =0  #index
    
        pose_x=0 #poseList[0]
        pose_y=0 #poseList[1]
        pose_z=0 #poseList[2]
    
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = 'none'
        print(cycleinfo)
        asked = 0
	#ready = 0
        return header, cycleinfo, pose_x, pose_y, pose_z, cycle_nr
    
    
#def handle_calibration():
 #   asked = 1
  #  nextCalibrationPose()
   # 
    #if calibtraion = True
    #
     #   calibration_status = "DONE"
    #else:1
    #    calibration_status = "FAILED"
        
    #return calibration_status    


def runserver(): 
    
		global asked
		global ready
		global programIsFinished
		global newGeneratedPose
		global simulationRaphael

		simulationRaphael = True
		newGeneratedPose = [0, 0, 0, 0, 0, 0]
		programIsFinished = True
    
		ready = 0
		asked = 0
		scriptDir = '/home/oliver/DATA/session_20200214'
		relativePath = "log_data.log"
		absFilePath = os.path.join(scriptDir, relativePath)
		f = open(absFilePath, "a")
    
		pose = Pose()
		dataProcessing = readMotionCaptureData.DataProcessing(testMode=False, poses=pose)

		time.sleep(2)
     
		ipAdress = '192.168.12.199' # '192.168.12.199'
    # Set the IP Adress to the server IP-address. 
    # Also change the adress in the robot if it differs of '192.168.12.10'
		#handler = SimpleXMLRPCRequestHandler()
		with SimpleXMLRPCServer((ipAdress,8000)) as server:
			server.register_introspection_functions()
        

			def poseDict(poseList):
				return {
					'x' : poseList[0],
                	'y' : poseList[1],
                	'z' : poseList[2],
                	'rx' : poseList[3],
                	'ry' : poseList[4],
                	'rz' : poseList[5]
                	}
			class ServerFunctions:
				def nextRelativePose(self, boxSize):
                # returns next relative Pose [x, y, z, rx, ry, rz]
                # the point of reference is defined in the RPS.
                # It is a neutral Pose in front of the worker
                # x, y, z in meter, at the moment there are unexpected directions ##x is in direction to the worker, and z is up
                # rx, ry, rz in radian they should be zero
					#boxSize = 0
					global programIsFinished
					global simulationRaphael
					global newGeneratedPose

					pose.pause = True #Auf Programmablauf achten!!!

					index = pose.box[boxSize].index
					randomIndex = pose.box[boxSize].randomIndex[index]

					if simulationRaphael:
						poseList = newGeneratedPose #funktioniert nur falls zuvor nextCycleXML ausgeführt wurde
					else:
						poseList = pose.box[boxSize].correctedList[index]#randomIndex]						
					print("boxSize=",boxSize)
					

					if index + 1 < len(pose.box[boxSize].correctedList):
						pose.box[boxSize].index = index + 1
					else:
						pose.box[boxSize].index = 0
						print('all runs are done')
					print("index: ", index, " Poseindex: ", randomIndex, " boxSize: ", boxSize, "poseList: ", poseList)
					#scriptDir = '/home/oliver/DATA/session_20200214' 
					#relativePath = "log_data.txt"
					#absFilePath = os.path.join(scriptDir, relativePath)
					#f = open(absFilePath, "a")
					f.write(json.dumps({"timestamp": int(round(time.time()*1000)), "pose": poseList, "boxSize": boxSize}) + ",\n")
					programIsFinished = False
					return poseDict(poseList)
				
				def nextCalibrationPose(self):
					calibration = False

					if calibration:
						calibrationList = [
                    	    [0.00, 0.00, 0.00, 0, 0, 0],
                    	    [0.2, 0.00, 0.00, 0, 0, 0],
                    	    [0.00, 0.2, 0.00, 0, 0, 0],
                    	    [0.00, 0.00, 0.2, 0, 0, 0]]
					else: #test poses
						boxSize = 1
						calibrationList  = pose.box[boxSize].list
                    
					index = pose.calibrationIndex
					poseList =calibrationList[index]
					print("index: ", index, " poseList: ", poseList)
					if index + 1 < len(calibrationList):
						pose.calibrationIndex = index + 1
					else:
						pose.calibrationIndex = 0
						print('Calibration is done')
					
					return poseDict(poseList)
				
				def getBoxSize(self):
                	# 0: big box
                	# 1: small box
					return 0
			
				def setIndex(self, boxSize, index):
					pose.box[boxSize].index = index
					print("The index of boxSize: ", boxSize, " is now: ", pose.box[boxSize].index)
					return pose.box[boxSize].index
				
				def setCalibrationIndex(self, index):
					pose.calibrationIndex = index
					print("The calbrationindex is now: ", pose.calibrationIndex)
					return pose.calibrationIndex
				
				def getPause(self):
					return pose.pause
				
				def setPause(self, value):
					pose.pause = value
					print("System pause is now: ", pose.pause)
					return pose.pause
				
				def togglePause(self):
					pose.pause = not pose.pause
					print("System pause is now: ", pose.pause)
					return pose.pause
				
				def setFinished(self):
					global programIsFinished
					programIsFinished = True
					return True 

				def setFinishedFalse(self):
					global programIsFinished
					programIsFinished = False
					return False 


				def getFinished(self): #nicht bentutzt
					global programIsFinished
					return programIsFinished

				def startAfterPause(self):
					pose.pause = False
					return True

				def nextCycleXML(self):
					global simulationRaphael
					global newGeneratedPose

					if programIsFinished == False:
						return [False]
					else:
						boxSize = self.getBoxSize()
						index = pose.box[boxSize].index
						randomIndex = pose.box[boxSize].randomIndex[index]
						if simulationRaphael:
							newGeneratedPose = dataProcessing.getNextPose().tolist()
						else:
							newGeneratedPose = pose.box[boxSize].correctedList[index]#randomIndex]
						print("newGeneratedPose: ", newGeneratedPose)

						return [True, newGeneratedPose[0], newGeneratedPose[1], newGeneratedPose[2], index + 1]


					
				
			server.register_instance(ServerFunctions())
			print('runserver started')

			# TODO ROS-Teil
			#while True:

			#while  asked == 0:
					#time.sleep(1) #wird rausgeworfen wenn alles funktioniert
					#print('waiting....')
					#ready = 1
					#pose.pause = False
					#rospy.wait_for_service('nextCycle')
					#print('still waiting....')
					#print("asked=", asked)
						# rospy.wait_for_service('robotCalibration')
						# cycleinfo = rospy.ServiceProxy('nextCycle', robot_nextcycle)
						# calibrationinfo = rospy.ServiceProxy('robotCalibration', robot_nextcycle)
				
				
			
			#print('while finished')
			#asked = 0
			#ready = 0  
			
				# Run the server's main loop
			print('server is running')
			try:
					server.serve_forever()
			except KeyboardInterrupt:
					print("\nKeyboard interrupt received, exiting.")
					server.server_close()
			server.serve_forever()
			print('server is not running')
			
		

if __name__ == "__main__":
	
	ready = 0
	asked = 0
	runserver_thread=threading.Thread(target=runserver)
	runserver_thread.start()
	rospy.loginfo("XMLRPC Server for communication of poses to the UR is running...")
	robot_nextcycle_server()

	rospy.loginfo("ROS-Service nextCycle has ended")
	
	


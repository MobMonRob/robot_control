#!/usr/bin/env python3
import xmlrpc.client
import time
import threading
import rospy
from robot_control.srv import nextcycle
from UR_control.srv import robotCommand
import sys
import std_msgs.msg 



def robot_nextcycle_server():
	rospy.init_node('robot_nextcycle',log_level=rospy.DEBUG)
	service_nextcycle = rospy.Service('nextCycle', nextcycle, handle_nextcycle)
	cycle_str = "ROS: nextCycle-Server started"
	rospy.loginfo(cycle_str)
	rospy.spin()

def handle_nextcycle(req):
    print(req)
    is_running_flag = True

    url = 'http://192.168.12.199:8000' # 'http://192.168.12.10:8000'
    s = xmlrpc.client.ServerProxy(url, encoding="UTF-8")
    s.setAsked(True)
    print("set asked to: True")
    cycle_nr = s.getCycle()
    header = std_msgs.msg.Header()
    print("ready state= ",s.getReady())
    if s.getReady() == False:
        cycleinfo = "WAITING"
        time.sleep (1)

        pose_x=0
        pose_y=0
        pose_z=0
    
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = 'none'
        print(cycleinfo)
        return header, cycleinfo, cycle_nr, pose_x, pose_y, pose_z
    
    else:
        
        cycleinfo = "REACHED"
        s.setReady(False)
        print("ready state in else: ",s.getReady())

        if False:
            cycle_nr = cycle_nr+1
    
            pose_x=0 #poseList[0]
            pose_y=0 #poseList[1]
            pose_z=0 #poseList[2]
        else:
            print("else")
           # try:
            result = s.nextCycleXML()
           # except:
          #      print("results nicht vollständig")
         #       result = [0,0,0,0]
            cycle_nr = result[4] #+50 #für Raphael
            pose_x = result[1]
            pose_y = result[2]
            pose_z = result[3]
        #while is_running_flag:
            #print("last while loop")
            #rospy.wait_for_service('robot_command_server')
            #robot_status = rospy.ServiceProxy('robot_command_server', robotCommand)
            #print("between")
            #is_running = robot_status(header, 5) #2nodes??
            #print("Client started")
            #if is_running == "Program running: False":
                #is_running_flag = False
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = 'none'
        print(cycleinfo)

        time.sleep(0.5)
        print("send REACHED at: ",time.time())
        return header, cycleinfo, cycle_nr, pose_x, pose_y, pose_z

def waiting_to_get_ready():
	url = 'http://192.168.12.199:8000' # 'http://192.168.12.10:8000'
	s = xmlrpc.client.ServerProxy(url, encoding="UTF-8")
	while True:
		asked = s.getAsked()
		while  asked == False:
			time.sleep(0.5) 
			rospy.wait_for_service('nextCycle')
			asked = s.getAsked()

		print("Left waiting-loop")
		asked = s.setAsked(False)

if __name__ == "__main__":
	try:
			waiting_for_server_thread=threading.Thread(target=waiting_to_get_ready)
			waiting_for_server_thread.daemon = True
			waiting_for_server_thread.start()
			robot_nextcycle_server()
			sys.exit()		
	except (KeyboardInterrupt, SystemExit):
			print("\nKeyboard interrupt received, ending.") 

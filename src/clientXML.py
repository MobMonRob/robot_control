
import xmlrpc.client
import rospy
import std_msgs.msg
from robot_control.srv import nextcycle


url = 'http://192.168.12.199:8000' # 'http://192.168.12.10:8000'
s = xmlrpc.client.ServerProxy(url, encoding="UTF-8")
boxSize = 0 # 0: big box, 1: small box
newIndex = 0
header = std_msgs.msg.Header()
header.seq = 0
#header.stamp = rospy.Time.now()
header.frame_id = 'none'
service_nextcycle = rospy.ServiceProxy('robot_nextcycle', nextcycle)
print("service call successful")
service_nextcycle(header, 2)
print(s.getBoxSize())  # Returns 0 or 1

#print("The new index is: ", s.setIndex(boxSize, newIndex)) # Sets the index for the next pose
print(s.nextRelativePose(boxSize))  # Returns the next pose

value = True #pause flag: True = pause, False = play
#print("System pause is now: ", s.getPause()) #get State of the pause flag
#print("System pause is now: ", s.setPause(value)) #sets the State of the pause flag
#print("System pause is now: ", s.togglePause()) #toggles the State of the pause flag

#print("The new calibrationindex is: ", s.setCalibrationIndex(newIndex)) # Sets the index for the next calibrationPose
#print(s.nextCalibrationPose()) # Returns the next calibration pose

# Print list of available methods
#print(s.system.listMethods())
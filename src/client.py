 #!/usr/bin/env python

import sys
import rospy
from robot_control.srv import nextcycle
import std_msgs.msg     


def add_two_ints_client(x, y):
    rospy.wait_for_service('nextCycle')
    
    header = std_msgs.msg.Header()
    add_two_ints = rospy.ServiceProxy('nextCycle', nextcycle)
    resp1 = add_two_ints(header, 2)
    print (header)
    
    
    #print ("Service call failed: ")
    return header
 
def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
	x=1 
	y=1
	print ("Requesting %s+%s"%(x, y))
	print ("%s + %s = %s"%(x, y, add_two_ints_client(x, y)))

#include "robot_control/ServerNode.h"
#include "ros/console.h"
#include "std_msg/Header.h"

std::string ServerNode::program_name = "";

std::unique_ptr<RobotConnection> ServerNode::robotConnection = nullptr;

ServerNode::ServerNode(int argc, char **argv)
{
	ros::init(argc, argv, "robot_nextcycle_server");
	node = std::make_unique<ros::NodeHandle>();
    if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug) )
    ros::console::notifyLoggerLevelsChanged();


void ServerNode::start()
{
	ros::ServiceServer server = node->advertiseService("robot_nextcycle", nextcycle);
	ROS_INFO("robot_nextcycle service activated.");


	ros::spin();
}

bool ServerNode::robotCommand(robot_control::nextCycleRequest &request, robot_control::nextCycleResponse &response)
{
	std::string cycleinfo;
    float pose_x,
    float pose_y,
    float pose_z;
    ros::Header::Header():


        cycleinfo = "WAITING";
       

        pose_x=0;
        pose_y=0;
        pose_z=0;
    
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = 'none'

    return header, cycleinfo, cycle_nr, pose_x, pose_y, pose_z
	
	response.response = robotConnection->receive();
	ROS_INFO("Robot response:\n%s", response.response.c_str());

	return true;
}


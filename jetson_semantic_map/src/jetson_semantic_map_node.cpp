#include <ros/ros.h>
#include <jetson_cloud_fusion.h>

int main(int argc, char **argv)
{

    ros::init(argc, argv, PROJECT_NAME);
    ros::NodeHandle nh("~");


    ROS_INFO_STREAM("Creating semantic map.");
    CloudColoring cloud_coloring(nh);

    ROS_INFO_STREAM("Started semantic mapping.");
    ros::spin();
    return 0;
}

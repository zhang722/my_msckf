#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <my_msckf/FrameMsg.h>
#include "feature.h"

namespace my_msckf {

class Vio {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Vio();
	~Vio() {}

private:
	void imu_callback(const sensor_msgs::ImuConstPtr & msg);
	void frame_callback(const FrameMsgConstPtr & msg);
	void feature_add_ob(const FrameMsgConstPtr & msg);

	bool is_first_img;

	StateServer state_server;
	MapServer   map_server;

	// ROS
	ros::NodeHandle nh_;
	ros::Subscriber imu_sub;
	ros::Subscriber feature_sub;
};
	


} //namespace my_msckf


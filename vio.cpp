#include "vio.h"

namespace my_msckf {

Vio::Vio() 
{
	imu_sub = nh_.subscribe("/imu0", 100,
		&Vio::imu_callback, this);
	feature_sub = nh_.subscribe("/features", 40,
		&Vio::frame_callback, this);
	
	is_first_img = true;
}

void 
Vio::imu_callback(const sensor_msgs::ImuConstPtr & msg)
{
	return;
}

void Vio::frame_callback(const FrameMsgConstPtr & msg)
{
	if (is_first_img) {
		is_first_img = false;
	}	

	// state_server.imu_state.id = msg->id;
	// state_server.cam_states.id = msg->id;

	feature_add_ob(msg);
}

void Vio::feature_add_ob(const FrameMsgConstPtr & msg)
{
	StateIdType state_id = state_server.imu_state.id;

	for (const auto & f:msg->features) {
		if (map_server.find(f.id) == map_server.end()) {
			// New feature
			map_server[f.id] = Feature();
			map_server[f.id].id = f.id;
			map_server[f.id].obserbations[state_id] =
				Eigen::Vector2d(f.u0, f.v0);
		} else {
			// Old feature
			map_server[f.id].obserbations[state_id] =
				Eigen::Vector2d(f.u0, f.v0);
		}
	}
	ROS_INFO("map size: %d", static_cast<int>(map_server.size()));
}


}

//Test
int main(int argc, char **argv)
{
  ros::init(argc, argv, "my_msckf");  

  my_msckf::Vio vio;
  std::cout << "vio process" << std::endl;

  ros::spin();

  return 0;
}

#pragma once

#include <map>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

namespace my_msckf {

using StateIdType = unsigned long long;

struct IMUState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		
  StateIdType id;
	double      time;

	// IMU state
	Eigen::Vector3d p;
	Eigen::Vector3d v;
	Eigen::Vector4d q;

	Eigen::Vector3d bg;
  Eigen::Vector3d ba;

	Eigen::Matrix3d R_imu_cam0;
	Eigen::Vector3d t_cam0_imu;

  // Process noise
	constexpr static double gyro_noise = 0.001;
	constexpr static double acc_noise = 0.01;
	constexpr static double gyro_bias_noise = 0.001;
	constexpr static double acc_bias_noise = 0.01;
};


struct CamState {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	StateIdType id;
	double      time;

	// Cam state
	Eigen::Vector3d p;
	Eigen::Vector4d q;
};

using CamStateServer = std::map<StateIdType, CamState,
	std::less<StateIdType>,
	Eigen::aligned_allocator<
	std::pair<const StateIdType, CamState>>>;

struct StateServer {
  IMUState       imu_state;
  CamStateServer cam_states;

	// State covariance matrix
	Eigen::MatrixXd state_cov;
	Eigen::Matrix<double, 12, 12> continuous_noise_cov;
};

} // namespace my_msckf



#include <eigen_conversions/eigen_msg.h>
#include "vio.h"



namespace my_msckf {
constexpr double IMUState::gyro_noise;
constexpr double IMUState::gyro_bias_noise;
constexpr double IMUState::acc_noise;
constexpr double IMUState::acc_bias_noise;

constexpr double StateServer::gyro_bias_cov;
constexpr double StateServer::acc_bias_cov;
constexpr double StateServer::extrinsic_rotation_cov;
constexpr double StateServer::extrinsic_translation_cov;

Vio::Vio() 
{
	imu_sub = nh_.subscribe("/imu0", 100,
		&Vio::imu_callback, this);
	feature_sub = nh_.subscribe("/features", 40,
		&Vio::frame_callback, this);
	
	is_first_img = true;
	is_gravity_set = false;

	state.continuous_noise_cov = 
		Eigen::Matrix<double, 12, 12>::Zero();
	state.continuous_noise_cov.block<3, 3>(0, 0) = 
		Eigen::Matrix3d::Identity() * state.imu.gyro_noise;
	state.continuous_noise_cov.block<3, 3>(3, 3) = 
		Eigen::Matrix3d::Identity() * state.imu.gyro_bias_noise;
	state.continuous_noise_cov.block<3, 3>(6, 6) = 
		Eigen::Matrix3d::Identity() * state.imu.acc_noise;
	state.continuous_noise_cov.block<3, 3>(9, 9) = 
		Eigen::Matrix3d::Identity() * state.imu.acc_bias_noise;

  state.state_cov = Eigen::MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state.state_cov(i, i) = state.gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state.state_cov(i, i) = state.velocity_cov;
  for (int i = 9; i < 12; ++i)
    state.state_cov(i, i) = state.acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state.state_cov(i, i) = state.extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state.state_cov(i, i) = state.extrinsic_translation_cov;
}

void Vio::imu_callback(const sensor_msgs::ImuConstPtr & msg)
{
	imu_msg_buffer.push_back(*msg);
	if (!is_gravity_set) {
		if (imu_msg_buffer.size() < 200) return;
		init_g_and_bg(imu_msg_buffer);		
		is_gravity_set = true;
	}
	return;
}

void Vio::frame_callback(const FrameMsgConstPtr & msg)
{
	if (!is_gravity_set) return;

	double frame_time = msg->header.stamp.toSec();
	StateIdType frame_id = msg->id;
	if (is_first_img) {
		state.imu.time = frame_time;
		is_first_img = false;
	}	
	
	batch_imu_propagate(frame_id, frame_time, imu_msg_buffer);
	ROS_INFO("imu id:%d", static_cast<int>(state.imu.id));
	ROS_INFO("frame id:%d", static_cast<int>(frame_id));

	state_augment(frame_time, state);

	feature_add_ob(msg);

	remove_lost_features(map);
}




void Vio::batch_imu_propagate(
	const StateIdType & frame_id,
	const double & frame_time,
	std::vector<sensor_msgs::Imu> & imu_msg_buffer)
{
	auto begin_iter = imu_msg_buffer.begin();
	while (begin_iter != imu_msg_buffer.end() &&
		begin_iter->header.stamp.toSec() < state.imu.time) {
		++ begin_iter;	
	}

	auto end_iter = begin_iter;
	while (end_iter != imu_msg_buffer.end() &&
		end_iter->header.stamp.toSec() < frame_time) {
		imu_propagate(*end_iter);
		++ end_iter;
	}
	//ROS_INFO("begin, end:%d,%d",std::distance(imu_msg_buffer.begin(), begin_iter), std::distance(imu_msg_buffer.begin(), end_iter));

	state.imu.id = frame_id;
	imu_msg_buffer.erase(imu_msg_buffer.begin(), end_iter);
}


void Vio::imu_propagate(const sensor_msgs::Imu & msg) 
{
	double imu_time = msg.header.stamp.toSec();
	double dt       = imu_time - state.imu.time;

	Eigen::Vector3d m_gyro, m_acc;
	tf::vectorMsgToEigen(msg.angular_velocity, m_gyro);
	tf::vectorMsgToEigen(msg.linear_acceleration, m_acc);

	// Remove bias
	Eigen::Vector3d gyro = m_gyro - state.imu.bg;
	Eigen::Vector3d acc  = m_acc - state.imu.ba;

	Eigen::Matrix<double, 21, 21> F = Eigen::Matrix<double, 21, 21>::Zero();
	Eigen::Matrix<double, 21, 12> G = Eigen::Matrix<double, 21, 12>::Zero();

	// Continuous F
  F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  F.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
  F.block<3, 3>(6, 0) = -quaternionToRotation(
			state.imu.q).transpose() * skewSymmetric(acc);
  F.block<3, 3>(6, 9) = -quaternionToRotation(state.imu.q).transpose();
  F.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();

	// Continuous G
  G.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
  G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  G.block<3, 3>(6, 6) = -quaternionToRotation(state.imu.q).transpose();
  G.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

	// Phi
	Eigen::Matrix<double, 21, 21> Fdt = F * dt;
	Eigen::Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
	Eigen::Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
	Eigen::Matrix<double, 21, 21> Phi = Eigen::Matrix<double, 21, 21>::Identity() + Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

	// Propogate PII_k+1
	state.state_cov.block<21, 21>(0, 0) = 
		Phi * state.state_cov.block<21, 21>(0, 0) * Phi.transpose()
	+ Phi * G * state.continuous_noise_cov * G.transpose() * Phi.transpose() * dt;

	// Propogate PIC_k+1 and (PIC_k+1)'
	if (state.cam.size() > 0) {
    state.state_cov.block(0, 21, 21, state.state_cov.cols()-21)
		= Phi * state.state_cov.block(0, 21, 21, state.state_cov.cols()-21);
    state.state_cov.block(21, 0, state.state_cov.rows()-21, 21)
		= state.state_cov.block(21, 0, state.state_cov.rows()-21, 21) * Phi.transpose();
	}

	// Propogate state vector
	imu_integrate(dt, gyro, acc, state);
	state.imu.time = imu_time;
}


void Vio::imu_integrate(
	const double & dt,
	const Eigen::Vector3d & gyro,
	const Eigen::Vector3d & acc,
	StateServer & state)
{
	Eigen::Matrix4d omega = Eigen::Matrix4d::Zero();
	omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
	omega.block<3, 1>(0, 3) = gyro;
	omega.block<1, 3>(3, 0) = -gyro;

	double gyro_norm = gyro.norm();
	// q_k+1(q_dt is q_k+1)
	Eigen::Vector4d q_dt, q_dt2;
	if (gyro_norm > 1e-5) {
		q_dt = (cos(gyro_norm*dt*0.5)*Eigen::Matrix4d::Identity() +
			1/gyro_norm*sin(gyro_norm*dt*0.5)*omega) * state.imu.q;
		q_dt2 = (cos(gyro_norm*dt*0.25)*Eigen::Matrix4d::Identity() +
			1/gyro_norm*sin(gyro_norm*dt*0.25)*omega) * state.imu.q;
	} else {
		q_dt = (Eigen::Matrix4d::Identity()+0.5*dt*omega) *
			cos(gyro_norm*dt*0.5) * state.imu.q;
		q_dt2 = (Eigen::Matrix4d::Identity()+0.25*dt*omega) *
			cos(gyro_norm*dt*0.25) * state.imu.q;
	}


	Eigen::Matrix3d R_dt_t = quaternionToRotation(q_dt).transpose();
	Eigen::Matrix3d R_dt2_t = quaternionToRotation(q_dt2).transpose();
	Eigen::Vector3d v = state.imu.v;


	// q_k+1
	state.imu.q = q_dt;

	// v_k+1
	Eigen::Vector3d k1_v = quaternionToRotation(state.imu.q).transpose() * acc + G;
	Eigen::Vector3d k2_v = R_dt2_t * acc + G;
	Eigen::Vector3d k3_v = R_dt2_t * acc + G;
	Eigen::Vector3d k4_v = R_dt_t * acc + G;
	state.imu.v += dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v);	

	// p_k+1
	Eigen::Vector3d k1_p = v;
	Eigen::Vector3d k2_p = v + k1_v * dt / 2;
	Eigen::Vector3d k3_p = v + k2_v * dt / 2;
	Eigen::Vector3d k4_p = v + k3_v * dt;
	state.imu.p += dt/6 * (k1_p + 2*k2_p + 2*k3_p + k4_p);	

	return;
}


void Vio::init_g_and_bg(
		const std::vector<sensor_msgs::Imu> & imu_msg_buffer)
{
	Eigen::Vector3d sum_angular_vel = Eigen::Vector3d::Zero();
	Eigen::Vector3d sum_linear_acc  = Eigen::Vector3d::Zero();
	int num = imu_msg_buffer.size();

	for (const auto & msg : imu_msg_buffer) {
		Eigen::Vector3d angular_vel{0, 0, 0};
		Eigen::Vector3d linear_acc{0, 0, 0};

		tf::vectorMsgToEigen(msg.angular_velocity, angular_vel);
		tf::vectorMsgToEigen(msg.linear_acceleration, linear_acc);

		sum_angular_vel += angular_vel;
		sum_linear_acc  += linear_acc;
	}
	Eigen::Vector3d g_imu = sum_linear_acc / num;
	state.imu.bg = sum_angular_vel / num;

	G = Eigen::Vector3d(0, 0, -g_imu.norm());
	Eigen::Quaterniond q0_i_w = Eigen::Quaterniond::FromTwoVectors(
    g_imu, -G);

  state.imu.q =
    rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());
}



void Vio::state_augment(
		const double & frame_time,
		StateServer & state)
{
  const Eigen::Matrix3d& R_i_c = state.imu.R_imu_cam0;
  const Eigen::Vector3d& t_c_i = state.imu.t_cam0_imu;

  // Add a new camera state to the state server.
	Eigen::Matrix3d R_w_i = quaternionToRotation(state.imu.q);
	Eigen::Matrix3d R_w_c = R_i_c * R_w_i;
	Eigen::Vector3d t_c_w = state.imu.p+R_w_i.transpose()*t_c_i;

	state.cam[state.imu.id] = CamState();
	CamState & cam_state = state.cam[state.imu.id];
	cam_state.id = state.imu.id;
  cam_state.time = frame_time;
  cam_state.q = rotationToQuaternion(R_w_c);
  cam_state.p = t_c_w;

  // Update the covariance matrix of the state.
  // To simplify computation, the matrix J below is the nontrivial block
  // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
  // -aided Inertial Navigation".
	Eigen::Matrix<double, 6, 21> J = Eigen::Matrix<double, 6, 21>::Zero();
  J.block<3, 3>(0, 0) = R_i_c;
  J.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
  J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose()*t_c_i);
  //J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
  J.block<3, 3>(3, 12) = Eigen::Matrix3d::Identity();
  J.block<3, 3>(3, 18) = Eigen::Matrix3d::Identity();

  // Resize the state covariance matrix.
  size_t old_rows = state.state_cov.rows();
  size_t old_cols = state.state_cov.cols();
  state.state_cov.conservativeResize(old_rows+6, old_cols+6);

  // Rename some matrix blocks for convenience.
  const Eigen::Matrix<double, 21, 21>& P11 =
    state.state_cov.block<21, 21>(0, 0);
  const Eigen::MatrixXd& P12 =
    state.state_cov.block(0, 21, 21, old_cols-21);

  // Fill in the augmented state covariance.
  state.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
  state.state_cov.block(0, old_cols, old_rows, 6) =
    state.state_cov.block(old_rows, 0, 6, old_cols).transpose();
  state.state_cov.block<6, 6>(old_rows, old_cols) =
    J * P11 * J.transpose();

	//ROS_INFO("cam size:%d", static_cast<int>(state.cam.size()));
	//ROS_INFO("imu id:%d", static_cast<int>(state.imu.id));

  return;	
}


void Vio::remove_lost_features(MapServer & map)
{
	std::vector<FeatureIdType> invalid_feature_ids;
	std::vector<FeatureIdType> processed_feature_ids;

	for (auto & id_and_f : map) {
		auto & f = id_and_f.second;

		if (f.obserbations.find(state.imu.id) != f.obserbations.end())
			continue;

		if (f.obserbations.size() < 3) {
			invalid_feature_ids.push_back(f.id);
			continue;
		}

		if (!f.is_init) {
			init_feature(f);
			ROS_INFO("feature position:%f,%f,%f",f.p(0),f.p(1),f.p(2));
		}

	}
}


void Vio::init_feature(Feature & feature)
{
	using DiffFunction = ceres::NumericDiffCostFunction<CostFunctor, ceres::CENTRAL, 2, 3>;
	ceres::Problem problem;
	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	double xyz[3] = {0, 0, 0};
	
	for (auto & cam : feature.obserbations) {
		Eigen::Vector4d & q_cam = state.cam[cam.first].q;
		Eigen::Vector3d & p_cam = state.cam[cam.first].p;
		Eigen::Vector2d & uv = cam.second;
		auto functor = new CostFunctor(q_cam, p_cam, uv);

		problem.AddResidualBlock(new DiffFunction(functor),
				nullptr, xyz);
	}
	ceres::Solve(options, & problem, & summary);
}



void Vio::feature_add_ob(const FrameMsgConstPtr & msg)
{
	StateIdType state_id = state.imu.id;

	for (const auto & f:msg->features) {
		if (map.find(f.id) == map.end()) {
			// New feature
			map[f.id] = Feature();
			map[f.id].id = f.id;
			map[f.id].obserbations[state_id] =
				Eigen::Vector2d(f.u0, f.v0);
		} else {
			// Old feature
			map[f.id].obserbations[state_id] =
				Eigen::Vector2d(f.u0, f.v0);
		}
	}
//	ROS_INFO("map size: %d", static_cast<int>(map.size()));
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

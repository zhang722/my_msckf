#pragma once

#include <map>
#include <eigen3/Eigen/Core>
#include "state.h"

namespace my_msckf {

using FeatureIdType = unsigned long long;

using FeatureStateMap = std::map<StateIdType, Eigen::Vector2d,
		std::less<StateIdType>,
		Eigen::aligned_allocator<std::pair<const StateIdType,
		Eigen::Vector2d>>>;

struct Feature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	FeatureIdType id;
	Eigen::Vector3d p;
	bool is_init;
	
	FeatureStateMap obserbations;

	static FeatureIdType next_id;
	
	// Config
	constexpr static double obserbation_noise = 0.01;
  constexpr static double translation_threshold = 0.2;
	constexpr static double huber_epsilon = 0.01;
	constexpr static double estimation_precision = 5e-7;
	constexpr static double initial_damping = 1e-3;
	constexpr static int outer_loop_max_iteration = 10;
	constexpr static int inner_loop_max_iteration = 10;
};

using MapServer = std::map<FeatureIdType, Feature, 
	std::less<int>,
	Eigen::aligned_allocator<
	std::pair<const FeatureIdType, Feature>>> ;

} //namespace my_msckf

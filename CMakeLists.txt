cmake_minimum_required(VERSION 3.0)
project(my_msckf)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	std_msgs
	sensor_msgs
	eigen_conversions
	cv_bridge
	image_transport
)

catkin_package()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_executable(my_msckf_node image_processor.cpp)

target_link_libraries(my_msckf_node ${catkin_LIBRARIES} ${OpenCV_LIBS}) 

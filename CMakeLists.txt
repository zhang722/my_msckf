cmake_minimum_required(VERSION 3.0)
project(my_msckf)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	std_msgs
	sensor_msgs
	message_generation
	eigen_conversions
	cv_bridge
	image_transport
)


add_message_files(
	FILES
	FeatureMsg.msg
	FrameMsg.msg
)



generate_messages(
	DEPENDENCIES
	std_msgs
)




catkin_package()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# Image processor node
add_executable(image_processor_node image_processor.cpp)

target_link_libraries(image_processor_node ${catkin_LIBRARIES} ${OpenCV_LIBS}) 

# Vio node
add_executable(vio_node vio.cpp feature.cpp)
 
target_link_libraries(vio_node ${catkin_LIBRARIES} ${OpenCV_LIBS}) 

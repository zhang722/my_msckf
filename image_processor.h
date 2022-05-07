#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <memory>
#include <array>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>


namespace my_msckf {


using FeatureIdType = unsigned long long; 
struct Feature
{
  FeatureIdType id;
  int lifetime;
  float x;
  float y;
  float response;
};



class ImageProcessor
{
public:
  ImageProcessor();
  ~ImageProcessor() {};

  void init_first_frame(const cv::Mat & img);


  void integrate_imu(cv::Matx33d & cam0_R_p_c);


  void predict_feature(
    const std::vector<cv::Point2f> & input_pts,
    const cv::Matx33d & R_p_c,
    const cv::Matx33d & intrinsics,
    std::vector<cv::Point2f> & output_pts);


  void track_features(std::vector<cv::Mat>& prev_pyramid,
    std::vector<cv::Mat>& curr_pyramid);


  void add_new_features(const cv::Mat & img);
  void create_pyramids(cv_bridge::CvImageConstPtr cam0_curr_img_ptr,
    std::vector<cv::Mat>& curr_cam0_pyramid_);
  void publish(ros::Publisher & frame_pub);


  static bool feature_compare(Feature & f1, Feature & f2)
  {
    return f1.response > f2.response;
  }


  void img_callback(const sensor_msgs::ImageConstPtr& cam_img);


  void imu_callback(const sensor_msgs::ImuConstPtr& msg);


private:
  constexpr static int fast_threshold = 20;
  constexpr static int grid_rows = 5;
  constexpr static int grid_cols = 5;
  constexpr static int grid_feature_num = 5;
  constexpr static int patch_size = 31;
  constexpr static int pyramid_levels = 3;
  constexpr static int max_iteration = 30;
  constexpr static double track_precision = 0.01;


  using GridFeatures = std::array<std::array<std::vector<Feature>, grid_rows>, grid_cols>; 

  bool is_first_img;


  FeatureIdType next_feature_id;

  cv::Matx33d R_cam0_imu;
  cv::Vec3d   t_cam0_imu;
  //Intrinsic matrix
  cv::Matx33d K;

  // IMU message buffer.
  std::vector<sensor_msgs::Imu> imu_msg_buffer;


  cv::Ptr<cv::Feature2D> detector_ptr;

  // Previous and current images
  cv_bridge::CvImageConstPtr cam0_prev_img_ptr;
  cv_bridge::CvImageConstPtr cam0_curr_img_ptr;

  // Pyramids for previous and current image
  std::vector<cv::Mat> prev_cam0_pyramid_;
  std::vector<cv::Mat> curr_cam0_pyramid_;
  //std::vector<cv::Mat> curr_cam1_pyramid_;

  // Features in the previous and current image.
  std::shared_ptr<GridFeatures> prev_features_ptr;
  std::shared_ptr<GridFeatures> curr_features_ptr;

  //ros
  ros::NodeHandle nh_;
  ros::Subscriber img_sub_;
  ros::Subscriber imu_sub_;

  ros::Publisher frame_pub;
};

}

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <utility>
#include <queue>
#include <memory>
#include <array>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>


class ImageProcessor
{
public:
  ImageProcessor();
  ~ImageProcessor() {};
  void fast_extract(const cv::Mat & img, 
                    std::vector<cv::KeyPoint> & new_features,
                    const std::vector<cv::KeyPoint> & old_features 
                    = std::vector<cv::KeyPoint>());
  void grid_features(const cv::Mat & img, 
                     std::vector<cv::KeyPoint> & new_features);

  static bool feature_compare(cv::KeyPoint& f1, cv::KeyPoint& f2)
  {
    return f1.response > f2.response;
  }

  void img_callback(const sensor_msgs::ImageConstPtr& cam_img);

private:


  constexpr static int fast_threshold = 20;
  constexpr static int grid_rows = 5;
  constexpr static int grid_cols = 5;
  constexpr static int grid_feature_num = 5;


  using FrameIdType = unsigned long long;
  using GridFeature = 
        std::array<std::array<std::vector<cv::KeyPoint>, grid_rows>, grid_cols>;
  struct FeaturePerFrame
  {
    FrameIdType id;
    GridFeature grid; 
  };
  using FeatureMananger = std::queue<FeaturePerFrame>;


  cv::Ptr<cv::Feature2D> detector_ptr;
  FeatureMananger feature_manager;

  cv_bridge::CvImageConstPtr cam0_prev_img_ptr_;
  cv_bridge::CvImageConstPtr cam0_curr_img_ptr_;


  ros::NodeHandle nh_;
  ros::Subscriber img_sub_;
};

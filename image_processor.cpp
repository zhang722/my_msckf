#include "image_processor.h"
#include <iostream>


ImageProcessor::ImageProcessor()
{
  detector_ptr = cv::FastFeatureDetector::create(
      fast_threshold);

  img_sub_ = nh_.subscribe("/cam0/image_raw",10, &ImageProcessor::img_callback, this);
}


//Extract fast corners
void ImageProcessor::fast_extract(const cv::Mat & img, 
                                     std::vector<cv::KeyPoint> & new_features,
                                     const std::vector<cv::KeyPoint> & old_features)
{
  //If not needed to set mask 
  if (old_features.empty()) {
    detector_ptr->detect(img, new_features);    
    return;
  }
  //Need a mask
  else {
    using namespace cv;
    // Create a mask to avoid redetecting existing features.
    Mat mask(img.rows, img.cols, CV_8U, Scalar(1));
    for (auto& old_f : old_features) {
      const int y = static_cast<int>(old_f.pt.y);
      const int x = static_cast<int>(old_f.pt.x);

      int up_lim = y-2, bottom_lim = y+3,
          left_lim = x-2, right_lim = x+3;
      if (up_lim < 0) up_lim = 0;
      if (bottom_lim > img.rows) bottom_lim = img.rows;
      if (left_lim < 0) left_lim = 0;
      if (right_lim > img.cols) right_lim = img.cols;

      Range row_range(up_lim, bottom_lim);
      Range col_range(left_lim, right_lim);
      mask(row_range, col_range) = 0;
    }
    // Detect new features.
    detector_ptr->detect(img, new_features, mask);
  }
}


//Griding fast corners into 5x5 (or some else) sections
void ImageProcessor::grid_features(const cv::Mat & img, 
                                     std::vector<cv::KeyPoint> & new_features)
{
  int grid_height = img.rows / grid_rows;
  int grid_width  = img.cols / grid_cols;

  GridFeature grid_feature;
  FeaturePerFrame f_per_frame;
  for (int i = 0; i < grid_rows; ++i) {
    for (int j  = 0; j < grid_cols; ++j ){
      grid_feature[i][j] = std::vector<cv::KeyPoint>(0);
    }
  }

  for (auto it:new_features) {
    int row = static_cast<int>(it.pt.y / grid_height);
    int col = static_cast<int>(it.pt.x / grid_width);
    grid_feature[row][col].push_back(it);
  }

  new_features.clear();

  // Sort the new features in each grid based on its response.
  for (auto& r : grid_feature) {
    for (auto& c : r) {
      std::sort(c.begin(), c.end(),
          &ImageProcessor::feature_compare);
      if (c.size() > grid_feature_num) {
        c.erase(c.begin() + grid_feature_num, c.end());
      }
      for (auto& f : c) {
        new_features.push_back(f);
      }
    }    
  }  
}

void ImageProcessor::img_callback(const sensor_msgs::ImageConstPtr& cam_img)
{
  
  cv::Mat image;
  cam0_curr_img_ptr_ = cv_bridge::toCvShare(cam_img, sensor_msgs::image_encodings::MONO8);
  image = cam0_curr_img_ptr_->image;

  std::vector<cv::KeyPoint> new_features;
  std::vector<cv::KeyPoint> add_features;

  fast_extract(image, new_features);
  grid_features(image, new_features);
  fast_extract(image, add_features, new_features);
  grid_features(image, add_features);

  // cv::drawKeypoints(image, new_features, image);
  // cv::drawKeypoints(image, add_features, image);
  // cv::imshow("关键点", image);
  ROS_INFO("size:%d\n", static_cast<int>(add_features.size()));

}



//Test
int main(int argc, char **argv)
{
  ros::init(argc, argv, "my_msckf");  

  ImageProcessor imageProcessor;
  std::cout << "img process" << std::endl;

  ros::spin();

  return 0;
}

#include "image_processor.h"
#include <iostream>
#include <my_msckf/FrameMsg.h>


namespace my_msckf {

ImageProcessor::ImageProcessor()
{
  std::vector<double> cam0_distortion(4);
  cv::Matx44d T_imu_cam0;
  XmlRpc::XmlRpcValue lines;


  if (!nh_.getParam("cam0/T_cam_imu", lines)) {
    throw (std::runtime_error("cannot find transform "));
  }
  if (lines.size() != 4 || lines.getType() != XmlRpc::XmlRpcValue::TypeArray) {
    throw (std::runtime_error("invalid transform "));
  }
  for (int i = 0; i < lines.size(); i++) {
    if (lines.size() != 4 || lines.getType() != XmlRpc::XmlRpcValue::TypeArray) {
      throw (std::runtime_error("bad line for transform "));
    }
    for (int j = 0; j < lines[i].size(); j++) {
      if (lines[i][j].getType() != XmlRpc::XmlRpcValue::TypeDouble) {
        throw (std::runtime_error("bad value for transform "));
      } else {
        T_imu_cam0(i,j) = lines[i][j];
        //ROS_INFO("T:%f", T_imu_cam0.at<double>(i,j));
      }
    }
  }


  nh_.getParam("cam0/distortion_coeffs", cam0_distortion);
  K = cv::Matx33d::ones();
  K(0, 0) = cam0_distortion[0];
  K(0, 2) = cam0_distortion[2];
  K(1, 1) = cam0_distortion[1];
  K(1, 2) = cam0_distortion[3];
  

  cv::Matx33d R_imu_cam0 = T_imu_cam0.get_minor<3,3>(0,0);
  cv::Vec3d   t_imu_cam0;
  for (int i = 0; i < t_imu_cam0.rows; i++) {
    t_imu_cam0(i) = T_imu_cam0(i,3);
  }
  R_cam0_imu = R_imu_cam0.t();
  t_cam0_imu = -R_imu_cam0.t() * t_imu_cam0;

  next_feature_id = 0;
  is_first_img = true;





  detector_ptr = cv::FastFeatureDetector::create(fast_threshold);
  curr_features_ptr = std::make_shared<GridFeatures>();
  prev_features_ptr = std::make_shared<GridFeatures>();

  img_sub_ = nh_.subscribe("/cam0/image_raw",10, &ImageProcessor::img_callback, this);
  imu_sub_ = nh_.subscribe("/imu0",10, &ImageProcessor::imu_callback, this);
  frame_pub = nh_.advertise<my_msckf::FrameMsg>("features", 3);
}




//Extract fast corners
void ImageProcessor::init_first_frame(const cv::Mat & img)
{

  GridFeatures & new_features = *curr_features_ptr;

  //features in vector
  std::vector<cv::KeyPoint> v_features;

  // Detect new features.
  detector_ptr->detect(img, v_features);

  //Grid into 5x5 sections
  int grid_height = img.rows / grid_rows;
  int grid_width  = img.cols / grid_cols;

  
  for (int i = 0; i < grid_rows; ++i) {
    for (int j  = 0; j < grid_cols; ++j ){
      new_features[i][j].clear();
    }
  }

  for (auto it : v_features) {
    Feature wrap_f; //Wrap a KeyPoint into Feature
    wrap_f.id = 0;
    wrap_f.lifetime = 1;
    wrap_f.x = it.pt.x; 
    wrap_f.y = it.pt.y; 
    wrap_f.response = it.response;
    int row = static_cast<int>(it.pt.y / grid_height);
    int col = static_cast<int>(it.pt.x / grid_width);
    new_features[row][col].push_back(wrap_f);
  }


  // Sort the new features in each grid based on its response.
  for (auto& r : new_features) {  //r is row
    for (auto& v : r) {   //v is vector of features
      std::sort(v.begin(), v.end(),
          &ImageProcessor::feature_compare);
      v.erase(v.begin() + grid_feature_num, v.end());
      for (auto& f : v) {
	f.id = next_feature_id ++;
      }
    }
  }

}


void ImageProcessor::integrate_imu(cv::Matx33d & cam0_R_p_c)
{
  double curr_img_stamp = cam0_curr_img_ptr->header.stamp.toSec();
  double prev_img_stamp = cam0_prev_img_ptr->header.stamp.toSec();
  double dt = prev_img_stamp - curr_img_stamp;


  //Find imu index between prev and curr images;
  auto begin_iter = imu_msg_buffer.begin();
  while (begin_iter != imu_msg_buffer.end() &&
    begin_iter->header.stamp.toSec() - prev_img_stamp < -0.01) {
    ++ begin_iter;
  }

  auto end_iter = begin_iter;
  while (end_iter != imu_msg_buffer.end() &&
    end_iter->header.stamp.toSec() - curr_img_stamp < 0.005) {
    ++ end_iter;
  }

  cv::Vec3d mean_ang_vel(0, 0, 0);
  for (auto iter = begin_iter; iter < end_iter; ++iter) {
    mean_ang_vel += cv::Vec3d(iter->angular_velocity.x,
                              iter->angular_velocity.y,
                              iter->angular_velocity.z);
  }

  if (end_iter - begin_iter > 0) {
    mean_ang_vel *= 1.0f / (end_iter - begin_iter);
  }

  cv::Rodrigues(R_cam0_imu.t() * mean_ang_vel * dt, cam0_R_p_c);
  cam0_R_p_c = cam0_R_p_c.t();

  imu_msg_buffer.erase(imu_msg_buffer.begin(), end_iter);
}





void ImageProcessor::predict_feature(
  const std::vector<cv::Point2f> & input_pts,
  const cv::Matx33d & R_p_c,
  const cv::Matx33d & intrinsics,
  std::vector<cv::Point2f> & output_pts)
{
  if (input_pts.size() == 0) {
    output_pts.clear();
    return;
  }

  output_pts.resize(input_pts.size());

  cv::Matx33d H = intrinsics * R_p_c * intrinsics.inv();

  for (int i = 0; i < input_pts.size(); ++i) {
    cv::Vec3d p1(input_pts[i].x, input_pts[i].y, 1.0);
    cv::Vec3d p2 = H * p1;
    output_pts[i].x = static_cast<float>(p2[0] / p2[2]);
    output_pts[i].y = static_cast<float>(p2[1] / p2[2]);
  }

}






void ImageProcessor::track_features(std::vector<cv::Mat>& prev_pyramid,
  std::vector<cv::Mat>& curr_pyramid)
{
  const cv::Mat& img = cam0_curr_img_ptr->image;
  // Size of each grid.
  int grid_height = img.rows / grid_rows;
  int grid_width  = img.cols / grid_cols;


  GridFeatures & old_features = *prev_features_ptr;
  GridFeatures & new_features = *curr_features_ptr;


  std::vector<cv::Point2f> old_points;
  std::vector<float> old_response;
  std::vector<FeatureIdType> old_id;
  std::vector<int> old_lifetime;

  std::vector<cv::Point2f> new_points;
  std::vector<unsigned char> track_inliers;
  std::vector<float> errors;

  for (auto& c : old_features) {
    for (auto& v : c) {
      for (auto& f : v) {
        cv::Point2f p;
        p.x = f.x;
        p.y = f.y;
        old_points.push_back(p);
        old_response.push_back(f.response);
        old_id.push_back(f.id);
        old_lifetime.push_back(f.lifetime);
      }
    }
  }

  cv::Matx33d cam0_R_p_c;
  // Get trans from prev img to curr img
  integrate_imu(cam0_R_p_c);

  // Predict features' u,v of curr img
  predict_feature(old_points, cam0_R_p_c, K, new_points);

  // Get precise u,v of curr img by LK
  cv::calcOpticalFlowPyrLK(
    prev_pyramid, curr_pyramid,
    old_points, new_points,
    track_inliers, errors);


  for (auto& c : new_features) {
    for (auto& v : c) {
      v.clear();
    }
  }

  for (auto i = 0; i < new_points.size(); ++i) {
    if (track_inliers[i]) {
      if (new_points[i].x < 0 || new_points[i].x > img.cols) continue;
      if (new_points[i].y < 0 || new_points[i].y > img.rows) continue;
      int row = static_cast<int>(new_points[i].y / grid_height);
      int col = static_cast<int>(new_points[i].x / grid_width);
      if (row + 1 > grid_rows) continue;
      if (col + 1 > grid_cols) continue;
      Feature f;
      f.x = new_points[i].x;
      f.y = new_points[i].y;
      f.lifetime = old_lifetime[i] + 1;
      f.response = old_response[i];
      f.id = old_id[i]; 

      new_features[row][col].push_back(f);
    }
  }

}



//Extract fast corners
void ImageProcessor::add_new_features(const cv::Mat & img)
{

  GridFeatures & old_features = *curr_features_ptr;

  //features in vector
  std::vector<cv::KeyPoint> v_features;


  //Need a mask
  using namespace cv;
  // Create a mask to avoid redetecting existing features.
  Mat mask(img.rows, img.cols, CV_8U, Scalar(1));
  for (auto& old_f_row : old_features) { 
    for (auto& v : old_f_row) {
      for (auto& f : v) {
        const int y = static_cast<int>(f.y);
        const int x = static_cast<int>(f.x);


        int up_lim = y-2, bottom_lim = y+3,
            left_lim = x-2, right_lim = x+3;
        if (up_lim < 0) up_lim = 0;
        if (up_lim > img.rows) up_lim = img.rows;
        if (bottom_lim < 0) bottom_lim = 0;
        if (bottom_lim > img.rows) bottom_lim = img.rows;
        if (left_lim < 0) left_lim = 0;
        if (left_lim > img.cols) left_lim = img.cols;
        if (right_lim < 0) right_lim = 0;
        if (right_lim > img.cols) right_lim = img.cols;

        Range row_range(up_lim, bottom_lim);
        Range col_range(left_lim, right_lim);

        mask(row_range, col_range) = 0;  
      }
    }
  }
  // Detect new features.
  detector_ptr->detect(img, v_features, mask);

  //Grid into 5x5 sections
  int grid_height = img.rows / grid_rows;
  int grid_width  = img.cols / grid_cols;


  GridFeatures grid_feature;
  for (int i = 0; i < grid_rows; ++i) {
    for (int j  = 0; j < grid_cols; ++j ){
      grid_feature[i][j].clear();
    }
  }
  for (auto it : v_features) {
    Feature wrap_f; //Wrap a KeyPoint into Feature
    wrap_f.id = 0;
    wrap_f.lifetime = 1;
    wrap_f.x = it.pt.x; 
    wrap_f.y = it.pt.y; 
    wrap_f.response = it.response;
    int row = static_cast<int>(it.pt.y / grid_height);
    int col = static_cast<int>(it.pt.x / grid_width);
    grid_feature[row][col].push_back(wrap_f);
  }


  // Sort the new features in each grid based on its response.
  for (auto& r : grid_feature) {  //r is row
    for (auto& v : r) {   //v is vector of features
      std::sort(v.begin(), v.end(),
          &ImageProcessor::feature_compare);
      if (v.size() > grid_feature_num) {
        v.erase(v.begin() + grid_feature_num, v.end());
      }
    }
  }


  for (int i = 0; i < grid_rows; ++i) {
    for (int j  = 0; j < grid_cols; ++j ){
      if (old_features[i][j].size() < grid_feature_num) {
        int vacancy_num = grid_feature_num - old_features[i][j].size();

        for (int k = 0; k < vacancy_num && k < grid_feature[i][j].size(); ++k) {
          old_features[i][j].push_back(grid_feature[i][j][k]);
          old_features[i][j].back().id = next_feature_id ++;
          old_features[i][j].back().lifetime = 1;
        }
      }
    }
  }

}


void ImageProcessor::img_callback(const sensor_msgs::ImageConstPtr& cam_img)
{
  cv::Mat image;
  cam0_curr_img_ptr = cv_bridge::toCvShare(cam_img, sensor_msgs::image_encodings::MONO8);

  create_pyramids(cam0_curr_img_ptr, curr_cam0_pyramid_);
  image = cam0_curr_img_ptr->image;

  if( cam0_curr_img_ptr->image.type()!= CV_8U)
         cv::cvtColor( cam0_curr_img_ptr->image, image, cv::COLOR_BGR2GRAY);
  
  if (is_first_img) {
    init_first_frame(image);
    is_first_img = false;
  } else {
    track_features(prev_cam0_pyramid_, curr_cam0_pyramid_);

    add_new_features(image);
  }

  publish(frame_pub);
  //cv::Mat out_img(image.rows, image.cols, CV_8UC3);
  //cvtColor(image, out_img.colRange(0, image.cols), CV_GRAY2RGB);

  //for (auto& c : *curr_features_ptr) {
  //  for (auto& v : c) {
  //    for (auto& f : v) {
  //      cv::Point2f point;
  //      point.x = f.x;
  //      point.y = f.y;
  //      cv::circle(out_img, point, 3, (0, 255, 0), -1);
  //    }
  //  }
  //}

  //cv::imshow("关键点", out_img);
  //cv::waitKey(0);
  


  // Update the previous image and previous features.
  cam0_prev_img_ptr = cam0_curr_img_ptr;
  prev_features_ptr = curr_features_ptr;
  std::swap(prev_cam0_pyramid_, curr_cam0_pyramid_);


  // Initialize the current features to empty vectors.
  curr_features_ptr.reset(new GridFeatures());
  for (auto& c : *curr_features_ptr) {
    for (auto& v : c) {
      v.clear();
    }
  }
}

void ImageProcessor::imu_callback(const sensor_msgs::ImuConstPtr& msg)
{
  if (is_first_img) return;
  imu_msg_buffer.push_back(*msg);
}



void ImageProcessor::create_pyramids(cv_bridge::CvImageConstPtr img_ptr,
                                      std::vector<cv::Mat>& pyramid)
{
  cv::Mat img;
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
  clahe->apply(img_ptr->image, img);
    
  const cv::Mat& curr_cam0_img = img;//cam0_curr_img_ptr->image;
  cv::buildOpticalFlowPyramid(
      curr_cam0_img, pyramid,
      cv::Size(patch_size, patch_size),
      pyramid_levels, true, cv::BORDER_REFLECT_101,
      cv::BORDER_CONSTANT, false);
}



void ImageProcessor::publish(ros::Publisher & frame_pub)
{  

  my_msckf::FrameMsg m;
  m.header.stamp = cam0_curr_img_ptr->header.stamp;

  GridFeatures & curr_features = *curr_features_ptr;
  for (auto& c : curr_features) {
    for (auto& v : c) {
      for (auto& f : v) {
	m.features.push_back(my_msckf::FeatureMsg());
	m.features.back().id = f.id;
	m.features.back().u0  = f.x;
	m.features.back().v0  = f.y;

      }
    }
  }
  frame_pub.publish(m);
  return;
}



}

//Test
int main(int argc, char **argv)
{
  ros::init(argc, argv, "my_msckf");  

  my_msckf::ImageProcessor imageProcessor;
  std::cout << "img process" << std::endl;

  ros::spin();

  return 0;
}

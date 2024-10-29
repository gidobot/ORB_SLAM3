#ifndef STEREOODOMETER_H
#define STEREOODOMETER_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <memory.h>
#include <boost/shared_ptr.hpp>

#include "Thirdparty/viso2/src/viso_stereo.h"
#include "Thirdparty/viso2/src/matcher_sift.h"

// to remove after debugging
#include <opencv2/highgui/highgui.hpp>

using namespace viso2;

namespace SIFT_SLAM3
{

class StereoOdometer
{

private:

  boost::shared_ptr<VisualOdometryStereo> visual_odometer_;
  VisualOdometryStereo::parameters visual_odometer_params_;

  bool got_lost_;
  bool first_frame_;
  bool initialized_;

  // change reference frame method. 0, 1 or 2. 0 means allways change. 1 and 2 explained below
  int ref_frame_change_method_;
  bool change_reference_frame_;
  double ref_frame_motion_threshold_; // method 1. Change the reference frame if last motion is small
  int ref_frame_inlier_threshold_; // method 2. Change the reference frame if the number of inliers is low
  Matrix reference_motion_;

  cv::Mat pose_delta_;

public:

  StereoOdometer(): got_lost_(false), change_reference_frame_(false)
  {
    // local_nh.param("ref_frame_change_method", ref_frame_change_method_, 0);
    // local_nh.param("ref_frame_motion_threshold", ref_frame_motion_threshold_, 5.0);
    // local_nh.param("ref_frame_inlier_threshold", ref_frame_inlier_threshold_, 150);
    ref_frame_change_method_ = 0;
    ref_frame_motion_threshold_ = 5.0;
    ref_frame_inlier_threshold_ = 150;

    reference_motion_ = Matrix::eye(4);
  }

  inline void copyMatchData(std::vector<cv::KeyPoint> &kpts_l, std::vector<cv::KeyPoint> &kpts_r, cv::Mat &desc_l, cv::Mat &desc_r,
    std::vector<Matcher_SIFT::p_match> &matches_circle, std::vector<cv::DMatch> &matches_lr)
  {
    visual_odometer_->copyMatchData(kpts_l, kpts_r, desc_l, desc_r, matches_circle, matches_lr);
  }

  inline cv::Mat getPoseDelta() const {return pose_delta_.clone();}

  inline bool isLost() const {return got_lost_;}

  inline bool isInitialized() const {return initialized_;}

  void init(const cv::FileStorage &fSettings)
  {
    // TODO: add error checking
    visual_odometer_params_.calib.cu  = fSettings["Camera1.cx"];
    visual_odometer_params_.calib.cv  = fSettings["Camera1.cy"];
    visual_odometer_params_.calib.f   = fSettings["Camera1.fx"];
    // visual_odometer_params_.base      = float(fSettings["Camera.bf"]) / float(fSettings["Camera.fx"]);
    visual_odometer_params_.base      = float(fSettings["Stereo.b"]);
    visual_odometer_params_.match.num_features = fSettings["SIFTextractor.nFeatures"]; 
    visual_odometer_params_.match.max_dim = fSettings["SIFTextractor.maxDim"]; 

    visual_odometer_.reset(new VisualOdometryStereo(visual_odometer_params_));
    first_frame_ = true;
    initialized_ = false;
  }

  void processNext( const cv::Mat &l_cv_img, const cv::Mat &r_cv_img)
  {
    assert(l_cv_img.cols == r_cv_img.cols);
    assert(l_cv_img.rows == r_cv_img.rows);

    // int32_t dims[] = {l_image_msg->width, l_image_msg->height, l_step};
    // on first run or when odometer got lost, only feed the odometer with
    // images without retrieving data
    if (first_frame_ || got_lost_)
    {
      // visual_odometer_->process(l_image_data, r_image_data, dims);
      visual_odometer_->process(l_cv_img, r_cv_img);
      got_lost_ = false;
      // on first run publish zero once
      if (first_frame_)
      {
        pose_delta_ = cv::Mat::eye(4,4,CV_32F);
        first_frame_ = false;
      }
      return;
    }

    // bool success = visual_odometer_->process(
        // l_image_data, r_image_data, dims, change_reference_frame_);
    bool success = visual_odometer_->process(
        l_cv_img, r_cv_img, change_reference_frame_);
    if (success)
    {
      if (!initialized_)
        initialized_ = true;

      // Matrix motion = Matrix::inv(visual_odometer_->getMotion());
      Matrix motion = visual_odometer_->getMotion();
      // std::cout << "Found %i matches with %i inliers.",
                // visual_odometer_->getNumberOfMatches(),
                // visual_odometer_->getNumberOfInliers();
      // ROS_DEBUG_STREAM("libviso2 returned the following motion:\n" << motion);
      Matrix camera_motion;
      // if image was replaced due to small motion we have to subtract the
      // last motion to get the increment
      if (change_reference_frame_)
      {
        camera_motion = Matrix::inv(reference_motion_) * motion;
      }
      else
      {
        // image was not replaced, report full motion from odometer
        camera_motion = motion;
      }
      reference_motion_ = motion; // store last motion as reference

      pose_delta_ = cv::Mat(4,4,CV_32F);
      pose_delta_.at<float>(0,0) = camera_motion.val[0][0]; pose_delta_.at<float>(0,1) = camera_motion.val[0][1];
      pose_delta_.at<float>(0,2) = camera_motion.val[0][2]; pose_delta_.at<float>(0,3) = camera_motion.val[0][3];
      pose_delta_.at<float>(1,0) = camera_motion.val[1][0]; pose_delta_.at<float>(1,1) = camera_motion.val[1][1];
      pose_delta_.at<float>(1,2) = camera_motion.val[1][2]; pose_delta_.at<float>(1,3) = camera_motion.val[1][3];
      pose_delta_.at<float>(2,0) = camera_motion.val[2][0]; pose_delta_.at<float>(2,1) = camera_motion.val[2][1];
      pose_delta_.at<float>(2,2) = camera_motion.val[2][2]; pose_delta_.at<float>(2,3) = camera_motion.val[2][3];
      pose_delta_.at<float>(3,0) = camera_motion.val[3][0]; pose_delta_.at<float>(3,1) = camera_motion.val[3][1];
      pose_delta_.at<float>(3,2) = camera_motion.val[3][2]; pose_delta_.at<float>(3,3) = camera_motion.val[3][3];
    }
    else
    {
      // ROS_DEBUG("Call to VisualOdometryStereo::process() failed.");
      // ROS_WARN_THROTTLE(10.0, "Visual Odometer got lost!");
      // std::cout << "Visual Odometer got lost!" << std::endl;
      got_lost_ = true;
    }

    if(success)
    {

      // Proceed depending on the reference frame change method
      switch ( ref_frame_change_method_ )
      {
        case 1:
        {
          // calculate current feature flow
          double feature_flow = computeFeatureFlow(visual_odometer_->getMatches());
          change_reference_frame_ = (feature_flow < ref_frame_motion_threshold_);
          // cout << "Feature flow is " << feature_flow
          //     << ", marking last motion as "
          //     << (change_reference_frame_ ? "small." : "normal.");
          break;
        }
        case 2:
        {
          change_reference_frame_ = (visual_odometer_->getNumberOfInliers() > ref_frame_inlier_threshold_);
          break;
        }
        default:
          change_reference_frame_ = false;
      }

    }
    else
      change_reference_frame_ = false;

    // if(!change_reference_frame_)
    // cout << "Changing reference frame" << std::endl;
    // ROS_DEBUG_STREAM("Changing reference frame");
  }

  double computeFeatureFlow(const std::vector<Matcher_SIFT::p_match>& matches)
  {
    double total_flow = 0.0;
    for (size_t i = 0; i < matches.size(); ++i)
    {
      double x_diff = matches[i].u1c - matches[i].u1p;
      double y_diff = matches[i].v1c - matches[i].v1p;
      total_flow += sqrt(x_diff * x_diff + y_diff * y_diff);
    }
    return total_flow / matches.size();
  }

};

} // end of namespace

#endif
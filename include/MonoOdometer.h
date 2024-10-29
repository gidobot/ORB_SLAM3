#ifndef MONOODOMETER_H
#define MONOODOMETER_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <memory.h>
#include <boost/shared_ptr.hpp>

#include "Thirdparty/viso2/src/viso_mono_omnidirectional.h"
#include "Thirdparty/viso2/src/matcher_sift.h"

// to remove after debugging
#include <opencv2/highgui/highgui.hpp>

using namespace viso2;

namespace SIFT_SLAM3
{

class MonoOdometer
{

private:

  boost::shared_ptr<VisualOdometryMonoOmnidirectional> visual_odometer_;
  VisualOdometryMonoOmnidirectional::parameters visual_odometer_params_;

  bool got_lost_;
  bool replace_;

  cv::Mat pose_delta_;

public:

  MonoOdometer(): got_lost_(false), replace_(false)
  {
    pose_delta_ = cv::Mat::eye(4,4,CV_32F);
  }

  inline void copyMatchData(std::vector<cv::KeyPoint> &kpts, cv::Mat &desc,
    std::vector<Matcher_SIFT::p_match> &matches, std::vector<cv::DMatch> &matches_cv)
  {
    visual_odometer_->copyMatchData(kpts, desc, matches, matches_cv);
  }

  inline cv::Mat getPoseDelta() const {return pose_delta_.clone();}

  inline bool isLost() const {return got_lost_;}

  void init(const cv::FileStorage &fSettings, const int camera_id)
  {
    if (camera_id == 1) {
      visual_odometer_params_.omnidirectional_calib.cx  = fSettings["Camera.cx"];
      visual_odometer_params_.omnidirectional_calib.cy  = fSettings["Camera.cy"];
      visual_odometer_params_.omnidirectional_calib.fx  = fSettings["Camera.fx"];
      visual_odometer_params_.omnidirectional_calib.fy  = fSettings["Camera.fy"];

      visual_odometer_params_.omnidirectional_calib.width = fSettings["Camera.width"];
      visual_odometer_params_.omnidirectional_calib.width = fSettings["Camera.height"];

      visual_odometer_params_.omnidirectional_calib.k[0] = fSettings["Camera.k1"];
      visual_odometer_params_.omnidirectional_calib.k[1] = fSettings["Camera.k2"];
      visual_odometer_params_.omnidirectional_calib.k[2] = fSettings["Camera.k3"];
      visual_odometer_params_.omnidirectional_calib.k[3] = fSettings["Camera.k4"];
    }
    else if (camera_id == 2) {
      visual_odometer_params_.omnidirectional_calib.cx  = fSettings["Camera2.cx"];
      visual_odometer_params_.omnidirectional_calib.cy  = fSettings["Camera2.cy"];
      visual_odometer_params_.omnidirectional_calib.fx  = fSettings["Camera2.fx"];
      visual_odometer_params_.omnidirectional_calib.fy  = fSettings["Camera2.fy"];

      visual_odometer_params_.omnidirectional_calib.width = fSettings["Camera.width"];
      visual_odometer_params_.omnidirectional_calib.width = fSettings["Camera.height"];

      visual_odometer_params_.omnidirectional_calib.k[0] = fSettings["Camera2.k1"];
      visual_odometer_params_.omnidirectional_calib.k[1] = fSettings["Camera2.k2"];
      visual_odometer_params_.omnidirectional_calib.k[2] = fSettings["Camera2.k3"];
      visual_odometer_params_.omnidirectional_calib.k[3] = fSettings["Camera2.k4"];
    }

    visual_odometer_params_.match.num_features = fSettings["SIFTextractor.nFeatures"]; 

    visual_odometer_.reset(new VisualOdometryMonoOmnidirectional(visual_odometer_params_));
  }

  void processNext( const cv::Mat &cv_img)
  {
    bool first_run = false;
    // create odometer if not exists
    if (!visual_odometer_)
    {
      first_run = true;
    }

    // int32_t dims[] = {l_image_msg->width, l_image_msg->height, l_step};
    // on first run or when odometer got lost, only feed the odometer with
    // images without retrieving data
    if (first_run)
    {
      // visual_odometer_->process(l_image_data, r_image_data, dims);
      visual_odometer_->process(cv_img, true);
      // on first run publish zero once
      pose_delta_ = cv::Mat::eye(4,4,CV_32F);
    }
    else
    {
      bool success = visual_odometer_->process(cv_img, got_lost_);
      if (true)
        got_lost_ = false;
      // if (success)
      else if (success)
      {
        Matrix motion = visual_odometer_->getMotion();
        // std::cout << "Found " << visual_odometer_->getNumberOfMatches() << " matches with " << visual_odometer_->getNumberOfInliers() << " inliers.\n",
        // ROS_DEBUG_STREAM("libviso2 returned the following motion:\n" << motion);

        pose_delta_ = cv::Mat(4,4,CV_32F);
        pose_delta_.at<float>(0,0) = motion.val[0][0]; pose_delta_.at<float>(0,1) = motion.val[0][1];
        pose_delta_.at<float>(0,2) = motion.val[0][2]; pose_delta_.at<float>(0,3) = motion.val[0][3];
        pose_delta_.at<float>(1,0) = motion.val[1][0]; pose_delta_.at<float>(1,1) = motion.val[1][1];
        pose_delta_.at<float>(1,2) = motion.val[1][2]; pose_delta_.at<float>(1,3) = motion.val[1][3];
        pose_delta_.at<float>(2,0) = motion.val[2][0]; pose_delta_.at<float>(2,1) = motion.val[2][1];
        pose_delta_.at<float>(2,2) = motion.val[2][2]; pose_delta_.at<float>(2,3) = motion.val[2][3];
        pose_delta_.at<float>(3,0) = motion.val[3][0]; pose_delta_.at<float>(3,1) = motion.val[3][1];
        pose_delta_.at<float>(3,2) = motion.val[3][2]; pose_delta_.at<float>(3,3) = motion.val[3][3];
      }
      else
      {
        // ROS_DEBUG("Call to VisualOdometryStereo::process() failed.");
        // ROS_WARN_THROTTLE(10.0, "Visual Odometer got lost!");
        // std::cout << "Visual Odometer got lost!" << std::endl;
        pose_delta_ = cv::Mat::eye(4,4,CV_32F);
        got_lost_ = true;
      }
    }
  }
  
};

} // end of namespace

#endif
#include "CUDAmatcher.h"

using namespace std;

namespace SIFT_SLAM3
{

  const float CUDAmatcher::TH_HIGH = 0.9;
  const float CUDAmatcher::TH_LOW = 0.6;

  CUDAmatcher::CUDAmatcher(const cv::FileStorage &fSettings, int nfeats, float nnratio, bool checkOri): fSettings(std::make_shared<cv::FileStorage>(fSettings)), mfNNratio(nnratio), mbCheckOrientation(checkOri)
  {
      InitSiftData(siftdata_1, nfeats, true, true);
      InitSiftData(siftdata_2, nfeats, true, true);
  }


  CUDAmatcher::~CUDAmatcher() {
     FreeSiftData(siftdata_1);
     FreeSiftData(siftdata_2);
  }

  double CUDAmatcher::computeFeaturesCUDA(const cv::Mat &I, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc)
  {
  	cv::Mat gI;
    I.convertTo(gI, CV_32FC1);

    // float initBlur = 1.6f;
    // float thresh = 1.2f; // for stereo
    // float thresh = 1.0f; // for hybrid
    float initBlur = 1.0f;
    float thresh = 2.0f;

    CudaImage img;
    // don't include memory allocation in timing as this can in theory be done once
    img.Allocate(gI.cols, gI.rows, iAlignUp(gI.cols, 128), false, NULL, (float*)gI.data);
    float *memoryTmpCUDA = AllocSiftTempMemory(I.cols, I.rows, 5, true);

    // time_point t1 = std::chrono::steady_clock::now();

    img.Download();

    // float *memoryTmpCUDA = AllocSiftTempMemory(I1.cols, I1.rows, 5, false);
    // ExtractSift(siftdata_1, img1, 5, initBlur, thresh, 0.0f, false, memoryTmpCUDA);
    ExtractSift(siftdata_1, img, 5, initBlur, thresh, 0.0f, true, memoryTmpCUDA);

    // time_point t2 = std::chrono::steady_clock::now();
    // double tt = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    FreeSiftTempMemory(memoryTmpCUDA);

    // plotPatches(siftdata_1);
    
    CUDAtoCV(siftdata_1, kpts, desc);

    // return tt;
    return 1;
  }

  int CUDAmatcher::SearchByBF(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches)
  {
    std::vector<cv::DMatch> matches, inliers;
    matchCUDA(pKF->mvKeysUn, pKF->mDescriptors, F.mvKeysUn, F.mDescriptors, matches, 0.9, true);
    findInliers(pKF->mvKeysUn, F.mvKeysUn, matches, inliers);
    int nmatches = 0;

    // set map point matches
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));
    for (size_t i = 0; i < inliers.size(); i++)
    {
        MapPoint* pMP = vpMapPointsKF[inliers[i].queryIdx];
        if (!pMP)
            continue;
        if(pMP->isBad())
            continue;
        vpMapPointMatches[inliers[i].trainIdx] = pMP;
        nmatches++;
    }

    return nmatches;
  }

  int CUDAmatcher::SearchByBF(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint*> &vpMatches12)
  {
    std::vector<cv::DMatch> matches, inliers;
    matchCUDA(pKF1->mvKeysUn, pKF1->mDescriptors, pKF2->mvKeysUn, pKF2->mDescriptors, matches, 0.9, true);
    // findInliers(pKF1->mvKeysUn, pKF2->mvKeysUn, matches, inliers);
    int nmatches = 0;

    // set map point matches
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    for (size_t i = 0; i < matches.size(); i++)
    {
        // MapPoint* pMP1 = vpMapPoints1[inliers[i].queryIdx];
        // if (!pMP1)
            // continue;
        // if(pMP1->isBad())
            // continue;

        MapPoint* pMP2 = vpMapPoints2[matches[i].trainIdx];
        if (!pMP2)
            continue;
        if(pMP2->isBad())
            continue;

        vpMatches12[matches[i].queryIdx] = pMP2;
        nmatches++;
    }

    return nmatches;
  }

  double CUDAmatcher::matchCUDA(const std::vector<cv::KeyPoint> &kpts1, const cv::Mat &desc1,
    const std::vector<cv::KeyPoint> &kpts2, const cv::Mat &desc2, std::vector<cv::DMatch> &matches,
    const float ratio_thresh, const bool cross_check)
  {
    // time_point t1 = std::chrono::steady_clock::now();

    // Copy keypoints to cudaSift objects
    #ifdef MANAGEDMEM
      SiftPoint *sift1 = siftdata_1.m_data;
      SiftPoint *sift2 = siftdata_2.m_data;
    #else
      SiftPoint *sift1 = siftdata_1.h_data;
      SiftPoint *sift2 = siftdata_2.h_data;
    #endif

    cv::Mat row;
    for (int32_t i=0; i<kpts1.size(); i++) {
      row = desc1.row(i);
      std::memcpy(sift1[i].data, row.data, 128*sizeof(float));
      sift1[i].xpos = kpts1[i].pt.x;
      sift1[i].ypos = kpts1[i].pt.y;
      sift1[i].orientation = kpts1[i].angle;
    }
    siftdata_1.numPts = kpts1.size();
    CopyToDevice(siftdata_1);

    for (int32_t i=0; i<kpts2.size(); i++) {
      row = desc2.row(i);
      std::memcpy(sift2[i].data, row.data, 128*sizeof(float));
      sift2[i].xpos = kpts2[i].pt.x;
      sift2[i].ypos = kpts2[i].pt.y;
      sift2[i].orientation = kpts2[i].angle;
    }
    siftdata_2.numPts = kpts2.size();
    CopyToDevice(siftdata_2);

    // BF matching with left right consistency
    int32_t i1c,i2c,i1c2;

    matches.clear();

    MatchSiftData(siftdata_1, siftdata_2);
    if (cross_check)
      MatchSiftData(siftdata_2, siftdata_1);

    std::vector<float> score_match(kpts2.size()*2, 0.0);
    std::vector<int> index_match(kpts2.size()*2, -1);

    int index;
    for (int32_t i1c=0; i1c<siftdata_1.numPts; i1c++) {
      // check distance ratio thresh
      if (sift1[i1c].ambiguity > ratio_thresh)
        continue;
      i2c = sift1[i1c].match;

      // for stereo pairs
      // if ((*fSettings)["Matcher.stereo"].real()) {
      //   // check epipolar constraint
      //   if (fabs(sift1[i1c].match_ypos - sift1[i1c].ypos) > (*fSettings)["Matcher.match_disp_tolerance"].real())
      //     continue;
      //   // filter negative disparities
      //   if (sift1[i1c].xpos>=sift2[i2c].xpos)
      //     continue;
      // }

      // check mutual best feature match in both images
      if (cross_check) {
        i1c2 = sift2[i2c].match;
        if (i1c != i1c2)
          continue;
      }

      // return matches where each trainIdx index is associated with only one queryIdx 
      // trainIdx has not been matched yet
      if (score_match[i2c] == 0.0) {
        score_match[i2c] = sift1[i1c].score;
        matches.push_back(cv::DMatch(i1c, i2c, sift1[i1c].score));
        index_match[i2c] = matches.size() - 1;
      }
      // we have already a match for trainIdx: if stored match is worse => replace it
      else if (sift1[i1c].score > score_match[i2c]) {
        index = index_match[i2c];
        assert(matches[index].trainIdx == i2c);
        matches[index].queryIdx = i1c;
        matches[index].distance = sift1[i1c].score;
      }
    }

    // time_point t2 = std::chrono::steady_clock::now();
    // double tt = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    // return tt;
    return 1;
  }

  std::vector<double> CUDAmatcher::findInliers(const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, 
    const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inliers, bool is_fish1, bool is_fish2)
  {
    inliers.clear();
    if (matches.size()<10)
      return vector<double>({0,0,0,0,0,0});

    // copy matched points into cv vector
    std::vector<cv::Point2d> points1(matches.size());
    std::vector<cv::Point2d> points2(matches.size());

    for(size_t i = 0; i < matches.size(); i++) {
      int idx1, idx2;
      idx1 = matches[i].queryIdx;
      idx2 = matches[i].trainIdx;
      points1[i] = cv::Point2f(kpts1[idx1].pt.x, kpts1[idx1].pt.y);
      points2[i] = cv::Point2f(kpts2[idx2].pt.x, kpts2[idx2].pt.y);
    }

    // ToDo: update for stereo
    double fx = (*fSettings)["Camera1.fx"].real();

    if (is_fish1) {
      undistortFisheye(points1);
    }
    else {
      undistortPerspective(points1);
    }
    if (is_fish2) {
      undistortFisheye(points2);
    }
    else {
      undistortPerspective(points2);
    }

    // find essential matrix with 5-point algorithm
    double focal = 1.0;
    cv::Point2d pp(0,0);
    float prob = 0.999;
    float thresh = (*fSettings)["SIFTmatcher.projThresh"].real()/fx; // projection error is measured by projection onto stereo
    // float thresh = 3.0/729.; // projection error is measured by projection onto stereo
    int method = cv::RANSAC;
    cv::Mat mask;
    cv::Mat essential_mat = cv::findEssentialMat(points1, points2, focal, pp, method, prob, thresh, mask);

    // recover motion from essential matrix
    cv::Mat t;
    cv::Mat R;
    int num_inliers = cv::recoverPose(essential_mat, points1, points2, R, t, focal, pp, mask);

    // if (num_inliers < 10)
      // return vector<double>({0,0,0,0,0,0,0});

    for(int i = 0; i < mask.rows; i++) {
      if(mask.at<unsigned char>(i)){
        inliers.push_back(matches[i]);
      }
    }

    vector<double> q_delta = toQuaternion(R);

    double t_mag = sqrtf(t.at<double>(0)*t.at<double>(0) + t.at<double>(1)*t.at<double>(1) + t.at<double>(2)*t.at<double>(2));
    
    // return parameter vector
    vector<double> tr_delta;
    tr_delta.resize(7);
    // x,y,z,qw,qx,qy,qz
    tr_delta[0] = t.at<double>(0)/t_mag;
    tr_delta[1] = t.at<double>(1)/t_mag;
    tr_delta[2] = t.at<double>(2)/t_mag;
    tr_delta[3] = q_delta[0];
    tr_delta[4] = q_delta[1];
    tr_delta[5] = q_delta[2];
    tr_delta[6] = q_delta[3];
    return tr_delta;
  }

  void CUDAmatcher::CUDAtoCV(SiftData &siftdata, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc)
  {
    #ifdef MANAGEDMEM
      SiftPoint *sift = siftdata.m_data;
    #else
      SiftPoint *sift = siftdata.h_data;
    #endif

    kpts.clear();
    desc = cv::Mat(siftdata.numPts, 128, CV_32F);

    cv::Mat row;
    for (int32_t i=0; i<siftdata.numPts; i++) {
      kpts.push_back(cv::KeyPoint(sift[i].xpos, sift[i].ypos, sift[i].scale, sift[i].orientation, 1, (int32_t)log2(sift[i].subsampling)));
      // int32_t octave = (int32_t)log2(sift[i].subsampling);
      row = desc.row(i);
      std::memcpy(row.data, sift[i].data, 128*sizeof(float));
    }
  }


  std::vector<double> CUDAmatcher::toQuaternion(const cv::Mat &M)
  {
      Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
      Eigen::Quaterniond q(eigMat);
      std::vector<double> v(4);
      v[0] = q.w();
      v[1] = q.x();
      v[2] = q.y();
      v[3] = q.z();
      return v;
  }

  Eigen::Matrix<double,3,3> CUDAmatcher::toMatrix3d(const cv::Mat &cvMat3)
  {
      Eigen::Matrix<double,3,3> M;
      M << cvMat3.at<double>(0,0), cvMat3.at<double>(0,1), cvMat3.at<double>(0,2),
           cvMat3.at<double>(1,0), cvMat3.at<double>(1,1), cvMat3.at<double>(1,2),
           cvMat3.at<double>(2,0), cvMat3.at<double>(2,1), cvMat3.at<double>(2,2);
      return M;
  }

  void CUDAmatcher::undistortFisheye(std::vector<cv::Point2d> &points)
  {
    double fx = (*fSettings)["Camera1.fx"].real();
    double fy = (*fSettings)["Camera1.fy"].real();
    double cx = (*fSettings)["Camera1.cx"].real();
    double cy = (*fSettings)["Camera1.cy"].real();
    double k1 = (*fSettings)["Camera1.k1"].real();
    double k2 = (*fSettings)["Camera1.k2"].real();
    double k3 = (*fSettings)["Camera1.k3"].real();
    double k4 = (*fSettings)["Camera1.k4"].real();
    // rectify points to normalized image coordinates
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f); 
    cv::Mat dist = (cv::Mat_<double>(1,4) << k1, k2, k3, k4);
    cv::fisheye::undistortPoints(points, points, cameraMatrix, dist);
  }

  void CUDAmatcher::undistortPerspective(std::vector<cv::Point2d> &points)
  {
    double fx = (*fSettings)["Camera1.fx"].real();
    double fy = (*fSettings)["Camera1.fy"].real();
    double cx = (*fSettings)["Camera1.cx"].real();
    double cy = (*fSettings)["Camera1.cy"].real();
    // double k1 = (*fSettings)["Camera1.k1"].real();
    // double k2 = (*fSettings)["Camera1.k2"].real();
    // double p1 = (*fSettings)["Camera1.p1"].real();
    // double p2 = (*fSettings)["Camera1.p2"].real();
    // Assume undistorted keypoints, but still project to identity camera matrix
    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    // rectify points to ormalized image coordinates
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f); 
    cv::Mat dist = (cv::Mat_<double>(1,4) << k1, k2, p1, p2);
    cv::undistortPoints(points, points, cameraMatrix, dist);
  }

} //namespace SIFT_SLAM3
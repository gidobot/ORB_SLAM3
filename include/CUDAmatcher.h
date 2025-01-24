#ifndef CUDAMATCHER_H
#define CUDAMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<Eigen/Dense>

#include"Thirdparty/CudaSift/cudaSift.h"
#include"Thirdparty/CudaSift/cudaImage.h"

#include"Frame.h"
#include"MapPoint.h"
#include"KeyFrame.h"

namespace SIFT_SLAM3
{
	class MapPoint;
	class KeyFrame;
	class Frame;

	class CUDAmatcher
	{
	public:

		CUDAmatcher(const cv::FileStorage &fSettings, int nfeats=2000, float nnratio=0.6, bool checkOri=true);

  		~CUDAmatcher();

  		int SearchByBF(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
        int SearchByBF(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint*> &vpMatches12);

  		double computeFeaturesCUDA(const cv::Mat &I, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);

		double matchCUDA(const std::vector<cv::KeyPoint> &kpts1, const cv::Mat &desc1,
			const std::vector<cv::KeyPoint> &kpts2, const cv::Mat &desc2, std::vector<cv::DMatch> &matches,
			const float ratio_thresh, const bool cross_check=false);

		std::vector<double> findInliers(const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, 
			const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inliers, bool is_fish1=false, bool is_fish2=false);

	private:

  		void CUDAtoCV(SiftData &siftdata, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);

  		void undistortFisheye(std::vector<cv::Point2d> &points);
  		void undistortPerspective(std::vector<cv::Point2d> &points);

  		std::vector<double> toQuaternion(const cv::Mat &M);
  		Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);

        static const float TH_LOW;
        static const float TH_HIGH;
        static const int HISTO_LENGTH;

        float mfNNratio;
        bool mbCheckOrientation;

  		SiftData siftdata_1;
  		SiftData siftdata_2;

  		std::shared_ptr<cv::FileStorage> fSettings;
  	};

} // namespace SIFT_SLAM3

#endif // CUDAMATCHER_H
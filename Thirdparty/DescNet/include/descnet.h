/**
 * @file
 * @brief The DescNet class wraps the DescNet Tensorflow network for feature description.
 */

#ifndef DESCNET_H
#define DESCNET_H

#include <iostream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "model.h"
#include <chrono>

using namespace std;

namespace descnet {
	
typedef std::chrono::steady_clock::time_point time_point;

class DescNet
{
public:
	DescNet(const string &model_path);
	DescNet() = delete;
	void compute(std::vector<cv::KeyPoint> &kpts);
	void getPatchParams(std::vector<cv::KeyPoint> &kpts, const cv::Mat &img, cv::Mat &patch_param);
	void getPatches(std::vector<cv::KeyPoint> &kpts, const cv::Mat &img, cv::Mat &all_patches);
	void getPatchesCuda(std::vector<cv::KeyPoint> &kpts, const cv::Mat &img, cv::Mat &all_patches);
	void convertKpts(std::vector<cv::KeyPoint> &kpts, cv::Mat &npy_kpts);
	void unpackOctave(cv::KeyPoint &kpt, int &octave, int &layer, float &scale);
	void computeFeatures(const cv::Mat& img, std::vector<cv::KeyPoint> &kpts, cv::Mat &desc);
	void computeFeatures(cv::Mat& all_patches, cv::Mat &desc);
	void computeFeaturesCUDA(const int batch_size, float *cuda_patches, cv::Mat &desc);
private:
	descnet::Model model;
};

} // namespace descnet

#endif // DESCNET_H
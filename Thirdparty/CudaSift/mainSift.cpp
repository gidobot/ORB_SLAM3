//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//  

#include <iostream>  
#include <cmath>
#include <iomanip>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

using namespace std;

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);
void FindEssentialMatrix(SiftData &siftData1, SiftData &siftData2);
void FindEssentialMatrixCV(SiftData &siftData1, SiftData &siftData2, cv::Mat &img1, cv::Mat &img2);
void FindEssentialMatrixStereo(SiftData &siftData1, SiftData &siftData2, cv::Mat &img1, cv::Mat &img2);
void PlotMatches(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2, std::vector<cv::DMatch> matches, std::vector<int32_t> good);

double ScaleUp(CudaImage &res, CudaImage &src);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{    
  int devNum = 0, imgSet = 0;
  if (argc>1)
    devNum = std::atoi(argv[1]);
  if (argc>2)
    imgSet = std::atoi(argv[2]);

  // Read images using OpenCV
  cv::Mat limg, rimg;
  // cv::imread("data/img1.png", 0).convertTo(limg, CV_32FC1);
  // cv::imread("data/img2.png", 0).convertTo(rimg, CV_32FC1);
  // cv::Mat imLeft = cv::imread("data/fish.png",cv::IMREAD_UNCHANGED);
  // cv::Mat imLeft = cv::imread("data/left.png",cv::IMREAD_UNCHANGED);
  // cv::Mat imRight = cv::imread("data/right.png",cv::IMREAD_UNCHANGED);
  cv::Mat imLeft = cv::imread("data/img1.png",cv::IMREAD_UNCHANGED);
  cv::Mat imRight = cv::imread("data/img2.png",cv::IMREAD_UNCHANGED);

  // cv::cvtColor(imLeft, imLeft, cv::COLOR_BGR2RGB);
  // cv::cvtColor(imRight, imRight, cv::COLOR_BGR2RGB);
  imLeft.convertTo(limg, CV_32FC1);
  imRight.convertTo(rimg, CV_32FC1);
  double minVal; double maxVal;
  cv::minMaxLoc(limg, &minVal, &maxVal);
  cout << "Min val: " << minVal << " Max val: " << maxVal << endl;
  cv::minMaxLoc(rimg, &minVal, &maxVal);
  cout << "Min val: " << minVal << " Max val: " << maxVal << endl;
  //cv::flip(limg, rimg, -1);
  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  std::cout << "Image size = (" << w << "," << h << ")" << std::endl;
  
  // Initial Cuda images and download images to device
  std::cout << "Initializing data..." << std::endl;
  InitCuda(devNum); 
  CudaImage img1, img2;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
  img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
  img1.Download();
  img2.Download(); 

  // Extract Sift features from images
  SiftData siftData1, siftData2;
  float initBlur = 1.6f;
  // float thresh = (imgSet ? 4.5f : 3.0f);
  // float thresh = (imgSet ? 4.5f : 1.5f);
  float thresh = 1.2f;
  InitSiftData(siftData1, 4000, true, true); 
  InitSiftData(siftData2, 4000, true, true);
  
  // A bit of benchmarking 
  //for (int thresh1=1.00f;thresh1<=4.01f;thresh1+=0.50f) {
  float *memoryTmp = AllocSiftTempMemory(w, h, 5, true);
    // for (int i=0;i<1000;i++) {
      ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f, true, memoryTmp);
      ExtractSift(siftData2, img2, 5, initBlur, thresh, 0.0f, true, memoryTmp);
    // }
    FreeSiftTempMemory(memoryTmp);
    
    // Match Sift features and find a homography
    // for (int i=0;i<1;i++)
    // MatchSiftData(siftData1, siftData2);
    // float homography[9];
    // int numMatches;
    // FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.9f, 10.0);
    // int numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.9f, 5.0);
    
    std::cout << "Number of original features: " <<  siftData1.numPts << " " << siftData2.numPts << std::endl;
    // std::cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numFit/std::min(siftData1.numPts, siftData2.numPts) << "% " << initBlur << " " << thresh << std::endl;
    //}
    // FindEssentialMatrix(siftData1, siftData2);
    // FindEssentialMatrixCV(siftData1, siftData2, imLeft, imRight);
    FindEssentialMatrixStereo(siftData1, siftData2, imLeft, imRight);
  
  // Print out and store summary data
  PrintMatchData(siftData1, siftData2, img1);
  cv::imwrite("data/limg_pts.pgm", limg);

  // MatchAll(siftData1, siftData2, homography);
  
  // Free Sift data from device
  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
}

void PlotMatches(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2, std::vector<cv::DMatch> matches, std::vector<int32_t> good)
{
  // draw matches
  std::vector<cv::DMatch> inliers;

  for (std::vector<int32_t>::const_iterator it=good.begin(); it!=good.end(); it++) {
    inliers.push_back(matches[*it]);
  }

  cv::Mat matchImg;
  cv::drawMatches(img1, kpts1, img2, kpts2, inliers, matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1));
  cv::namedWindow("Matches", cv::WINDOW_NORMAL);
  cv::imshow("Matches", matchImg);
  cv::resizeWindow("Matches", 600, 600);
  cv::waitKey(0);
}

void FindEssentialMatrixCV(SiftData &siftData1, SiftData &siftData2, cv::Mat &img1, cv::Mat &img2)
{
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;

  vector<cv::KeyPoint> kpts1;
  vector<cv::KeyPoint> kpts2;
  cv::Mat desc1 = cv::Mat(numPts1, 128, CV_32F);
  cv::Mat desc2 = cv::Mat(numPts2, 128, CV_32F);

  cv::Mat row;
  for (int32_t i=0; i<numPts1; i++) {
    kpts1.push_back(cv::KeyPoint(sift1[i].xpos, sift1[i].ypos, sift1[i].scale, sift1[i].orientation, 1, (int32_t)log2(sift1[i].subsampling)));
    row = desc1.row(i);
    std::memcpy(row.data, sift1[i].data, 128*sizeof(float));
  }

  for (int32_t i=0; i<numPts2; i++) {
    kpts2.push_back(cv::KeyPoint(sift2[i].xpos, sift2[i].ypos, sift2[i].scale, sift2[i].orientation, 1, (int32_t)log2(sift2[i].subsampling)));
    row = desc2.row(i);
    std::memcpy(row.data, sift2[i].data, 128*sizeof(float));
  }

  // matching descriptors
  cv::Ptr<cv::DescriptorMatcher> matcher;
  // cv::BFMatcher matcher(cv::NORM_L2, true);
  matcher = cv::BFMatcher::create(cv::NORM_L2);
  vector<vector<cv::DMatch>> matches;
  vector<cv::DMatch> goodMatches;
  matcher->knnMatch(desc1, desc2, matches, 2);

  for (int i=0; i<matches.size(); i++)
  {
    if (matches[i][0].distance < 0.90 * matches[i][1].distance)
      goodMatches.push_back(matches[i][0]);
  }

  cout << "Initial matches: " << goodMatches.size() << endl;

  // copy matched points into cv vector
  std::vector<cv::Point2d> pointsFish(goodMatches.size());
  std::vector<cv::Point2d> pointsStereo(goodMatches.size());

  for(size_t i = 0; i < goodMatches.size(); i++) {
    int iF = goodMatches[i].queryIdx;
    int iS = goodMatches[i].trainIdx;

    cv::KeyPoint kF = kpts1[iF];
    cv::KeyPoint kS = kpts2[iS];

    pointsFish[i] = cv::Point2f(kF.pt);
    pointsStereo[i] = cv::Point2f(kS.pt);
  }

  // vectors for rectified points
  std::vector<cv::Point2d> pointsRect1;
  std::vector<cv::Point2d> pointsRect2;

  double fx1 = 769.5519232429078;
  double fy1 = 768.8322015619301;
  double cx1 = 1268.7948261550591;
  double cy1 = 1023.8486413295748;
  double k1  = 0.025513637146558403;
  double k2  = -0.011386600859137145;
  double k3  = 0.013688146542497151;
  double k4  = -0.0076052132438100654;

  double fx2 = 1899.8417;
  double fy2 = 1899.8417;
  double cx2 = 1202.86649;
  double cy2 = 985.03928; 

  cv::Mat K1 = (cv::Mat_<double>(3,3) << fx1, 0.f, cx1, 0.f, fy1, cy1, 0.f, 0.f, 1.f); 
  cv::Mat D1 = (cv::Mat_<double>(1,4) << k1, k2, k3, k4);

  cv::Mat K2 = (cv::Mat_<double>(3,3) << fx2, 0.f, cx2, 0.f, fy2, cy2, 0.f, 0.f, 1.f); 
  cv::Mat D2 = (cv::Mat_<double>(1,4) << 0, 0, 0, 0);

  // rectify points to normalized image coordinates
  cv::fisheye::undistortPoints(pointsFish, pointsRect1, K1, D1);
  cv::undistortPoints(pointsStereo, pointsRect2, K2, D2);

  // find essential matrix with 5-point algorithm
  double focal = 1.0;
  cv::Point2d pp(0,0);
  float prob = 0.999;
  float thresh = 0.5/769;
  int method = cv::RANSAC;
  cv::Mat mask;
  cv::Mat essentialMat = cv::findEssentialMat(pointsRect1, pointsRect2, focal, pp, method, prob, thresh, mask);

  vector<int32_t> inliers;

  for(int i = 0; i < mask.rows; i++) {
    if(mask.at<unsigned char>(i)){
      inliers.push_back(i);
    }
  }

  cout << "Inliers: " << inliers.size() << endl;
  cout << "Percentage inliers: " << 100.0f*inliers.size()/goodMatches.size() << "%" << endl;

  PlotMatches(img1, img2, kpts1, kpts2, goodMatches, inliers);
}
void FindEssentialMatrixStereo(SiftData &siftData1, SiftData &siftData2, cv::Mat &img1, cv::Mat &img2)
{
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;

  vector<cv::KeyPoint> kpts1;
  vector<cv::KeyPoint> kpts2;
  cv::Mat desc1 = cv::Mat(numPts1, 128, CV_32F);
  cv::Mat desc2 = cv::Mat(numPts2, 128, CV_32F);

  cv::Mat row;
  for (int32_t i=0; i<numPts1; i++) {
    kpts1.push_back(cv::KeyPoint(sift1[i].xpos, sift1[i].ypos, sift1[i].scale, sift1[i].orientation, 1, (int32_t)log2(sift1[i].subsampling)));
    row = desc1.row(i);
    std::memcpy(row.data, sift1[i].data, 128*sizeof(float));
  }

  for (int32_t i=0; i<numPts2; i++) {
    kpts2.push_back(cv::KeyPoint(sift2[i].xpos, sift2[i].ypos, sift2[i].scale, sift2[i].orientation, 1, (int32_t)log2(sift2[i].subsampling)));
    row = desc2.row(i);
    std::memcpy(row.data, sift2[i].data, 128*sizeof(float));
  }

  // matching descriptors
  cv::Ptr<cv::DescriptorMatcher> matcher;
  // cv::BFMatcher matcher(cv::NORM_L2, true);
  matcher = cv::BFMatcher::create(cv::NORM_L2);
  vector<vector<cv::DMatch>> matches;
  vector<cv::DMatch> goodMatches;
  matcher->knnMatch(desc1, desc2, matches, 2);

  for (int i=0; i<matches.size(); i++)
  {
    if (matches[i][0].distance < 0.90 * matches[i][1].distance)
      goodMatches.push_back(matches[i][0]);
  }

  cout << "Initial matches: " << goodMatches.size() << endl;

  // copy matched points into cv vector
  std::vector<cv::Point2d> pointsLeft(goodMatches.size());
  std::vector<cv::Point2d> pointsRight(goodMatches.size());

  for(size_t i = 0; i < goodMatches.size(); i++) {
    int iF = goodMatches[i].queryIdx;
    int iS = goodMatches[i].trainIdx;

    cv::KeyPoint kF = kpts1[iF];
    cv::KeyPoint kS = kpts2[iS];

    pointsLeft[i] = cv::Point2f(kF.pt);
    pointsRight[i] = cv::Point2f(kS.pt);
  }

  // vectors for rectified points
  std::vector<cv::Point2d> pointsRect1;
  std::vector<cv::Point2d> pointsRect2;

  double fx2 = 1899.8417;
  double fy2 = 1899.8417;
  double cx2 = 1202.86649;
  double cy2 = 985.03928; 

  cv::Mat K2 = (cv::Mat_<double>(3,3) << fx2, 0.f, cx2, 0.f, fy2, cy2, 0.f, 0.f, 1.f); 
  cv::Mat D2 = (cv::Mat_<double>(1,4) << 0, 0, 0, 0);

  // rectify points to normalized image coordinates
  cv::undistortPoints(pointsLeft, pointsRect1, K2, D2);
  cv::undistortPoints(pointsRight, pointsRect2, K2, D2);

  // find essential matrix with 5-point algorithm
  double focal = 1.0;
  cv::Point2d pp(0,0);
  float prob = 0.999;
  float thresh = 0.5/1000.0;
  int method = cv::RANSAC;
  cv::Mat mask;
  cv::Mat essentialMat = cv::findEssentialMat(pointsRect1, pointsRect2, focal, pp, method, prob, thresh, mask);

  vector<int32_t> inliers;

  for(int i = 0; i < mask.rows; i++) {
    if(mask.at<unsigned char>(i)){
      inliers.push_back(i);
    }
  }

  cout << "Inliers: " << inliers.size() << endl;
  cout << "Percentage inliers: " << 100.0f*inliers.size()/goodMatches.size() << "%" << endl;

  vector<int32_t> all;

  for(int i = 0; i < goodMatches.size(); i++) {
    all.push_back(i);
  }

  cout << "All matches" << endl;
  PlotMatches(img1, img2, kpts1, kpts2, goodMatches, all);
  cout << "Inlier matches" << endl;
  PlotMatches(img1, img2, kpts1, kpts2, goodMatches, inliers);
}

void FindEssentialMatrix(SiftData &siftData1, SiftData &siftData2)
{
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;


  vector<int> goodMatches;
  for (int i=0; i<numPts1; i++)
  {
    if (sift1[i].ambiguity < 0.95)
      goodMatches.push_back(i);
  }

  cout << "Initial matches: " << goodMatches.size() << endl;

  // copy matched points into cv vector
  std::vector<cv::Point2d> points1(goodMatches.size());
  std::vector<cv::Point2d> points2(goodMatches.size());

  for(size_t i = 0; i < goodMatches.size(); i++) {
    int i1 = goodMatches[i];
    points1[i] = cv::Point2f(sift1[i1].xpos, sift1[i1].ypos);
    points2[i] = cv::Point2f(sift1[i1].match_xpos, sift1[i1].match_ypos);
  }

  // vectors for rectified points
  std::vector<cv::Point2d> pointsRect1;
  std::vector<cv::Point2d> pointsRect2;

  double fx1 = 769.5519232429078;
  double fy1 = 768.8322015619301;
  double cx1 = 1268.7948261550591;
  double cy1 = 1023.8486413295748;
  double k1  = 0.025513637146558403;
  double k2  = -0.011386600859137145;
  double k3  = 0.013688146542497151;
  double k4  = -0.0076052132438100654;

  double fx2 = 1899.8417;
  double fy2 = 1899.8417;
  double cx2 = 1202.86649;
  double cy2 = 985.03928; 

  cv::Mat K1 = (cv::Mat_<double>(3,3) << fx1, 0.f, cx1, 0.f, fy1, cy1, 0.f, 0.f, 1.f); 
  cv::Mat D1 = (cv::Mat_<double>(1,4) << k1, k2, k3, k4);

  cv::Mat K2 = (cv::Mat_<double>(3,3) << fx2, 0.f, cx2, 0.f, fy2, cy2, 0.f, 0.f, 1.f); 
  cv::Mat D2 = (cv::Mat_<double>(1,4) << 0, 0, 0, 0);

  // rectify points to normalized image coordinates
  cv::fisheye::undistortPoints(points1, pointsRect1, K1, D1);
  cv::undistortPoints(points2, pointsRect2, K2, D2);

  // find essential matrix with 5-point algorithm
  double focal = 1.0;
  cv::Point2d pp(0,0);
  float prob = 0.999;
  float thresh = 0.3/700;
  int method = cv::RANSAC;
  cv::Mat mask;
  cv::Mat essentialMat = cv::findEssentialMat(pointsRect1, pointsRect2, focal, pp, method, prob, thresh, mask);

  vector<int32_t> inliers;

  for(int i = 0; i < mask.rows; i++) {
    if(mask.at<unsigned char>(i)){
      inliers.push_back(i);
    }
  }

  cout << "Inliers: " << inliers.size() << endl;
  cout << "Percentage inliers: " << 100.0f*inliers.size()/goodMatches.size() << "%" << endl;
}

void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography)
{
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;
  int numFound = 0;
#if 1
  homography[0] = homography[4] = -1.0f;
  homography[1] = homography[3] = homography[6] = homography[7] = 0.0f;
  homography[2] = 1279.0f;
  homography[5] = 959.0f;
#endif
  for (int i=0;i<numPts1;i++) {
    float *data1 = sift1[i].data;
    // std::cout << i << ":" << sift1[i].scale << ":" << (int)sift1[i].orientation << " " << sift1[i].xpos << " " << sift1[i].ypos << std::endl;
    std::cout << i << ": " << "score=" << sift1[i].score << "  ambiguity=" << sift1[i].ambiguity << std::endl;
    bool found = false;
    for (int j=0;j<numPts2;j++) {
      float *data2 = sift2[j].data;
      float sum = 0.0f;
      for (int k=0;k<128;k++) 
	sum += data1[k]*data2[k];    
      float den = homography[6]*sift1[i].xpos + homography[7]*sift1[i].ypos + homography[8];
      float dx = (homography[0]*sift1[i].xpos + homography[1]*sift1[i].ypos + homography[2]) / den - sift2[j].xpos;
      float dy = (homography[3]*sift1[i].xpos + homography[4]*sift1[i].ypos + homography[5]) / den - sift2[j].ypos;
      float err = dx*dx + dy*dy;
      if (err<100.0f) // 100.0
	found = true;
      if (err<100.0f || j==sift1[i].match) { // 100.0
	if (j==sift1[i].match && err<100.0f)
	  std::cout << " *";
	else if (j==sift1[i].match) 
	  std::cout << " -";
	else if (err<100.0f)
	  std::cout << " +";
	else
	  std::cout << "  ";
	std::cout << j << ":" << sum << ":" << (int)sqrt(err) << ":" << sift2[j].scale << ":" << (int)sift2[j].orientation << " " << sift2[j].xpos << " " << sift2[j].ypos << " " << (int)dx << " " << (int)dy << std::endl;
      }
    }
    std::cout << std::endl;
    if (found)
      numFound++;
  }
  std::cout << "Number of finds: " << numFound << " / " << numPts1 << std::endl;
  std::cout << homography[0] << " " << homography[1] << " " << homography[2] << std::endl;//%%%
  std::cout << homography[3] << " " << homography[4] << " " << homography[5] << std::endl;//%%%
  std::cout << homography[6] << " " << homography[7] << " " << homography[8] << std::endl;//%%%
}

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img)
{
  int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  float *h_img = img.h_data;
  int w = img.width;
  int h = img.height;
  std::cout << std::setprecision(3);
  for (int j=0;j<numPts;j++) { 
    int k = sift1[j].match;
    if (sift1[j].match_error<5) {
      float dx = sift2[k].xpos - sift1[j].xpos;
      float dy = sift2[k].ypos - sift1[j].ypos;
#if 0
  if (false && sift1[j].xpos>550 && sift1[j].xpos<600) {
  	std::cout << "pos1=(" << (int)sift1[j].xpos << "," << (int)sift1[j].ypos << ") ";
  	std::cout << j << ": " << "score=" << sift1[j].score << "  ambiguity=" << sift1[j].ambiguity << "  match=" << k << "  ";
  	std::cout << "scale=" << sift1[j].scale << "  ";
  	std::cout << "error=" << (int)sift1[j].match_error << "  ";
  	std::cout << "orient=" << (int)sift1[j].orientation << "," << (int)sift2[k].orientation << "  ";
  	std::cout << " delta=(" << (int)dx << "," << (int)dy << ")" << std::endl;
  }
#endif
#if 1
      int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l=0;l<len;l++) {
	int x = (int)(sift1[j].xpos + dx*l/len);
	int y = (int)(sift1[j].ypos + dy*l/len);
	h_img[y*w+x] = 255.0f;
      }
#endif
    }
    int x = (int)(sift1[j].xpos+0.5);
    int y = (int)(sift1[j].ypos+0.5);
    int s = std::min(x, std::min(y, std::min(w-x-2, std::min(h-y-2, (int)(1.41*sift1[j].scale)))));
    int p = y*w + x;
    p += (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] = h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 255.0f;
  }
  std::cout << std::setprecision(6);
}



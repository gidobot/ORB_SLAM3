/*
Copyright 2019. All rights reserved.
Faculty of Engineering of the University of Porto, Portugal

Authors: André Aguiar

libviso2 is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libviso2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libviso2; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

#include "viso_mono_omnidirectional.h"

using namespace std;

namespace viso2 
{
  
VisualOdometryMonoOmnidirectional::VisualOdometryMonoOmnidirectional (parameters param) : param(param), VisualOdometry((VisualOdometry::parameters)param) {}

VisualOdometryMonoOmnidirectional::~VisualOdometryMonoOmnidirectional () {
}

bool VisualOdometryMonoOmnidirectional::process (const cv::Mat& I, bool reset, bool replace)
{
  matcher->pushBackCUDA(I,replace);
  // matcher->matchFeaturesCUDA(0, reset);
  // matcher->bucketFeatures(param.bucket.max_features,param.bucket.bucket_width,param.bucket.bucket_height);                          
  // p_matched = matcher->getMatches();
  // return updateMotion();
  return true;
}

void VisualOdometryMonoOmnidirectional::copyMatchData(std::vector<cv::KeyPoint> &kpts, cv::Mat &desc,
  std::vector<Matcher_SIFT::p_match> &matches_viso, std::vector<cv::DMatch> &matches_cv) {
  matcher->copyMatchData(kpts, desc, matches_viso, matches_cv);
}

std::vector<double> VisualOdometryMonoOmnidirectional::estimateMotion (std::vector<Matcher_SIFT::p_match> p_matched)
{
  uint32_t N = p_matched.size();
  if (N<10)
    return vector<double>();

  // copy matched points into cv vector
  std::vector<cv::Point2d> points1(p_matched.size());
  std::vector<cv::Point2d> points2(p_matched.size());

  for(size_t i = 0; i < p_matched.size(); i++) {
    points1[i] = cv::Point2f(p_matched[i].u1p, p_matched[i].v1p);
    points2[i] = cv::Point2f(p_matched[i].u1c, p_matched[i].v1c);
  }

  // vectors for rectified points
  std::vector<cv::Point2d> points1_rect;
  std::vector<cv::Point2d> points2_rect;

  double fx = param.omnidirectional_calib.fx;
  double fy = param.omnidirectional_calib.fy;
  double cx = param.omnidirectional_calib.cx;
  double cy = param.omnidirectional_calib.cy;
  double k1 = param.omnidirectional_calib.k[0];
  double k2 = param.omnidirectional_calib.k[1];
  double k3 = param.omnidirectional_calib.k[2];
  double k4 = param.omnidirectional_calib.k[3];

  // double cameraVec[9] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
  // cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64F, cameraVec);
  cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f); 
  // float distVec[4];
  // cv::Mat dist = cv::Mat(1, 4, CV_64F, distVec);
  cv::Mat dist = (cv::Mat_<double>(1,4) << k1, k2, k3, k4);

  // rectify points to normalized image coordinates
  cv::fisheye::undistortPoints(points1, points1_rect, cameraMatrix, dist);
  cv::fisheye::undistortPoints(points2, points2_rect, cameraMatrix, dist);

  // find essential matrix with 5-point algorithm
  double focal = 1.0;
  cv::Point2d pp(0,0);
  float prob = 0.999;
  float thresh = 0.3/fx;
  int method = cv::RANSAC;
  cv::Mat mask;
  cv::Mat essentialMat = cv::findEssentialMat(points1_rect, points2_rect, focal, pp, method, prob, thresh, mask);

  // recover motion from essential matrix
  cv::Mat t;
  cv::Mat R;
  int num_inliers = cv::recoverPose(essentialMat, points1_rect, points2_rect, R, t, focal, pp, mask);

  inliers.clear();

  if (num_inliers < 10)
    return vector<double>();

  for(int i = 0; i < mask.rows; i++) {
    if(mask.at<unsigned char>(i)){
      inliers.push_back(i);
    }
  }

  // compute rotation angles
  double ry = asin(R.at<double>(0,2));
  double rx = asin(-R.at<double>(1,2)/cos(ry));
  double rz = asin(-R.at<double>(0,1)/cos(ry));
  
  // return parameter vector
  vector<double> tr_delta;
  tr_delta.resize(6);
  tr_delta[0] = rx;
  tr_delta[1] = ry;
  tr_delta[2] = rz;
  tr_delta[3] = t.at<double>(0,0);
  tr_delta[4] = t.at<double>(0,1);
  tr_delta[5] = t.at<double>(0,2);
  return tr_delta;

}

// std::vector<double> VisualOdometryMonoOmnidirectional::estimateMotion (std::vector<Matcher_SIFT::p_match> p_matched)
// {
//   uint32_t N = p_matched.size();
//   if (N<10)
//     return vector<double>();

//   // project 2d matches into the unit sphere
//   vector<Matcher_SIFT::p_match_3d> p_matched_3d = projectMatches(p_matched);

//   // RANSAC to estimate F
//   Matrix E,F;
//   inliers.clear();

//   for (int32_t k=0;k<param.ransac_iters;k++) {
//     // draw random sample set
//     vector<int32_t> active = getRandomSample(N,8);

//     // estimate fundamental matrix and get inliers
//     fundamentalMatrix(p_matched_3d,active,F);
//     vector<int32_t> inliers_curr = getInlier(p_matched_3d,F);

//     // update model if we are better
//     if (inliers_curr.size()>inliers.size())
//       inliers = inliers_curr;
//   }
  
//   // are there enough inliers?
//   if (inliers.size()<10)
//     return vector<double>();

//   // refine F using all inliers
//   fundamentalMatrix(p_matched_3d,inliers,F); 
//   E = F;
  
//   // re-enforce rank 2 constraint on essential matrix
//   Matrix U,W,V;
//   E.svd(U,W,V);
//   W.val[2][0] = 0;
//   E = U*Matrix::diag(W)*~V;

//   // compute 3d points X and R|t up to scale
//   Matrix X,R,t;
//   EtoRt(E,p_matched_3d,X,R,t);

//   // normalize 3d points and remove points behind image plane
//   X = X/X.getMat(3,0,3,-1);
//   vector<int32_t> pos_idx;
//   for (int32_t i=0; i<X.n; i++)
//     if (X.val[2][i]>0)
//       pos_idx.push_back(i);
//   Matrix X_plane = X.extractCols(pos_idx);

//   // we need at least 10 points to proceed
//   if (X_plane.n<10)
//     return vector<double>();
  
//   // get elements closer than median
//   double median;
//   smallerThanMedian(X_plane,median);
  
//   // return error on large median (litte motion)
//   if (median>param.motion_threshold)
//     return vector<double>();
  
//   // project features to 2d
//   Matrix x_plane(2,X_plane.n);
//   x_plane.setMat(X_plane.getMat(1,0,2,-1),0,0);
  
//   Matrix n(2,1);
//   n.val[0][0]       = cos(-param.pitch);
//   n.val[1][0]       = sin(-param.pitch);
//   Matrix   d        = ~n*x_plane;
//   double   sigma    = median/50.0;
//   double   weight   = 1.0/(2.0*sigma*sigma);
//   double   best_sum = 0;
//   int32_t  best_idx = 0;

//   // find best plane
//   for (int32_t i=0; i<x_plane.n; i++) {
//     if (d.val[0][i]>median/param.motion_threshold) {
//       double sum = 0;
//       for (int32_t j=0; j<x_plane.n; j++) {
//         double dist = d.val[0][j]-d.val[0][i];
//         sum += exp(-dist*dist*weight);
//       }
//       if (sum>best_sum) {
//         best_sum = sum;
//         best_idx = i;
//       }
//     }
//   }
//   t = t*param.height/d.val[0][best_idx];
  
//   // compute rotation angles
//   double ry = asin(R.val[0][2]);
//   double rx = asin(-R.val[1][2]/cos(ry));
//   double rz = asin(-R.val[0][1]/cos(ry));
  
//   // return parameter vector
//   vector<double> tr_delta;
//   tr_delta.resize(6);
//   tr_delta[0] = rx;
//   tr_delta[1] = ry;
//   tr_delta[2] = rz;
//   tr_delta[3] = t.val[0][0];
//   tr_delta[4] = t.val[1][0];
//   tr_delta[5] = t.val[2][0];
//   return tr_delta;
// }

Matrix VisualOdometryMonoOmnidirectional::smallerThanMedian (Matrix &X,double &median)
{
  // set distance and index vector
  vector<double> dist;
  vector<int32_t> idx;
  for (int32_t i=0; i<X.n; i++) {
    dist.push_back(fabs(X.val[0][i])+fabs(X.val[1][i])+fabs(X.val[2][i]));
    idx.push_back(i);
  }
  
  // sort elements
  sort(idx.begin(),idx.end(),idx_cmp<vector<double>&>(dist));
  
  // get median
  int32_t num_elem_half = idx.size()/2;
  median = dist[idx[num_elem_half]];
  
  // create matrix containing elements closer than median
  Matrix X_small(4,num_elem_half+1);
  for (int32_t j=0; j<=num_elem_half; j++)
    for (int32_t i=0; i<4; i++)
      X_small.val[i][j] = X.val[i][idx[j]];
  return X_small;
}

void VisualOdometryMonoOmnidirectional::fundamentalMatrix (const std::vector<Matcher_SIFT::p_match_3d> &p_matched_3d,const std::vector<int32_t> &active,Matrix &F)
{
  // number of active p_matched
  int32_t N = active.size();
  
  // create constraint matrix A
  Matrix A(N,9);
  for (int32_t i=0; i<N; i++) {
    Matcher_SIFT::p_match_3d m = p_matched_3d[active[i]];
    A.val[i][0] = m.xc*m.xp;
    A.val[i][1] = m.xc*m.yp;
    A.val[i][2] = m.xc*m.zp;
    A.val[i][3] = m.yc*m.xp;
    A.val[i][4] = m.yc*m.yp;
    A.val[i][5] = m.yc*m.zp;
    A.val[i][6] = m.xp*m.zc;
    A.val[i][7] = m.yp*m.zc;
    A.val[i][8] = m.zc*m.zp;
  }
   
  // compute singular value decomposition of A
  Matrix U,W,V;
  A.svd(U,W,V);
   
  // extract fundamental matrix from the column of V corresponding to the smallest singular value
  F = Matrix::reshape(V.getMat(0,8,8,8),3,3);
  
  // enforce rank 2
  F.svd(U,W,V);
  W.val[2][0] = 0;
  F = U*Matrix::diag(W)*~V;
}

std::vector<int32_t> VisualOdometryMonoOmnidirectional::getInlier (std::vector<Matcher_SIFT::p_match_3d> &p_matched_3d,Matrix &F)
{
  // extract fundamental matrix
  double f00 = F.val[0][0]; double f01 = F.val[0][1]; double f02 = F.val[0][2];
  double f10 = F.val[1][0]; double f11 = F.val[1][1]; double f12 = F.val[1][2];
  double f20 = F.val[2][0]; double f21 = F.val[2][1]; double f22 = F.val[2][2];
  
  // loop variables
  double u1,v1,w1,u2,v2,w2;
  double x2tFx1;
  double Fx1u,Fx1v,Fx1w;
  double Ftx2u,Ftx2v;
  
  // vector with inliers
  vector<int32_t> inliers;
  
  // for all matches do
  for (int32_t i=0; i<(int32_t)p_matched_3d.size(); i++) {
    // extract matches
    u1 = p_matched_3d[i].xp;
    v1 = p_matched_3d[i].yp;
    w1 = p_matched_3d[i].zp;
    u2 = p_matched_3d[i].xc;
    v2 = p_matched_3d[i].yc;
    w2 = p_matched_3d[i].zc;
    
    // F*x1
    Fx1u = f00*u1+f01*v1+f02*w1;
    Fx1v = f10*u1+f11*v1+f12*w1;
    Fx1w = f20*u1+f21*v1+f22*w1;
    
    // F'*x2
    Ftx2u = f00*u2+f10*v2+f20*w2;
    Ftx2v = f01*u2+f11*v2+f21*w2;
    
    // x2'*F*x1
    x2tFx1 = u2*Fx1u+v2*Fx1v+w2*Fx1w;
    
    // sampson distance
    double d = x2tFx1*x2tFx1 / (Fx1u*Fx1u+Fx1v*Fx1v+Ftx2u*Ftx2u+Ftx2v*Ftx2v);
    
    // check threshold
    if (fabs(d)<param.inlier_threshold)
      inliers.push_back(i);
  }

  // return set of all inliers
  return inliers;
}

void VisualOdometryMonoOmnidirectional::EtoRt(Matrix &E,std::vector<Matcher_SIFT::p_match_3d> &p_matched,Matrix &X,Matrix &R,Matrix &t)
{

  // hartley matrices
  double W_data[9] = {0,-1,0,+1,0,0,0,0,1};
  double Z_data[9] = {0,+1,0,-1,0,0,0,0,0};
  Matrix W(3,3,W_data);
  Matrix Z(3,3,Z_data); 
  
  // extract T,R1,R2 (8 solutions)
  Matrix U,S,V;
  E.svd(U,S,V);
  Matrix T  = U*Z*~U;
  Matrix Ra = U*W*(~V);
  Matrix Rb = U*(~W)*(~V);
  
  // convert T to t
  t = Matrix(3,1);
  t.val[0][0] = T.val[2][1];
  t.val[1][0] = T.val[0][2];
  t.val[2][0] = T.val[1][0];
  
  // assure determinant to be positive
  if (Ra.det()<0) Ra = -Ra;
  if (Rb.det()<0) Rb = -Rb;
  
  // create vector containing all 4 solutions
  vector<Matrix> R_vec;
  vector<Matrix> t_vec;
  R_vec.push_back(Ra); t_vec.push_back( t);
  R_vec.push_back(Ra); t_vec.push_back(-t);
  R_vec.push_back(Rb); t_vec.push_back( t);
  R_vec.push_back(Rb); t_vec.push_back(-t);
  
  // try all 4 solutions
  Matrix X_curr;
  int32_t max_inliers = 0;
  for (int32_t i=0; i<4; i++) {
    int32_t num_inliers = triangulateChieral(p_matched,R_vec[i],t_vec[i],X_curr);
    if (num_inliers>max_inliers) {
      max_inliers = num_inliers;
      X = X_curr;
      R = R_vec[i];
      t = t_vec[i];
    }
  }
}

int32_t VisualOdometryMonoOmnidirectional::triangulateChieral (std::vector<Matcher_SIFT::p_match_3d> &p_matched_3d,Matrix &R,Matrix &t,Matrix &X) {
  
  // init 3d point matrix
  X = Matrix(4,p_matched.size());
  
  // projection matrices
  double P_[12] = {1,0,0,0,0,1,0,0,0,0,1,0};
  Matrix P1(3,4,P_);
  Matrix P2(3,4);
  P2.setMat(R,0,0);
  P2.setMat(t,0,3);
  
  // triangulation via orthogonal regression
  Matrix J(6,4);
  Matrix U,S,V;
  for (int32_t i=0; i<(int)p_matched.size(); i++) {
    for (int32_t j=0; j<4; j++) {
      J.val[0][j] = P1.val[2][j]*p_matched_3d[i].xp - P1.val[0][j]*p_matched_3d[i].zp;
      J.val[1][j] = P1.val[1][j]*p_matched_3d[i].xp - P1.val[0][j]*p_matched_3d[i].yp;
      J.val[2][j] = P1.val[2][j]*p_matched_3d[i].yp - P1.val[1][j]*p_matched_3d[i].zp;
      J.val[3][j] = P2.val[2][j]*p_matched_3d[i].xc - P2.val[0][j]*p_matched_3d[i].zc;
      J.val[4][j] = P2.val[1][j]*p_matched_3d[i].xc - P2.val[0][j]*p_matched_3d[i].yc;
      J.val[5][j] = P2.val[2][j]*p_matched_3d[i].yc - P2.val[1][j]*p_matched_3d[i].zc;
    }
    J.svd(U,S,V);
    X.setMat(V.getMat(0,3,3,3),0,i);
  }
  
  // compute inliers
  Matrix  AX1 = P1*X;
  Matrix  BX1 = P2*X;
  int32_t num = 0;
  for (int32_t i=0; i<X.n; i++)
    if (AX1.val[2][i]*X.val[3][i]>0 && BX1.val[2][i]*X.val[3][i]>0)
      num++;
  
  // return number of inliers
  return num;
}

std::vector<Matcher_SIFT::p_match_3d> VisualOdometryMonoOmnidirectional::projectMatches(std::vector<Matcher_SIFT::p_match> p_matched)
{
  std::vector<Matcher_SIFT::p_match_3d> matches_3d(p_matched.size());
  for(size_t i = 0; i < p_matched.size(); i++)
    matches_3d[i] = projectIntoUnitSphere(p_matched[i]);

  return matches_3d;
}

std::vector<Matcher_SIFT::p_match> VisualOdometryMonoOmnidirectional::reprojectMatches(Matrix X, Matrix P1, Matrix P2)
{
  std::vector<Matcher_SIFT::p_match> matches(X.n);
  for(int32_t i = 0; i < X.n; i++)
    matches[i] = projectIntoImage(X.getMat(0,i,3,i),P1,P2);

  return matches;
}

Matcher_SIFT::p_match_3d VisualOdometryMonoOmnidirectional::projectIntoUnitSphere(Matcher_SIFT::p_match p_matched)
{
  struct Matcher_SIFT::p_match_3d match_3d;

  double p1[2] = {p_matched.u1p,p_matched.v1p};
  double p2[2] = {p_matched.u1c,p_matched.v1c};
  double p1_[3];
  double p2_[3];

  cam2world(p1_,p1);
  cam2world(p2_,p2);

  match_3d.xp = p1_[0];
  match_3d.yp = p1_[1];
  match_3d.zp = p1_[2];
  match_3d.xc = p2_[0];
  match_3d.yc = p2_[1];
  match_3d.zc = p2_[2];

  return match_3d;
}

Matcher_SIFT::p_match VisualOdometryMonoOmnidirectional::projectIntoImage(Matrix X, Matrix P1, Matrix P2)
{
  Matcher_SIFT::p_match p_matched;
  Matrix p1 = P1*X;
  Matrix p2 = P2*X;

  double p1_[3] = {p1.val[0][0],p1.val[1][0],p1.val[2][0]};
  double p2_[3] = {p2.val[0][0],p2.val[1][0],p2.val[2][0]};
  double _p1[2];
  double _p2[2];

  world2cam(_p1,p1_);
  world2cam(_p2,p2_);

  p_matched.u1p = _p1[0];
  p_matched.v1p = _p1[1];
  p_matched.u1c = _p2[0];
  p_matched.v1c = _p2[1];

  return p_matched;
}

void VisualOdometryMonoOmnidirectional::world2cam(double point2D[2], double point3D[3]) 
{
  double *invpol     = param.omnidirectional_calib.invpol;
  double xc          = param.omnidirectional_calib.xc;
  double yc          = param.omnidirectional_calib.yc;
  double c           = param.omnidirectional_calib.c;
  double d           = param.omnidirectional_calib.d;
  double e           = param.omnidirectional_calib.e;
  int    width       = param.omnidirectional_calib.width;
  int    height      = param.omnidirectional_calib.height;
  int length_invpol  = param.omnidirectional_calib.length_invpol;

  double norm        = sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1]);
  double theta       = atan(point3D[2]/norm);
  double t, t_i;
  double rho, x, y;
  double invnorm;
  int i;

  if (norm != 0) {
    invnorm = 1/norm;
    t  = theta;
    rho = invpol[0];
    t_i = 1;

    for (i = 1; i < length_invpol; i++) {
      t_i *= t;
      rho += t_i*invpol[i];
    }

    x = point3D[0]*invnorm*rho;
    y = point3D[1]*invnorm*rho;

    point2D[0] = x*c + y*d + xc;
    point2D[1] = x*e + y   + yc;
  }
  else {
    point2D[0] = xc;
    point2D[1] = yc;
  }
}

void VisualOdometryMonoOmnidirectional::cam2world(double point3D[3], double point2D[2])
{
  double *pol    = param.omnidirectional_calib.pol;
  double xc      = param.omnidirectional_calib.xc;
  double yc      = param.omnidirectional_calib.yc;
  double c       = param.omnidirectional_calib.c;
  double d       = param.omnidirectional_calib.d;
  double e       = param.omnidirectional_calib.e;
  int length_pol = param.omnidirectional_calib.length_pol;

  double invdet  = 1/(c-d*e); // 1/det(A), where A = [c,d;e,1] as in the Matlab file

  double xp = invdet*(    (point2D[0] - xc) - d*(point2D[1] - yc) );
  double yp = invdet*( -e*(point2D[0] - xc) + c*(point2D[1] - yc) );
 
  double r   = sqrt(  xp*xp + yp*yp ); //distance [pixels] of  the point from the image center
  double zp  = pol[0];
  double r_i = 1;
  int i;

  for (i = 1; i < length_pol; i++) {
    r_i *= r;
    zp  += r_i*pol[i];
  }

  //normalize to unit norm
  double invnorm = 1/sqrt( xp*xp + yp*yp + zp*zp );
 
  point3D[0] = invnorm*xp;
  point3D[1] = invnorm*yp;
  point3D[2] = invnorm*zp;
}

}
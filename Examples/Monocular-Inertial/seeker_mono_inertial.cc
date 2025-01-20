/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <string>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <filesystem>

#include <opencv2/core/core.hpp>

#include <System.h>
#include "ImuTypes.h"

using namespace std;
namespace fs = std::filesystem;


void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

double get_image_timestamp(const std::string& image_name);

double ttrack_tot = 0;
int main(int argc, char *argv[])
{

    if(argc != 5)
    {
        cerr << endl << "Usage: ./seeker_mono_inertial path_to_vocabulary path_to_settings path_to_image_folder path_to_imu_file" << endl;
        return 1;
    }

    // Load sequence:
    vector<string> vstrImageFilenames;
    vector<double> vTimestampsCam;
    vector<cv::Point3f> vAcc, vGyro;
    vector<double> vTimestampsImu;
    int nImages;
    int nImu;
    int first_imu = 0;

    int tot_images = 0;

    cout << "Loading images...";

    string pathSeq(argv[3]);
    string pathImu(argv[4]);

    LoadImages(pathSeq, vstrImageFilenames, vTimestampsCam);
    cout << "LOADED!" << endl;

    cout << "Loading IMU...";
    LoadIMU(pathImu, vTimestampsImu, vAcc, vGyro);
    cout << "LOADED!" << endl;

    nImages = vstrImageFilenames.size();
    tot_images += nImages;
    nImu = vTimestampsImu.size();

    if((nImages<=0)||(nImu<=0))
    {
        cerr << "ERROR: Failed to load images or IMU" << endl;
        return 1;
    }

    // Find first imu to be considered, supposing imu measurements start first

    while(vTimestampsImu[first_imu]<=vTimestampsCam[0]) {
        if (first_imu < 100)
            cout << "diff: " << vTimestampsCam[0] - vTimestampsImu[first_imu] << endl;
        first_imu++;
    }
    first_imu--; // first imu measurement to be considered

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout.precision(17);

    cout << "Number of images: " << nImages << endl;
    cout << "Number of IMU measurements: " << nImu << endl;
    cout << "First imu: " << first_imu << endl;

//     // Create SLAM system. It initializes all system threads and gets ready to process frames.
//     ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_MONOCULAR, true);
//     float imageScale = SLAM.GetImageScale();

//     double t_resize = 0.f;
//     double t_track = 0.f;

//     int proccIm=0;
//     for (seq = 0; seq<num_seq; seq++)
//     {

//         // Main loop
//         cv::Mat im;
//         vector<ORB_SLAM3::IMU::Point> vImuMeas;
//         proccIm = 0;
//         for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
//         {
//             // Read image from file
//             im = cv::imread(vstrImageFilenames[seq][ni],cv::IMREAD_UNCHANGED); //CV_LOAD_IMAGE_UNCHANGED);

//             double tframe = vTimestampsCam[seq][ni];

//             if(im.empty())
//             {
//                 cerr << endl << "Failed to load image at: "
//                      <<  vstrImageFilenames[seq][ni] << endl;
//                 return 1;
//             }

//             if(imageScale != 1.f)
//             {
// #ifdef REGISTER_TIMES
//                 std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
// #endif
//                 int width = im.cols * imageScale;
//                 int height = im.rows * imageScale;
//                 cv::resize(im, im, cv::Size(width, height));
// #ifdef REGISTER_TIMES
//                 std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();

//                 t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
//                 SLAM.InsertResizeTime(t_resize);
// #endif
//             }

//             // Load imu measurements from previous frame
//             vImuMeas.clear();

//             if(ni>0)
//             {
//                 // cout << "t_cam " << tframe << endl;

//                 while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni])
//                 {
//                     vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x,vAcc[seq][first_imu[seq]].y,vAcc[seq][first_imu[seq]].z,
//                                                              vGyro[seq][first_imu[seq]].x,vGyro[seq][first_imu[seq]].y,vGyro[seq][first_imu[seq]].z,
//                                                              vTimestampsImu[seq][first_imu[seq]]));
//                     first_imu[seq]++;
//                 }
//             }

//             std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

//             // Pass the image to the SLAM system
//             // cout << "tframe = " << tframe << endl;
//             SLAM.TrackMonocular(im,tframe,vImuMeas); // TODO change to monocular_inertial

//             std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

// #ifdef REGISTER_TIMES
//             t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
//             SLAM.InsertTrackTime(t_track);
// #endif

//             double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//             ttrack_tot += ttrack;
//             // std::cout << "ttrack: " << ttrack << std::endl;

//             vTimesTrack[ni]=ttrack;

//             // Wait to load the next frame
//             double T=0;
//             if(ni<nImages[seq]-1)
//                 T = vTimestampsCam[seq][ni+1]-tframe;
//             else if(ni>0)
//                 T = tframe-vTimestampsCam[seq][ni-1];

//             if(ttrack<T)
//                 usleep((T-ttrack)*1e6); // 1e6
//         }
//         if(seq < num_seq - 1)
//         {
//             cout << "Changing the dataset" << endl;

//             SLAM.ChangeDataset();
//         }
//     }

//     // Stop all threads
//     SLAM.Shutdown();

//     // Save camera trajectory
//     if (bFileName)
//     {
//         const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
//         const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
//         SLAM.SaveTrajectoryEuRoC(f_file);
//         SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
//     }
//     else
//     {
//         SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
//         SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
//     }

    return 0;
}

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);

    // Check if the folder exists
    if (!fs::exists(strImagePath) || !fs::is_directory(strImagePath)) {
        throw std::runtime_error("Invalid folder path: " + strImagePath);
    }

    // Iterate over files in the directory
    for (const auto& entry : fs::directory_iterator(strImagePath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            vstrImages.push_back(entry.path().filename().string());
        }
    }

    // Sort the filenames in lexicographical order
    std::sort(vstrImages.begin(), vstrImages.end());
    for (const auto& image : vstrImages) {
        vTimeStamps.push_back(get_image_timestamp(image));
    }
}

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    std::ifstream file(strImuPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + strImuPath);
    }

    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    std::string line;

    // Read the header line and ignore it
    if (!std::getline(file, line)) {
        throw std::runtime_error("File is empty or invalid format.");
    }

    // Read each subsequent line
    while (std::getline(file, line)) {
        std::istringstream line_stream(line);
        std::string cell;

        // Parse timestamp
        std::getline(line_stream, cell, ',');
        double timestamp = std::stoll(cell);

        // Parse gyroscope data (w_x, w_y, w_z)
        std::array<float, 3> gyro{};
        for (int i = 0; i < 3; ++i) {
            if (!std::getline(line_stream, cell, ',')) {
                throw std::runtime_error("Missing gyroscope data in line: " + line);
            }
            gyro[i] = std::stod(cell);
        }

        // Parse acceleration data (a_x, a_y, a_z)
        std::array<float, 3> accel{};
        for (int i = 0; i < 3; ++i) {
            if (!std::getline(line_stream, cell, ',')) {
                throw std::runtime_error("Missing acceleration data in line: " + line);
            }
            accel[i] = std::stod(cell);
        }

        // Store parsed data into vectors
        vTimeStamps.push_back(timestamp/1e6);
        vAcc.push_back(cv::Point3f(accel[0], accel[1], accel[2]));
        vGyro.push_back(cv::Point3f(gyro[0], gyro[1], gyro[2]));
    }
}

double get_image_timestamp(const std::string& image_name) {
    // Split the input string by '_'
    std::istringstream iss(image_name);
    std::string token;
    std::vector<std::string> utc_time_list;
    while (std::getline(iss, token, '_')) {
        utc_time_list.push_back(token);
    }

    // Construct the UTC time string
    std::string utc_time_str = utc_time_list[1] + "_" + utc_time_list[2];

    // Parse the UTC time string into a time structure
    std::tm utc_time = {};
    std::istringstream time_stream(utc_time_str);
    time_stream >> std::get_time(&utc_time, "%Y%m%d_%H%M%S");
    if (time_stream.fail()) {
        throw std::runtime_error("Failed to parse UTC time string.");
    }

    // Convert to time_point and get the timestamp in microseconds
    std::chrono::system_clock::time_point time_point =
        std::chrono::system_clock::from_time_t(std::mktime(&utc_time));

    // Add the millisecond part from utc_time_list[3]
    double milliseconds = std::stoi(utc_time_list[3]);
    double microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(time_point.time_since_epoch()).count();
    double seconds = microseconds/1e6 + milliseconds/1e3;
    seconds -= 60*60*10; // Add 10 hours to convert to UTC+0

    return seconds;
}
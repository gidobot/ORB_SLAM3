/**
* This file is part of SIFT-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* SIFT-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* SIFT-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with SIFT-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <filesystem>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;
namespace fs = std::filesystem;

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps);
double get_image_timestamp(const std::string& image_name);

int main(int argc, char **argv)
{  
    if(argc != 4)
    {
        cerr << endl << "Usage: ./seeker_mono path_to_vocabulary path_to_settings path_to_image_folder" << endl;
        return 1;
    }

    // Load all sequences:
    vector<string> vstrImageFilenames;
    vector<double> vTimestampsCam;
    int nImages;

    int tot_images = 0;

    cout << "Loading images...";
    string pathSeq(argv[3]);

    LoadImages(pathSeq, vstrImageFilenames, vTimestampsCam);
    cout << "LOADED!" << endl;

    nImages = vstrImageFilenames.size();
    tot_images += nImages;

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);


    int fps = 2;
    float dT = 1.f/fps;
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    SIFT_SLAM3::System SLAM(argv[1],argv[2],SIFT_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    // Main loop
    cv::Mat im;
    int proccIm = 0;
    for(int ni=485; ni<nImages; ni++, proccIm++)
    {

        // Read image from file
        im = cv::imread(string(pathSeq) + '/' + vstrImageFilenames[ni],cv::IMREAD_UNCHANGED); //CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 <<  vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();

            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        // cout << "tframe = " << tframe << endl;
        SLAM.TrackMonocular(im,tframe); // TODO change to monocular_inertial

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestampsCam[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestampsCam[ni-1];

        //std::cout << "T: " << T << std::endl;
        //std::cout << "ttrack: " << ttrack << std::endl;

        if(ttrack<T) {
            //std::cout << "usleep: " << (dT-ttrack) << std::endl;
            usleep((T-ttrack)*1e6); // 1e6
        }
    }

    // if(seq < num_seq - 1)
    // {
    //     string kf_file_submap =  "./SubMaps/kf_SubMap_" + std::to_string(seq) + ".txt";
    //     string f_file_submap =  "./SubMaps/f_SubMap_" + std::to_string(seq) + ".txt";
    //     SLAM.SaveTrajectoryEuRoC(f_file_submap);
    //     SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file_submap);

    //     cout << "Changing the dataset" << endl;

    //     SLAM.ChangeDataset();
    // }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    // if (bFileName)
    // {
    //     const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
    //     const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
    //     SLAM.SaveTrajectoryEuRoC(f_file);
    //     SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    // }
    // else
    // {
    //     SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
    //     SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    // }

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

    return seconds;
}
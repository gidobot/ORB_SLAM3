/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./stereo_uw path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    SIFT_SLAM3::System SLAM(argv[1],argv[2],SIFT_SLAM3::System::STEREO,true); // Final bool for using viewer
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    double t_track = 0.f;
    double t_resize = 0.f;

    // Main loop
    cv::Mat imLeft, imRight;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],cv::IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#endif
            int width = imLeft.cols * imageScale;
            int height = imLeft.rows * imageScale;
            cv::resize(imLeft, imLeft, cv::Size(width, height));
            cv::resize(imRight, imRight, cv::Size(width, height));
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();

            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft,imRight,tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        cout << "Total time to track frame " << ni << ": " << ttrack << endl;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Pause for viewing map
    do 
    {
        cout << '\n' << "Trajectory finished. Press a key to continue...";
    } while (cin.get() != '\n');

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    // SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/images/";
    string strPrefixRight = strPathToSequence + "/images/";

    const int nTimes = vTimestamps.size();
    // const int nTimes = 100;
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        // ss << setfill('0') << setw(4) << i;
        ss << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + "_0.png";
        vstrImageRight[i] = strPrefixRight + ss.str() + "_1.png";
        // vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        // vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

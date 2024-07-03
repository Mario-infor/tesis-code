#ifndef __STRUCTSFILE__
#define __STRUCTSFILE__

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// Struct to store information about each frame saved.
struct CameraInput
{
    int index;
    int time;
    cv::Mat frame;

    CameraInput &operator=(const CameraInput &other)
    {
        if (this != &other)
        {
            index = other.index;
            time = other.time;
            frame = other.frame.clone();
        }
        return *this;
    }
};

// Struct to store information about each IMU data saved (Jetson Board).
struct ImuInputJetson
{
    int index;
    int time;
    Eigen::Vector3d gyroVect;
    Eigen::Vector3d eulerVect;
    Eigen::Quaterniond rotQuat;
    Eigen::Vector3d accVect;
    Eigen::Vector3d gravVect;
};

// Struct to store a list of rvects and tvects.
struct FrameMarkersData
{
    std::vector<int> markerIds;
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;
    std::vector<cv::Vec4d> qvecs;
};

struct CameraInterpolatedData
{
    int originalOrNot; // 0 = original, 1 = interpolated.
    CameraInput frame;
    FrameMarkersData frameMarkersData;
};

struct TransformBetweenMarkers
{
    int baseMarkerId;
    int secundaryMarkerId;
    Eigen::Matrix4d G;
};

#endif
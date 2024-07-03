#ifndef __STRUCTSFILE__
#define __STRUCTSFILE__

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

/**
 * @brief Struct to store information about each camera frame saved. It stores the index, time and frame of a 
 * measurement from the camera. It also overwrites the = operator to make it easier to make a copy of the struct
 * if necessary.
*/
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

/**
 * @brief Struct to store information about each IMU measurement saved. It stores the index, time, gyro, euler, quaternion,
 * acceleration and gravity data from one IMU measurement. 
*/
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

/**
 * @brief Struct to store information about each rotation and traslation from all markes detected on one image frame.
 * It stores the marker ids, rotation vectors, traslation vectors and quaternion vectors from all markers detected on
 * one frame. 
*/
struct FrameMarkersData
{
    std::vector<int> markerIds;
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;
    std::vector<cv::Vec4d> qvecs;
};

/**
 * @brief Struct to store information about the transformation between two markers. It stores the base marker id, secundary
 * marker id and the transformation matrix from the secundary marker to the base marker. 
*/
struct TransformBetweenMarkers
{
    int baseMarkerId;
    int secundaryMarkerId;
    Eigen::Matrix4d G;
};

#endif
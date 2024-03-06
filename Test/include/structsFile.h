#ifndef __STRUCTSFILE__
#define __STRUCTSFILE__

#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

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
    glm::vec3 gyroVect;
    glm::vec3 eulerVect;
    glm::quat rotQuat;
    glm::vec3 accVect;
    glm::vec3 gravVect;
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

#endif
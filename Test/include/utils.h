#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <glm/gtc/quaternion.hpp>
#include <structsFile.h>

#define IMU_ADDRESS "/dev/i2c-1" // Address of the IMU sensor.
#define	MATH_PI					3.1415926535 // Definition of variable pi.
#define	MATH_DEGREE_TO_RAD		(MATH_PI / 180.0) // Conversion from degrees to radians.
#define	MATH_RAD_TO_DEGREE		(180.0 / MATH_PI) // Conversion from radians to degrees.

// Convert rotation vector to quaternion.
glm::quat convertOpencvRotVectToQuat(cv::Vec3d rotVect);

// Convert quaternion to rotation vector.
cv::Vec3d convertQuatToOpencvRotVect(glm::quat quaternion);

// Create a hard copy of camera vector.
std::vector<CameraInput> hardCopyCameraVector(
    std::vector<CameraInput> cameraReadVector);

// Get rotation and translation from all frames.
std::vector<FrameMarkersData> getRotationTraslationFromAllFrames(
    std::vector<CameraInput> cameraReadVector,
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs);

// Get rotation and translation from one single frame.
FrameMarkersData getRotationTraslationFromFrame(
    CameraInput frame, 
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs);

// Prints Euler or Accel data from the IMU to the console.
void printIMUData();

#endif // UTILS_H
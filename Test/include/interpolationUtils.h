#ifndef INTERPOLATIONUTILS_H
#define INTERPOLATIONUTILS_H

#include <vector>
#include <glm/gtc/quaternion.hpp>
#include <structsFile.h>

// Create spline points.
std::vector<glm::vec3> createSplinePoint(
    std::vector<ImuInputJetson> imuReadVector);

// Create slerp points for quaternions (tests at home).
std::vector<glm::quat> createSlerpPoint(std::vector<ImuInputJetson> imuReadVector);

// Tests Slerp and Spline methods.
void testSlerpAndSpline(
    std::vector<ImuInputJetson> imuReadVector,
    std::vector<CameraInput> cameraReadVector);

// Interpolate camera rotation to fit IMU data.
std::vector<CameraInterpolatedData> interpolateCameraRotation(const std::vector<ImuInputJetson> imuReadVectorCopy,
                                                              const std::vector<CameraInput> cameraReadVectorCopy,
                                                              std::vector<FrameMarkersData> frameMarkersDataVector);

// Test interpolate camera rotation to fit IMU data.
void testInterpolateCamera(
    std::vector<CameraInterpolatedData> interpolatedPoints,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs);

#endif // INTERPOLATIONUTILS_H
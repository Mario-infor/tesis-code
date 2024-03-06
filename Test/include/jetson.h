#ifndef JETSON_H
#define JETSON_H

#include <readWriteData.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/spline.hpp>
#include <glm/gtc/quaternion.hpp>


// Amount of IMU data and frames to read from devices.
//#define RING_BUFFER_LENGTH_CAMERA 1875
//#define RING_BUFFER_LENGTH_CAMERA 3750

#define RING_BUFFER_LENGTH_CAMERA 150
#define RING_BUFFER_LENGTH_IMU 300
#define	MATH_PI					3.1415926535 // Definition of variable pi.
#define	MATH_DEGREE_TO_RAD		(MATH_PI / 180.0) // Conversion from degrees to radians.
#define	MATH_RAD_TO_DEGREE		(180.0 / MATH_PI) // Conversion from radians to degrees.
#define IMU_ADDRESS				"/dev/i2c-1" // Address of the IMU sensor.

// Global variables that need to be accessed from different threads or methods.
std::mutex myMutex;
std::chrono::time_point<std::chrono::steady_clock> timeCameraStart;
std::chrono::time_point<std::chrono::steady_clock> timeIMUStart;
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

bool doneCalibrating = false;

// Pipeline for camera on Jetson Board.
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// Thread in charge of readng data from camera and store it on camera buffer.
void cameraCaptureThread();

// Method to calibrate de IMU sensors.
void imuCalibration();

// Thead in charge of reading data from the IMU.
void imuThreadJetson();

// Convert rotation vector to quaternion.
glm::quat convertOpencvRotVectToQuat(cv::Vec3d rotVect);

// Convert quaternion to rotation vector.
cv::Vec3d convertQuatToOpencvRotVect(glm::quat quaternion);

// Create spline points.
std::vector<glm::vec3> createSplinePoint(std::vector<ImuInputJetson> imuReadVector);

// Create slerp points for quaternions (tests at home).
std::vector<glm::quat> createSlerpPoint(std::vector<ImuInputJetson> imuReadVector);

// Tests Slerp and Spline methods.
void testSlerpAndSpline(std::vector<ImuInputJetson> imuReadVector, std::vector<CameraInput> cameraReadVector);

// Get rotation and translation from frame.
FrameMarkersData getRotationTraslationFromFrame(CameraInput frame);

// Get rotation and translation from all frames.
std::vector<FrameMarkersData> getRotationTraslationFromAllFrames(std::vector<CameraInput> cameraReadVector);

// Interpolate camera rotation to fit IMU data.
std::vector<CameraInterpolatedData> interpolateCameraRotation(const std::vector<ImuInputJetson> imuReadVectorCopy,
                                                              const std::vector<CameraInput> cameraReadVectorCopy,
                                                              std::vector<FrameMarkersData> frameMarkersDataVector);

// Test interpolate camera rotation to fit IMU data.
void testInterpolateCamera(std::vector<CameraInterpolatedData> interpolatedPoints);

// Create a hard copy of camera vector.
std::vector<CameraInput> hardCopyCameraVector(std::vector<CameraInput> cameraReadVector);

// Print data from the IMU to the console for testing.
void printIMUData();

#endif // JETSON_H
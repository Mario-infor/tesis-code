#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <BNO055-BBB_driver.h>
#include <chrono>
#include <curses.h>
#include <vector>
#include <thread>
#include <mutex>
#include <fstream>
#include "RingBuffer.h"

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

#define CAMERA_MATRIX = (cv::Mat_<double>(3, 3) << 493.02975478, 0, 310.67004724,
                        0, 495.25862058, 166.53292108,
                        0, 0, 1);

#define CAMERA_DIST_COEFF = (cv::Mat_<double>(1, 5) << 0.12390713, 0.17792574, -0.00934536, -0.01052198, -1.13104202);

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

// Buffer to store camera structs.
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(RING_BUFFER_LENGTH_CAMERA);

// Buffer to store IMU structs.
RingBuffer<ImuInputJetson> imuDataJetsonBuffer = RingBuffer<ImuInputJetson>(RING_BUFFER_LENGTH_IMU);

// Global variables that need to be accessed from different threads or methods.
std::mutex myMutex;
std::chrono::time_point<std::chrono::steady_clock> timeCameraStart;
std::chrono::time_point<std::chrono::steady_clock> timeIMUStart;
std::string dirCameraFolder = "./data/camera/";
std::string dirIMUFolder = "./data/imu/";
std::string dirRotationsFolder = "./data/rotations/";
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

bool stopProgram = false;
bool doneCalibrating = false;
bool generateNewData = true;
bool preccessData = false;


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

// Write IMU data to files.
void IMUDataJetsonWrite();

// Write IMU data to files for testing the axis distribution.
void IMUDataWriteTestAxis();

// Read IMU data from files.
std::vector<ImuInputJetson> readDataIMUJetson();

// Write camera time data to file and store all frams as .png files.
void cameraDataWrite();

// Read camera data and frames from files.
std::vector<CameraInput> readDataCamera();

// Write camera rotations after slerp and store on .csv file.
void cameraRotationSlerpDataWrite(std::vector<CameraInterpolatedData> cameraSlerpRotationsVector);
// Write camera rotations without slerp and store on .csv file.
void cameraRotationSlerpDataWrite(std::vector<FrameMarkersData> cameraRotationsVector);

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

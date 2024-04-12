#ifndef JETSON_H
#define JETSON_H

#include <Eigen/Dense>

// Amount of IMU data and frames to read from devices.
//#define RING_BUFFER_LENGTH_CAMERA 1875
//#define RING_BUFFER_LENGTH_CAMERA 3750

#define RING_BUFFER_LENGTH_CAMERA 375
#define RING_BUFFER_LENGTH_IMU 1500

// Global variables that need to be accessed from different threads or methods.
std::chrono::time_point<std::chrono::steady_clock> timeCameraStart;
std::chrono::time_point<std::chrono::steady_clock> timeIMUStart;

bool doneCalibrating = false;

// Thread in charge of readng data from camera and store it on camera buffer.
void cameraCaptureThread();

// Method to calibrate de IMU sensors.
void imuCalibration();

// Thead in charge of reading data from the IMU.
void imuThreadJetson();

// Initialisation of the Kalman Filter state and parameters.
void initKalmanFilter(cv::KalmanFilter &KF, int stateSize);

void predict(cv::KalmanFilter &KF);

void doMeasurement(cv::Mat_<float> &measurement, cv::Mat_<float> measurementOld,
                    FrameMarkersData frameMarkersData, float deltaT);

void correct(cv::KalmanFilter &KF, cv::Mat_<float> measurement);

void correctIMU(cv::KalmanFilter &KF, Eigen::Matrix<double, 19, 1> measurement);

// Update the transition matrix (A) with new deltaT value.
void updateTransitionMatrix(cv::KalmanFilter &KF, float deltaT);

// Update the transition matrix (A) for IMU KF with new deltaT and gyro values.
void updateTransitionMatrixIMU(cv::KalmanFilter &KF, Eigen::Matrix<double, 19, 1> measurenment, float deltaT);

// Initialisation of statePost the first time when no prediction have been made.
void initStatePostFirstTime(cv::KalmanFilter &KF, cv::Mat_<float> measurement);

// Method to predict the next state of the imu data.
void imuPreintegration(const float deltaT, const Eigen::Vector3d acc,
 const Eigen::Vector3d gyro, Eigen::Vector3d &deltaPos, Eigen::Vector3d &deltaVel,
 Eigen::Matrix3d imuRot);

// Method to predict and correct the state of the camera data.
void runKalmanFilterCamera();

// Method to predict the state of the IMU data.
void runIMUPrediction();

#endif // JETSON_H
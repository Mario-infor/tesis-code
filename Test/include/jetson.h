#ifndef JETSON_H
#define JETSON_H

#include <Eigen/Dense>

// Amount of IMU data and frames to read from devices.
//#define RING_BUFFER_LENGTH_CAMERA 1875
//#define RING_BUFFER_LENGTH_CAMERA 3750

#define RING_BUFFER_LENGTH_CAMERA 900
#define RING_BUFFER_LENGTH_IMU 3500

#define THRESHOLD_IMU_ACC 0.001
#define THRESHOLD_IMU_GYRO 0.01

#define BASE_MARKER_ID 30

//#define THRESHOLD_IMU_ACC_MAX 0.001
//#define THRESHOLD_IMU_ACC_MIN -0.0001
//#define THRESHOLD_IMU_GYRO_MAX 0.07
//#define THRESHOLD_IMU_GYRO_MIN -0.05

bool doneCalibrating = false;

// Global variables that need to be accessed from different threads or methods.
std::chrono::time_point<std::chrono::steady_clock> timeCameraStart;
std::chrono::time_point<std::chrono::steady_clock> timeIMUStart;

// Thread in charge of readng data from camera and store it on camera buffer.
void cameraCaptureThread();

// Thead in charge of reading data from the IMU.
void imuThreadJetson();

// Method to calibrate de IMU sensors.
void imuCalibration();

// Initialisation of the Kalman Filter state and parameters.
void initKalmanFilter(cv::KalmanFilter &KF, int stateSize);

void predict(cv::KalmanFilter &KF);

void correct(cv::KalmanFilter &KF, cv::Mat_<float> measurement, cv::Mat measurementNoiseCov);

void correctIMU_EKF(
    cv::KalmanFilter &KF,
    cv::Mat measurementNoiseCov,
    Eigen::MatrixXd measurement,
    Eigen::MatrixXd h,
    Eigen::MatrixXd H);

void updateTransitionMatrixFusion(cv::KalmanFilter &KF, float deltaT, int stateSize, Eigen::Vector3d w);

// Method to predict the next state of the imu data.
void imuPreintegration(
    const float deltaT,
    const Eigen::Vector3d acc,
    Eigen::Vector3d &deltaPos,
    Eigen::Vector3d &deltaVel);

// Method to predict and correct the state of the IMU and Camera together.
void runCameraAndIMUKalmanFilter();

#endif // JETSON_H
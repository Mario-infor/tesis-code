#ifndef JETSON_H
#define JETSON_H

#include <Eigen/Dense>

// Amount of IMU data and frames to read from devices.
//#define RING_BUFFER_LENGTH_CAMERA 1875
//#define RING_BUFFER_LENGTH_CAMERA 3750

#define RING_BUFFER_LENGTH_CAMERA 375
#define RING_BUFFER_LENGTH_IMU 1500

#define THRESHOLD_IMU_ACC 0.001
#define THRESHOLD_IMU_GYRO 0.01

//#define THRESHOLD_IMU_ACC_MAX 0.001
//#define THRESHOLD_IMU_ACC_MIN -0.0001
//#define THRESHOLD_IMU_GYRO_MAX 0.07
//#define THRESHOLD_IMU_GYRO_MIN -0.05

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

void correctIMU(cv::KalmanFilter &KF, Eigen::Matrix<double, 23, 1> measurement);

void correctIMU_EKF(cv::KalmanFilter &KF, Eigen::Matrix<double, 13, 1> measurement, Eigen::Matrix<double, 13, 1> h);

// Update the transition matrix (A) with new deltaT value.
void updateTransitionMatrix(cv::KalmanFilter &KF, float deltaT);

// Update the transition matrix (A) for IMU KF with new deltaT and gyro values.
void updateTransitionMatrixIMU(cv::KalmanFilter &KF, Eigen::Matrix<double, 23, 1> measurenment, float deltaT);

void updateTransitionMatrixFusionIMU(cv::KalmanFilter &KF, float deltaT);

void updateTransitionMatrixFusionCamera (cv::KalmanFilter &KF, float deltaT);

// Method to predict the next state of the imu data.
void imuPreintegration(const float deltaT, const Eigen::Vector3d acc,
 const Eigen::Vector3d gyro, Eigen::Vector3d &deltaPos, Eigen::Vector3d &deltaVel,
 Eigen::Matrix3d imuRot);

// Method to predict and correct the state of the camera data.
void runKalmanFilterCamera();

// Method to predict the state of the IMU data.
void runIMUPrediction();

// Method to predict and correct the state of the IMU and Camera together.
void runCameraAndIMUKalmanFilter();

Eigen::Matrix<double, 12, 1> getMeasurenmentEstimateFromState(
    cv::KalmanFilter &KF,
    Eigen::Matrix<double, 4, 4> Gti,
    Eigen::Matrix<double, 4, 4> Gci);

#endif // JETSON_H
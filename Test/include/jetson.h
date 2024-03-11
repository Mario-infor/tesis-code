#ifndef JETSON_H
#define JETSON_H

// Amount of IMU data and frames to read from devices.
//#define RING_BUFFER_LENGTH_CAMERA 1875
//#define RING_BUFFER_LENGTH_CAMERA 3750

#define RING_BUFFER_LENGTH_CAMERA 375
#define RING_BUFFER_LENGTH_IMU 750

// Global variables that need to be accessed from different threads or methods.
std::mutex myMutex;
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
void initKalmanFilter(cv::KalmanFilter &KF);

void predict(cv::KalmanFilter &KF);

void doMeasurement(cv::Mat_<float> &measurement, cv::Mat_<float> measurementOld,
                    FrameMarkersData frameMarkersData, float deltaT);

void correct(cv::KalmanFilter &KF, cv::Mat_<float> measurement);

// Update the transition matrix (A) with new deltaT value.
void updateTransitionMatrix(cv::KalmanFilter &KF, float deltaT);

// Initialisation of statePost the first time when no prediction have been made.
void initStatePostFirstTime(cv::KalmanFilter &KF, cv::Mat_<float> measurement);

// Method to predict the next state of the imu data.
void imuPreintegration();

#endif // JETSON_H
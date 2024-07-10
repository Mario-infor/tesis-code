#ifndef JETSON_H
#define JETSON_H

#include <Eigen/Dense>

// Amount of IMU data and frames to read from devices.
//#define RING_BUFFER_LENGTH_CAMERA 1875
//#define RING_BUFFER_LENGTH_CAMERA 3750

#define RING_BUFFER_LENGTH_CAMERA 900
#define RING_BUFFER_LENGTH_IMU 3500

//#define RING_BUFFER_LENGTH_CAMERA 100
//#define RING_BUFFER_LENGTH_IMU 500

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


/**
 * @brief This method is in charge of reading data from the camera and storing it in the camera buffer. It also calculates 
 * the time it takes to read the camera data from the start of the algorithm to the current loop iteration. It waits for 
 * the IMU calibration to be done and then it starts taking measurements.
 * @param None
 * @return It updates the camera buffer with the new frame and time data. 
*/
void cameraCaptureThread();

/**
 * @brief This method is in charge of reading data from the IMU and storing it in the IMU buffer. It also calculates
 * the time it takes to read the IMU data from the start of the algorithm to the current loop iteration. It waits for
 * the calibration of its sensors to be done and then it starts taking measurements.
 * @param None
 * @return It updates the IMU buffer with the new IMU data and time data.
*/
void imuThreadJetson();

/**
 * @brief This method is in charge of comunicating with the IMU sensor and activate its routing to calibrate the sensors.
 * Aswell as showing the calibration status to the user via console.
 * @param None
 * @return None
*/
void imuCalibration();

/**
 * @brief This method is in charge of giving initial values to some of the Kalman Filter matrices such as the state(X'),
 * the processNoiseCov (Q), the errorCovPost (R), the transitionMatrix (A) and the measurementMatrix (H).
 * @param KF Object from OpenCV that contains the Kalman Filter params.
 * @param stateSize Size of the state vector.
 * @return It updates the KF object.
*/
void initKalmanFilter(cv::KalmanFilter &KF, const int stateSize);

/**
 * @brief This method corresponds to the prediction step of the Kalman Filter. It uses the Discrte Kalman Filter equations
 * to predict the next state of the system.
 * @param KF Object from OpenCV that contains the Kalman Filter params.
 * @return It updates the KF object with the new predicted state and predicted covariance matrix.
*/
void predict(cv::KalmanFilter &KF);


/**
 * @brief This method corresponds to the correction step of the Kalman Filter. It is used when the measurement of the current
 * iteration comes from the camera. It uses the Discrete Kalman Filter equations to update the state and the covariance matrix.
 * @param KF Object from OpenCV that contains the Kalman Filter params.
 * @param measurement Measurement vector for the camera.
 * @param measurementNoiseCov Measurement noise covariance matrix for the camera.
 * @return It updates the KF object with the new Kalman Gain, the corrected state and corrected covariance matrix.
*/
void correct(cv::KalmanFilter &KF, const cv::Mat_<float> measurement, const cv::Mat measurementNoiseCov);

/**
 * @brief This method corresponds to the correction step of the Kalman Filter. It is used when the measurement of the current
 * iteration comes from the IMU. It uses the Extended Kalman Filter equations to update the state and the covariance matrix.
 * @param KF Object from OpenCV that contains the Kalman Filter params.
 * @param measurementNoiseCov Measurement noise covariance matrix for the IMU.
 * @param measurement Measurement vector for the IMU.
 * @param h Measurement function for the IMU. Function that maps the state space to the IMU measurement space.
 * @param H Jacobian of the measurement function h.
 * @return It updates the KF object with the new Kalman Gain, the corrected state and corrected covariance matrix.
*/
void correctIMU_EKF(
    cv::KalmanFilter &KF,
    const cv::Mat measurementNoiseCov,
    const Eigen::MatrixXd measurement,
    const Eigen::MatrixXd h,
    const Eigen::MatrixXd H);

/**
 * @brief This method is in charge of updating the transition matrix of the Kalman Filter. It uses delta time and the angular
 * velocity of the camera measurenment or its estimate calculated from the IMU measurenment to update the transition matrix. It
 * uses the camera data or the IMU data depending on where the mearurements comes from at the current iteration.
 * @param KF Object from OpenCV that contains the Kalman Filter params.
 * @param deltaT Time difference between the current and the last iteration. The last iteration could be from the camera or the IMU.
 * @param stateSize Size of the state vector.
 * @param w Angular velocity of the camera measurenment or the camera's angular velocity calculated from the IMU measurement.
 * @return It updates the transition matrix inside the KF object.
*/
void updateTransitionMatrixFusion(
    cv::KalmanFilter &KF,
    const float deltaT,
    const int stateSize,
    const Eigen::Vector3d w);

/**
 * @brief This method is in charge of integrating the acceleration of the IMU with respecto to its origin to get the position and
 * velocity of the IMU. It uses the acceleration rotated with respect to the origin reference frame and the time difference between
 * the current and the last IMU measurement to calculate the position and velocity of the IMU with respecto to its origin.
 * @param deltaT Time difference between the current and the last IMU measurement.
 * @param acc Acceleration of the IMU rotated with respect to the origin reference frame.
 * @param deltaPos Position of the IMU with respect to its origin.
 * @param deltaVel Velocity of the IMU with respect to its origin.
 * @return It updates the deltaPos and deltaVel vectors with the new position and velocity of the IMU.
*/
void imuPreintegration(
    const float deltaT,
    const Eigen::Vector3d acc,
    Eigen::Vector3d &deltaPos,
    Eigen::Vector3d &deltaVel);

/**
 * @brief This method is the hole Kalman Filter algorithm. It is in charge of reading data from the camera and the IMU,
 * predicting the next state of the system, correcting the state with the camera and the IMU data and updating the transition
 * matrix of the Kalman Filter. It also integrates the acceleration of the IMU with respect to its origin to get the position
 * and velocity of the IMU.
 * @param None
 * @return It writes all measurement of the camera and the state to a csv file.
*/
void runCameraAndIMUKalmanFilter();

#endif // JETSON_H
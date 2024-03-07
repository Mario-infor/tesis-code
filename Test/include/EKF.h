#ifndef EKF_H
#define EKF_H

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <structsFile.h>

// Initialisation of the Kalman Filter state and parameters.
void initKalmanFilter(class KalmanFilter &KF);

void predict(class KalmanFilter &KF);

void doMeasurement(
    cv::Mat_<float> &measurement,
    CameraInput frame,
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs);

void correct();

// Update the transition matrix (A) with new deltaT value.
void updateTransitionMatrix(class KalmanFilter &KF, float deltaT);

// Initialisation of statePost the first time when no prediction have been made.
void initStatePostFirstTime(cv::KalmanFilter &KF, cv::Mat_<float> measurement);

#endif // EKF_H
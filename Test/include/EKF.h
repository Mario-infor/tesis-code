#ifndef EKF_H
#define EKF_H

#include <structsFile.h>

// Initialisation of the Kalman Filter state and parameters.
void initKalmanFilter(cv::KalmanFilter &KF);

void predict(cv::KalmanFilter &KF);

void doMeasurement(cv::Mat_<float> &measurement, FrameMarkersData frameMarkersData);

void correct(cv::KalmanFilter &KF, cv::Mat_<float> measurement);

// Update the transition matrix (A) with new deltaT value.
void updateTransitionMatrix(cv::KalmanFilter &KF, float deltaT);

// Initialisation of statePost the first time when no prediction have been made.
void initStatePostFirstTime(cv::KalmanFilter &KF, cv::Mat_<float> measurement);

#endif // EKF_H

#ifndef EKF_H
#define EKF_H

// Initialisation of the Kalman Filter state and parameters.
void initKalmanFilter(class KalmanFilter &KF);

void predict(class KalmanFilter &KF);

void doMeasurement(class Mat_ &measurement);

void correct();

// Update the transition matrix (A) with new deltaT value.
void updateTransitionMatrix(class KalmanFilter &KF, float deltaT);

// Initialisation of statePost the first time when no prediction have been made.
void initStatePostFirstTime(cv::KalmanFilter &KF, cv::Mat_<float> measurement);

#endif // EKF_H
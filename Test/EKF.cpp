#include <opencv2/opencv.hpp>
#include <EKF.h>
#include <utils.h>


void initKalmanFilter(cv::KalmanFilter &KF)
{
    KF.statePre.at<float>(0) = 0; // x traslation.
    KF.statePre.at<float>(1) = 0; // y traslation.
    KF.statePre.at<float>(2) = 0; // z traslation.
    KF.statePre.at<float>(3) = 0; // w quat rotation.
    KF.statePre.at<float>(4) = 0; // x quat rotation.
    KF.statePre.at<float>(5) = 0; // y quat rotation.
    KF.statePre.at<float>(6) = 0; // z quat rotation.
    KF.statePre.at<float>(7) = 0; // x traslation velocity.
    KF.statePre.at<float>(8) = 0; // y traslation velocity.
    KF.statePre.at<float>(9) = 0; // z traslation velocity.
    KF.statePre.at<float>(10) = 0; // w rotation velocity.
    KF.statePre.at<float>(11) = 0; // x rotation velocity.
    KF.statePre.at<float>(12) = 0; // y rotation velocity.
    KF.statePre.at<float>(13) = 0; // z rotation velocity.

    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4)); // Q.
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-2)); // R.
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1)); // P'.
    cv::setIdentity(KF.transitionMatrix, cv::Scalar::all(1)); // A.
}

void predict(cv::KalmanFilter &KF)
{
    KF.statePre = KF.transitionMatrix * KF.statePost;
    KF.errorCovPre = KF.transitionMatrix * KF.errorCovPost * KF.transitionMatrix.t() + KF.processNoiseCov;
}

void doMeasurement(cv::Mat_<float> &measurement)
{
    //TODO: Read measurement from data files.
    measurement(0) = 1;
    measurement(1) = 1;
    measurement(2) = 1;
    measurement(3) = 1;
    measurement(4) = 1;
    measurement(5) = 1;    
}

void correct(cv::KalmanFilter &KF, cv::Mat_<float> &measurement)
{
    cv::Mat_<float> y = measurement - KF.statePre;
    cv::Mat_<float> S = KF.measurementMatrix * KF.errorCovPre * KF.measurementMatrix.t() + KF.measurementNoiseCov;
    KF.gain = KF.errorCovPre * KF.measurementMatrix.t() * S.inv();
    
    int stateSize = KF.statePre.rows;

    KF.statePost = KF.statePre + KF.gain * y;
    KF.errorCovPost = (cv::Mat::eye(stateSize, stateSize, KF.statePost.type()) - KF.gain * KF.measurementMatrix) * KF.errorCovPre;
}

void updateTransitionMatrix(cv::KalmanFilter &KF, float deltaT)
{
    KF.transitionMatrix =
    (cv::Mat_<float>(14, 14) <<
    1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0, \
    0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, \
    0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, \
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, \
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, \
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, \
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT, \
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    );
}

void updateMeasurementMatrix(cv::KalmanFilter &KF)
{
    KF.measurementMatrix =
    (cv::Mat_<float>(6, 14) <<
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
    );
}

void initStatePostFirstTime(cv::KalmanFilter &KF, cv::Mat_<float> measurement)
{
    KF.statePost = KF.statePre;

    cv::Vec3d rotVect = cv::Vec3d(measurement.at<float>(3), measurement.at<float>(4), measurement.at<float>(5));
    glm::quat quaternion = convertOpencvRotVectToQuat(rotVect);

    KF.statePost.at<float>(0) = measurement.at<float>(0);
    KF.statePost.at<float>(1) = measurement.at<float>(1);
    KF.statePost.at<float>(2) = measurement.at<float>(2);
    KF.statePost.at<float>(3) = quaternion.w;
    KF.statePost.at<float>(4) = quaternion.x;
    KF.statePost.at<float>(5) = quaternion.y;
    KF.statePost.at<float>(6) = quaternion.z;
}

int main()
{
    bool firstRun = true;
    float deltaT = 10;
    cv::KalmanFilter KF(14, 6, 0);

    // Create measurement vector (rvec, tvec).
    cv::Mat_ <float> measurement(6, 1);
    measurement.setTo(cv::Scalar(0));

    // Initialize Kalman Filter.
    initKalmanFilter(KF);

    if (!firstRun)
    {
        predict(KF);

        updateTransitionMatrix(KF, deltaT);
        updateMeasurementMatrix(KF);

        doMeasurement(measurement);

        correct(KF, measurement);
    }
    else
    {
        doMeasurement(measurement);
        initStatePostFirstTime(KF, measurement);
        firstRun = false;
    }
}
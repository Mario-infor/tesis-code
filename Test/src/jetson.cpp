#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <curses.h>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>
#include <fstream>
#include <BNO055-BBB_driver.h>
#include <readWriteData.h>
#include <jetson.h>
#include <utils.h>
#include <interpolationUtils.h>
#include <cameraInfo.h>
#include <Eigen/Dense>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/spline.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

using namespace Eigen;

// Buffer to store camera structs.
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(RING_BUFFER_LENGTH_CAMERA);

// Buffer to store IMU structs.
RingBuffer<ImuInputJetson> imuDataJetsonBuffer = RingBuffer<ImuInputJetson>(RING_BUFFER_LENGTH_IMU);

void cameraCaptureThread()
{
    std::string pipeline = gstreamerPipeline(FRAME_WIDTH,
            FRAME_HEIGHT,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            FRAME_RATE,
            FLIP_METHOD);
    
    cv::VideoCapture cap;
    cap.open(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened())
        std::cerr << "Error openning the camera." << std::endl;
    else
    {
        int index = 0;

        while (!doneCalibrating)
        {
            std::cout << "Camera: " << index << std::endl;
        }

        while (index < RING_BUFFER_LENGTH_CAMERA)
        {
            std::cout << "Camera: " << index << std::endl;

            cv::Mat frame, grayscale;
            cap.read(frame);

            if (frame.empty())
            {
                std::cerr << "Could not capture frame." << std::endl;
                break;
            }
            else
            {
                cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
                CameraInput capture;
                capture.index = index;
                capture.frame = grayscale.clone();
                capture.time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeCameraStart).count();

                cameraFramesBuffer.Queue(capture);
                index++;
            }
        }
    }

    std::cout << "Camera Thread Finished." << std::endl;
}

void imuCalibration()
{
    char filename[] = IMU_ADDRESS;
    BNO055 sensors;
    sensors.openDevice(filename);

    // Wait for calibration to finish.
    do
    {
        sensors.readCalibVals();
        printf("Sys: %d, Mag: %d, Gyro: %d, Acc: %d\n", sensors.calSys, sensors.calMag, sensors.calGyro, sensors.calAcc);
        doneCalibrating = sensors.calSys == 3 && sensors.calMag == 3 && sensors.calGyro == 3 && sensors.calAcc == 3;
        doneCalibrating = false;
    } while (!doneCalibrating);
}

// Thead in charge of reading data from the IMU.
void imuThreadJetson()
{
    int cont = 0;
    char filename[] = IMU_ADDRESS;
    BNO055 sensors;
    sensors.openDevice(filename);

    // Wait for calibration to finish.
    do
    {
        sensors.readCalibVals();
        doneCalibrating = sensors.calSys == 3 && sensors.calMag == 3 && sensors.calGyro == 3 && sensors.calAcc == 3;
    } while (cont++ < 2000 && !doneCalibrating);

    doneCalibrating = true;
    int index = 0;

    while (index < RING_BUFFER_LENGTH_IMU)
    {
        sensors.readAll();

        ImuInputJetson imuInputJetson;
        imuInputJetson.index = index;
        imuInputJetson.time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeIMUStart).count();

        imuInputJetson.gyroVect = glm::vec3(sensors.gyroVect.vi[0] * 0.01, sensors.gyroVect.vi[1] * 0.01, sensors.gyroVect.vi[2] * 0.01);
        imuInputJetson.eulerVect = glm::vec3(sensors.eOrientation.vi[0] * sensors.Scale, sensors.eOrientation.vi[1] * sensors.Scale, sensors.eOrientation.vi[2] * sensors.Scale);
        imuInputJetson.rotQuat = glm::quat(sensors.qOrientation.vi[3] * sensors.Scale, sensors.qOrientation.vi[0] * sensors.Scale,
                                           sensors.qOrientation.vi[1] * sensors.Scale, sensors.qOrientation.vi[2] * sensors.Scale);

        imuInputJetson.accVect = glm::vec3(sensors.accelVect.vi[0] * sensors.Scale, sensors.accelVect.vi[1] * sensors.Scale, sensors.accelVect.vi[2] * sensors.Scale);
        imuInputJetson.gravVect = glm::vec3(sensors.gravVect.vi[0] * 0.01, sensors.gravVect.vi[1] * 0.01, sensors.gravVect.vi[2] * 0.01);

        imuDataJetsonBuffer.Queue(imuInputJetson);
        index++;
    }

    std::cout << "IMU Thread Finished." << std::endl;
}

void initKalmanFilter(cv::KalmanFilter &KF, int stateSize)
{
    KF.statePre = cv::Mat::zeros(stateSize, 1, CV_32F);
    
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));     // Q.
    //cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-2)); // R.
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));           // P'.
    cv::setIdentity(KF.transitionMatrix, cv::Scalar::all(1));       // A.
    cv::setIdentity(KF.measurementMatrix, cv::Scalar::all(1));      // H.
}

void predict(cv::KalmanFilter &KF)
{
    KF.statePre = KF.transitionMatrix * KF.statePost;
    KF.errorCovPre = KF.transitionMatrix * KF.errorCovPost * KF.transitionMatrix.t() + KF.processNoiseCov;
}

void doMeasurement(cv::Mat_<float> &measurement, cv::Mat_<float> measurementOld,
FrameMarkersData frameMarkersData, float deltaT)
{
    measurement(0) = frameMarkersData.tvecs[0].val[0]; // traslation (x)
    measurement(1) = frameMarkersData.tvecs[0].val[1]; // traslation (y)
    measurement(2) = frameMarkersData.tvecs[0].val[2]; // traslation (z)
    measurement(3) = frameMarkersData.rvecs[0].val[0]; // rotation (x)
    measurement(4) = frameMarkersData.rvecs[0].val[1]; // rotation (y)
    measurement(5) = frameMarkersData.rvecs[0].val[2]; // rotation (z)

    measurement(6) = (measurement(0) - measurementOld(0)) / deltaT; // traslation speed (x)
    measurement(7) = (measurement(1) - measurementOld(1)) / deltaT; // traslation speed (y)
    measurement(8) = (measurement(2) - measurementOld(2)) / deltaT; // traslation speed (z)
    measurement(9) = (measurement(3) - measurementOld(3)) / deltaT; // rotation speed (x)
    measurement(10) = (measurement(4) - measurementOld(4)) / deltaT; // rotation speed (y)
    measurement(11) = (measurement(5) - measurementOld(5)) / deltaT; // rotation speed (z)    
}

void correct(cv::KalmanFilter &KF, cv::Mat_<float> measurement, cv::Mat measurementNoiseCov)
{
    cv::Mat_<float> S = KF.measurementMatrix * KF.errorCovPre * KF.measurementMatrix.t() + measurementNoiseCov;
    KF.gain = KF.errorCovPre * KF.measurementMatrix.t() * S.inv();
    
    int stateSize = KF.statePre.rows;

    KF.statePost = KF.statePre + KF.gain * (measurement - KF.statePre);
    KF.errorCovPost = (cv::Mat::eye(stateSize, stateSize, KF.statePost.type()) - KF.gain * KF.measurementMatrix) * KF.errorCovPre;
}

void correctIMU(cv::KalmanFilter &KF, Eigen::Matrix<double, 23, 1> measurement)
{
    cv::Mat_<float> S = KF.measurementMatrix * KF.errorCovPre * KF.measurementMatrix.t() + KF.measurementNoiseCov;
    KF.gain = KF.errorCovPre * KF.measurementMatrix.t() * S.inv();

    int stateSize = KF.statePre.rows;

    cv::Mat tempMeasurement = convertEigenMatToOpencvMat(measurement);

    KF.statePost = KF.statePre + KF.gain * (tempMeasurement - KF.statePre);
    KF.errorCovPost = (cv::Mat::eye(stateSize, stateSize, KF.statePost.type()) - KF.gain * KF.measurementMatrix) * KF.errorCovPre;
}

void correctIMU_EKF(
    cv::KalmanFilter &KF,
    cv::Mat measurementNoiseCov,
    Eigen::MatrixXd measurement,
    Eigen::MatrixXd h,
    Eigen::MatrixXd H)
{
    cv::Mat cv_H = convertEigenMatToOpencvMat(H);

    cv::Mat S = cv_H * KF.errorCovPre * cv_H.t() + measurementNoiseCov;
    cv::Mat SInvert = S.inv();

    cv::Mat cv_H_traspose = cv_H.t();

    KF.gain = KF.errorCovPre * cv_H_traspose * SInvert;
    int stateSize = KF.statePre.rows;

    cv::Mat error = convertEigenMatToOpencvMat(measurement - h);

    KF.statePost = KF.statePre + KF.gain * error;
    KF.errorCovPost = (cv::Mat::eye(stateSize, stateSize, KF.statePost.type()) - KF.gain * cv_H) * KF.errorCovPre;
}

void updateTransitionMatrix(cv::KalmanFilter &KF, float deltaT)
{
    KF.transitionMatrix =
        (cv::Mat_<float>(13, 13) << 
        1, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, deltaT, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, deltaT, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, deltaT, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, deltaT,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
        );
}

void updateTransitionMatrixFusion(cv::KalmanFilter &KF, float deltaT, int stateSize, Eigen::Vector3d w)
{
    float dT2 = deltaT / 2;

    //float w1 = KF.statePre.at<float>(10);
    //float w2 = KF.statePre.at<float>(11);
    //float w3 = KF.statePre.at<float>(12);

    float w1 = w(0);
    float w2 = w(1);
    float w3 = w(2);
    
    KF.transitionMatrix =
        (cv::Mat_<float>(stateSize, stateSize) << 
        1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0,
        0, 0, 0, 1, -dT2*w1, -dT2*w2, -dT2*w3, 0, 0, 0, 0, 0, 0,
        0, 0, 0, dT2*w1, 1, dT2*w3, -dT2*w2, 0, 0, 0, 0, 0, 0,
        0, 0, 0, dT2*w2, -dT2*w3, 1, dT2*w1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, dT2*w3, dT2*w2, -dT2*w1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
        );
}

/*
Eigen::Matrix<double, 12, 1> getMeasurenmentEstimateFromState(
    cv::KalmanFilter &KF,
    Eigen::Matrix<double, 4, 4> Gti,
    Eigen::Matrix<double, 4, 4> Gci)
{
    Eigen::Matrix<double, 12, 1> camMeasurementFromIMU;

    Eigen::Quaterniond camQuat(KF.statePost.at<float>(3), KF.statePost.at<float>(4),
     KF.statePost.at<float>(5), KF.statePost.at<float>(6));

    Eigen::Matrix<double, 3, 3> camRotMat = camQuat.toRotationMatrix();
    Eigen::Matrix<double, 3, 1> camT(KF.statePost.at<float>(0), KF.statePost.at<float>(1), KF.statePost.at<float>(2));

    Eigen::Matrix<double, 4, 4> Gmc;
    Gmc.setIdentity();
    Gmc.block<3, 3>(0, 0) = camRotMat;
    Gmc.block<3, 1>(0, 3) = camT;
    
    Eigen::Matrix<double, 4, 4> Gmi = Gci * Gmc;

    // Rotation and position of the IMU in the IMU frame.
    Eigen::Matrix<double, 4, 4> Gi = invertG(Gti) * Gmi; 

    Eigen::Vector3d stateVel = Eigen::Vector3d(KF.statePost.at<float>(7), KF.statePost.at<float>(8), KF.statePost.at<float>(9));
    Eigen::Vector3d stateAngularVel = Eigen::Vector3d(KF.statePost.at<float>(9), KF.statePost.at<float>(10), KF.statePost.at<float>(11));
    Eigen::Matrix3d wHat = getWHat(stateAngularVel);

    Eigen::Matrix<double, 4 ,4> camGhi;
    camGhi.setZero();
    camGhi.block<3, 3>(0, 0) = wHat;
    camGhi.block<3, 1>(0, 3) = stateVel;

    // Angular and linear velocity of the IMU in the IMU frame.
    Eigen::Matrix<double, 4 ,4> imuGhi = Gci * camGhi * invertG(Gci); 

    Eigen::Vector3d estimateMeasurenmentVel =  imuGhi.block<3, 1>(0, 3);
    Eigen::Matrix3d estimateMeasurenmentAngularVel = imuGhi.block<3, 3>(0, 0);
    Eigen::Vector3d estimateMeasurenmentPos = Gi.block<3, 1>(0, 3);
    Eigen::Matrix<double, 3, 3> estimateMeasurenmentRot = Gi.block<3, 3>(0, 0);
    Eigen::Quaterniond estimateMeasurenmentQuat(estimateMeasurenmentRot);
    estimateMeasurenmentQuat.normalize();
    
    camMeasurementFromIMU(0) = estimateMeasurenmentAngularVel(2,1);
    camMeasurementFromIMU(1) = estimateMeasurenmentAngularVel(0,2);
    camMeasurementFromIMU(2) = estimateMeasurenmentAngularVel(1,0);
    camMeasurementFromIMU(3) = estimateMeasurenmentQuat.w();
    camMeasurementFromIMU(4) = estimateMeasurenmentQuat.x();
    camMeasurementFromIMU(5) = estimateMeasurenmentQuat.y();
    camMeasurementFromIMU(6) = estimateMeasurenmentQuat.z();
    camMeasurementFromIMU.block<3, 1>(7, 0) = estimateMeasurenmentVel;
    camMeasurementFromIMU.block<3, 1>(10, 0) = estimateMeasurenmentPos;

    return camMeasurementFromIMU;
}
*/

void updateTransitionMatrixIMU(cv::KalmanFilter &KF, Eigen::Matrix<double, 23, 1> measurenment, float deltaT)
{
    /*float w1 = KF.statePost.at<float>(0);
    float w2 = KF.statePost.at<float>(1);
    float w3 = KF.statePost.at<float>(2);

    KF.transitionMatrix =
        (cv::Mat_<float>(23, 23) << 
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);*/


    /*KF.transitionMatrix =
        (cv::Mat_<float>(23, 23) << 
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, -dT2*w1, -dT2*w2, -dT2*w3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, dT2*w1, 1, dT2*w3, -dT2*w2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, dT2*w2, -dT2*w3, 1, dT2*w1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, dT2*w3, dT2*w2, -dT2*w1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);*/
}

void imuPreintegration(
    const float deltaT,
    const Eigen::Vector3d acc,
    const Eigen::Vector3d gyro,
    Eigen::Vector3d &deltaPos,
    Eigen::Vector3d &deltaVel,
    Eigen::Matrix3d imuRot)
{
    deltaPos += deltaVel * deltaT + 0.5 * imuRot * acc * deltaT * deltaT;
    deltaVel += imuRot * acc * deltaT;
}

void runCameraAndIMUKalmanFilter()
{
    std::vector<Eigen::Vector3d> vectorOfPointsOne;
    std::vector<Eigen::Vector3d> vectorOfPointsTwo;
    std::vector<Eigen::Vector3d> vectorOfMarkers;
    std::vector<TransformBetweenMarkers> vectorOfTransforms;
    std::vector<Eigen::Vector3d> vectorErrorPoints;
    std::vector<Eigen::VectorXd> vectorCamMeasurenments;
    std::vector<Eigen::VectorXd> vectorStates;

    std::vector<Eigen::VectorXd> vectorImuAcc;
    std::vector<Eigen::VectorXd> vectorIIRAcc;

    std::vector<Eigen::VectorXd> vectorImuGyro;
    std::vector<Eigen::VectorXd> vectorIIRGyro;

    std::vector<float> timeStamps;

    FILE *output;
    output = popen("gnuplot", "w");

    std::vector<CameraInput> cameraData = readDataCamera();
    std::vector<ImuInputJetson> imuReadVector = readDataIMUJetson();

    CameraInput firstCamMeasurement = cameraData.at(0);
    cameraData.erase(cameraData.begin());

    FrameMarkersData firstFrameMarkersData = getRotationTraslationFromFrame(firstCamMeasurement,
         dictionary, cameraMatrix, distCoeffs);

    int stateSize = 13;
    int measurementSize = 13;
    int cameraIgnoredTimes = 0;
    int indexCamera = 0;
    int indexImu = getImuStartingIdexBaseOnCamera(cameraData, imuReadVector);
    size_t ites = imuReadVector.size() + cameraData.size();

    float deltaTCam = -1;
    float oldDeltaTCam = 0;
    float deltaTImu = -1;
    float oldDeltaTImu = 0;
    float printErrorX = 0;

    Eigen::Vector3d accBias{1.35588e-05, 0.000179604, 0.000180296};
    Eigen::Vector3d gyroBias{0.000333556, 0.000200133, 0.000593729};

    bool isCameraNext = true;
    bool lastOneWasCamera = true;
    bool firstRun = true;

    Eigen::Vector3d oldCamT;
    oldCamT.setZero();

    Eigen::Vector3d oldCamAngSpeed;
    oldCamAngSpeed.setZero();

    Eigen::Vector3d oldCamLinearSpeed;
    oldCamLinearSpeed.setZero();

    Eigen::Quaterniond oldCamQuat;
    oldCamQuat.setIdentity();

    Eigen::Vector3d deltaPos;
    deltaPos.setZero();

    Eigen::Vector3d deltaVel;
    deltaVel.setZero();

    cv::KalmanFilter KF(stateSize, measurementSize, 0);
    cv::Mat measurementNoiseCovCam = cv::Mat::eye(measurementSize, measurementSize, CV_32F) * 1e-3;
    cv::Mat measurementNoiseCovImu = cv::Mat::eye(measurementSize, measurementSize, CV_32F) * 1e-2;
    
    Eigen::MatrixXd measurementCam(measurementSize, 1);
    measurementCam.setZero();

    Eigen::MatrixXd measurementImu(measurementSize, 1);
    measurementImu.setZero();

    initKalmanFilter(KF, stateSize);

    Eigen::Matrix4d Gci;
    Gci << 
    -0.99787874, 0.05596833, -0.03324997, 0.09329806,
    0.03309321, -0.00372569, -0.99944533, 0.01431868,
    -0.05606116, -0.99842559, 0.00186561, -0.12008699,
    0.0, 0.0, 0.0, 1.0;

    /* Gci << 
    -0.99787874, 0.05596833, -0.03324997, 0.09329806,
    0.03309321, -0.00372569, -0.99944533, 0.01431868,
    -0.05606116, -0.99842559, 0.00186561, -0.06008699,
    0.0, 0.0, 0.0, 1.0; */

    Eigen::Matrix4d Gmc;
    Eigen::Matrix4d oldGmc;
    Eigen::Matrix4d Gcm;
    Eigen::Matrix4d Gi;
    Eigen::Matrix4d Gmi;
    Eigen::Matrix4d Gci_inv;
    Eigen::Matrix4d Gcm_second;
    Eigen::Matrix4d Gmc_second;
    Eigen::Matrix4d Gmm_secondToBase;


    Eigen::Matrix4d Gti; // Constant that converts IMU measurenment to IMU pos in camera world.
    Eigen::Matrix4d Gni; // Constant used to convert camera Ghi to IMU Ghi.

    Gmc.setIdentity();
    oldGmc.setIdentity();
    Gcm.setIdentity();
    Gi.setIdentity();
    Gmi.setIdentity();
    Gti.setIdentity();
    Gni.setIdentity();
    Gcm_second.setIdentity();
    Gmc_second.setIdentity();
    Gmm_secondToBase.setIdentity();

    Gci_inv = invertG(Gci);

    CameraInput tempCameraData = cameraData.at(indexCamera);
    ImuInputJetson tempImuData = imuReadVector.at(indexImu);

    int sizeImuData = imuReadVector.size();
    int sizeCameraData = cameraData.size();

    Eigen::Matrix3d firstImuRot;
    Eigen::Vector3d firstImuGyro;
    Eigen::Vector3d firstImuAcc;

    Eigen::Vector3d accFiltered;
    Eigen::Vector3d gyroFiltered;
    accFiltered.setZero();
    gyroFiltered.setZero();

    FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(tempCameraData,
         dictionary, cameraMatrix, distCoeffs);

    Eigen::Vector3d gyro{tempImuData.gyroVect.x, tempImuData.gyroVect.y, tempImuData.gyroVect.z};
    Eigen::Vector3d acc{tempImuData.accVect.x, tempImuData.accVect.y, tempImuData.accVect.z};

    gyro = gyro - gyroBias;
    acc = acc - accBias;

    applyIIRFilterToAccAndGyro(acc, gyro, accFiltered, gyroFiltered);

    vectorImuGyro.push_back(gyro);
    vectorIIRGyro.push_back(gyroFiltered);

    vectorImuAcc.push_back(acc);
    vectorIIRAcc.push_back(accFiltered);

    Eigen::Quaterniond imuQuat{
        tempImuData.rotQuat[0],
        tempImuData.rotQuat[1],
        tempImuData.rotQuat[2],
        tempImuData.rotQuat[3]
    };
    imuQuat.normalize();

    if(imuQuat.w() < 0)
        imuQuat.coeffs() *= -1;

    for (size_t i = 0; i < ites; i++)
    {
        if (!firstRun)
        {
            if (cameraData.at(indexCamera + 1).time == imuReadVector.at(indexImu + 1).time)
            {
                isCameraNext = true;
                indexCamera++;
                indexImu++;
            }
            else if (cameraData.at(indexCamera + 1).time < imuReadVector.at(indexImu + 1).time)
            {
                isCameraNext = true;
                indexCamera++;
            }
            else
            {
                isCameraNext = false;
                indexImu++;
            }

            if(isCameraNext)
            {
                CameraInput tempCameraData = cameraData.at(indexCamera);
                FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(tempCameraData,
                    dictionary, cameraMatrix, distCoeffs);

                Eigen::Vector3d camT;
                Eigen::Matrix3d camRot;
                Eigen::Quaterniond camQuat;

                 if(lastOneWasCamera)
                    deltaTCam = tempCameraData.time - oldDeltaTCam;
                else
                    deltaTCam = tempCameraData.time - oldDeltaTImu;
                deltaTCam /= 1000;

                float deltaTCamMeasurement = tempCameraData.time - cameraData.at(indexCamera - 1).time;
                deltaTCamMeasurement /= 1000;

                Eigen::Vector3d w;
                w.setZero();

                for (size_t j = 0; j < frameMarkersData.markerIds.size(); j++)
                {
                    Eigen::Vector3d markerPos = Gcm.block<3, 1>(0, 3);
                    //vectorOfMarkers.push_back(markerPos);
                
                    if(frameMarkersData.markerIds[j] == BASE_MARKER_ID)
                    {
                        Gcm = getGFromFrameMarkersData(frameMarkersData, j);
                        //vectorOfMarkers.push_back(Eigen::Vector3d{0,0,0});
                    }
                    else
                    {
                        Eigen::Matrix4d tempG = getGFromFrameMarkersData(frameMarkersData, j);

                        for (size_t k = 0; k < vectorOfTransforms.size(); k++)
                        {
                            if(vectorOfTransforms[k].secundaryMarkerId == frameMarkersData.markerIds[j])
                            {
                                Gcm = tempG * vectorOfTransforms[k].G;

                                //Eigen::Vector3d markerPos = invertG(vectorOfTransforms[k].G).block<3, 1>(0, 3);
                                //vectorOfMarkers.push_back(markerPos);
                                break;
                            }
                        }
                    }

                    Gmc = invertG(Gcm);

                    camT = Gmc.block<3,1>(0,3);
                    camRot = Gmc.block<3,3>(0,0);
                    camQuat = Eigen::Quaterniond(camRot);

                    w = getAngularVelocityFromTwoQuats(oldCamQuat, camQuat, deltaTCamMeasurement);
                    std::cout << "w: " << std::endl << w << std::endl << std::endl;

                    /* float angSpeedDiffNorm = (oldCamAngSpeed - w).norm();

                    if(angSpeedDiffNorm > 10)
                    {
                        cameraIgnoredTimes++;
                        camT = oldCamT;
                        camQuat = oldCamQuat;
                    
                        w = oldCamAngSpeed;
                    } */

                    measurementCam.block<3,1>(0,0) = camT; // Traslation
                    measurementCam(3) = camQuat.w();
                    measurementCam(4) = camQuat.x();
                    measurementCam(5) = camQuat.y();
                    measurementCam(6) = camQuat.z();
                    measurementCam(7) = (measurementCam(0) - oldCamT(0)) / deltaTCamMeasurement; // traslation speed (x)
                    measurementCam(8) = (measurementCam(1) - oldCamT(1)) / deltaTCamMeasurement; // traslation speed (y)
                    measurementCam(9) = (measurementCam(2) - oldCamT(2)) / deltaTCamMeasurement; // traslation speed (z)
                    measurementCam(10) = w.x(); // angular speed (x)
                    measurementCam(11) = w.y(); // angular speed (y)
                    measurementCam(12) = w.z(); // angular speed (z)

                    cv::Mat tempMeasurement = convertEigenMatToOpencvMat(measurementCam);
                    correct(KF, tempMeasurement, measurementNoiseCovCam);
                    fixStateQuaternion(KF, "post");

                    std::cout << "State Pre: " << std::endl << KF.statePre << std::endl << std::endl;
                    std::cout << "statePost: " << std::endl << KF.statePost << std::endl << std::endl;
                    std::cout << "measurementCam: " << std::endl << measurementCam << std::endl << std::endl;
                }

                updateTransitionMatrixFusion(KF, deltaTCam, stateSize, w);

                predict(KF);
                fixStateQuaternion(KF, "pre");
                
                oldDeltaTCam = tempCameraData.time;
                oldCamT = camT;
                oldCamQuat = camQuat;
                oldCamAngSpeed = measurementCam.block<3,1>(10,0);
                oldGmc = Gmc;
                
                Eigen::Vector3d camLinearSpeed{KF.statePre.at<float>(7), KF.statePre.at<float>(8), KF.statePre.at<float>(9)};
                Eigen::Vector3d newLinearAcc = (camLinearSpeed - oldCamLinearSpeed) / deltaTCam;
                Eigen::Vector4d homogeneusNewLinearSpeed;
                homogeneusNewLinearSpeed << newLinearAcc, 1;

                oldCamLinearSpeed = camLinearSpeed;

                // Reset IMU information.

                Eigen::Quaterniond stateQuat(KF.statePre.at<float>(3), KF.statePre.at<float>(4),
                    KF.statePre.at<float>(5), KF.statePre.at<float>(6));
                Eigen::Matrix<double, 3, 3> stateRot = stateQuat.toRotationMatrix();
                Eigen::Vector3d statePos(KF.statePre.at<float>(0), KF.statePre.at<float>(1), KF.statePre.at<float>(2));
               
                Eigen::Matrix4d resetGmi;
                resetGmi.setIdentity();
                resetGmi.block<3, 3>(0, 0) = stateRot;
                resetGmi.block<3, 1>(0, 3) = statePos;
                
                Gmi = Gci * resetGmi;
                
                Eigen::Matrix4d stateGhi;
                stateGhi << 
                    0,                              -KF.statePre.at<float>(12),    KF.statePre.at<float>(11),     KF.statePre.at<float>(7),
                    KF.statePre.at<float>(12),     0,                              -KF.statePre.at<float>(10),    KF.statePre.at<float>(8),
                    -KF.statePre.at<float>(11),    KF.statePre.at<float>(10),     0,                              KF.statePre.at<float>(9),
                    0,0,0,0;
                
                Eigen::Matrix4d imuGhi = Gci * stateGhi * invertG(Gci);

                Eigen::Vector4d homogeneusImuAcc = Gci * homogeneusNewLinearSpeed;

                Eigen::Vector3d oldDeltaVel = deltaVel;

                deltaPos = Eigen::Vector3d(Gmi.block<3, 1>(0, 3)); // Delete Eigen::Vector3d
                firstImuRot = Gmi.block<3, 3>(0, 0);
                deltaVel = Eigen::Vector3d(imuGhi.block<3, 1>(0, 3));
                firstImuGyro = Eigen::Vector3d{imuGhi(2, 1), imuGhi(0, 2), imuGhi(1, 0)};
                firstImuAcc = Eigen::Vector3d{homogeneusImuAcc(0), homogeneusImuAcc(1), homogeneusImuAcc(2)};
                
                lastOneWasCamera = true;                                  
            }
            else
            {
                ImuInputJetson tempImuData = imuReadVector.at(indexImu);

                Eigen::Vector3d gyro{tempImuData.gyroVect.x, tempImuData.gyroVect.y, tempImuData.gyroVect.z};
                Eigen::Vector3d acc{tempImuData.accVect.x, tempImuData.accVect.y, tempImuData.accVect.z};
                
                gyro = gyro - gyroBias;
                acc = acc - accBias;

                applyIIRFilterToAccAndGyro(acc, gyro, accFiltered, gyroFiltered);

                vectorImuGyro.push_back(gyro);
                vectorIIRGyro.push_back(gyroFiltered);

                vectorImuAcc.push_back(acc);
                vectorIIRAcc.push_back(accFiltered);

                Eigen::Quaterniond imuQuat{
                    tempImuData.rotQuat[0],
                    tempImuData.rotQuat[1],
                    tempImuData.rotQuat[2],
                    tempImuData.rotQuat[3]
                };
                fixQuatEigen(imuQuat);

                if (lastOneWasCamera)
                    deltaTImu = tempImuData.time - oldDeltaTCam;
                else
                    deltaTImu = tempImuData.time - oldDeltaTImu;

                deltaTImu /= 1000;

                /////////////////////////// Measurenment ////////////////////////////////////

                Eigen::Matrix3d imuRot = imuQuat.toRotationMatrix();

                Eigen::Matrix3d imuRotFromNewOrigen = firstImuRot.transpose() * imuRot;
                Eigen::Vector3d imuGyroFromNewOrigen = gyroFiltered - firstImuGyro;
                Eigen::Vector3d imuAccFromNewOrigen = accFiltered - firstImuAcc;
                //Eigen::Matrix<double, 3, 3> imuRotFromNewOrigen = imuRot;

                Eigen::Quaterniond imuQuatNewOrigen(imuRotFromNewOrigen);
                fixQuatEigen(imuQuatNewOrigen);

                float deltaTForIntegration = tempImuData.time - imuReadVector.at(indexImu - 1).time;
                deltaTForIntegration /= 1000;
                
                imuPreintegration(deltaTForIntegration, imuAccFromNewOrigen, imuGyroFromNewOrigen, deltaPos, deltaVel, imuRotFromNewOrigen);

                measurementImu(0,0) = imuGyroFromNewOrigen[0];
                measurementImu(1,0) = imuGyroFromNewOrigen[1];
                measurementImu(2,0) = imuGyroFromNewOrigen[2];
                measurementImu(3,0) = imuQuatNewOrigen.w();
                measurementImu(4,0) = imuQuatNewOrigen.x();
                measurementImu(5,0) = imuQuatNewOrigen.y();
                measurementImu(6,0) = imuQuatNewOrigen.z();
                measurementImu(7,0) = deltaVel[0];
                measurementImu(8,0) = deltaVel[1];
                measurementImu(9,0) = deltaVel[2];
                measurementImu(10,0) = deltaPos[0];
                measurementImu(11,0) = deltaPos[1];
                measurementImu(12,0) = deltaPos[2];

                std::cout << "measurementImu: " << std::endl <<  measurementImu << std::endl << std::endl;

                oldDeltaTImu = tempImuData.time;

                /////////////////////////// Update ////////////////////////////////////

                Eigen::MatrixXd h(measurementSize, 1);
                Eigen::MatrixXd H(measurementSize, stateSize);
                h.setZero();
                H.setZero();

                calculateHAndJacobian(KF, Gci, Gci_inv, h, H);

                std::cout << "h: " << std::endl << h << std::endl << std::endl;
                std::cout << "H: " << std::endl << H << std::endl << std::endl;

                correctIMU_EKF(KF, measurementNoiseCovImu, measurementImu, h, H);
                fixStateQuaternion(KF, "post");

                Eigen::Vector3d gyroInCameraFrame = multiplyVectorByG(Gci_inv, gyroFiltered);                

                updateTransitionMatrixFusion(KF, deltaTImu, stateSize, gyroInCameraFrame);

                predict(KF);
                fixStateQuaternion(KF, "pre");

                std::cout << "State Pre: " << std::endl << KF.statePre << std::endl << std::endl;
                std::cout << "State Post: " << std::endl << KF.statePost << std::endl << std::endl;

                oldCamLinearSpeed = Eigen::Vector3d{ KF.statePost.at<float>(7),  KF.statePost.at<float>(8),  KF.statePost.at<float>(9)};

                lastOneWasCamera = false;
            }
        }
        else
        {
            int indexBaseMarker = getBaseMarkerIndex(frameMarkersData.markerIds, BASE_MARKER_ID);

            Gcm = getGFromFrameMarkersData(frameMarkersData, indexBaseMarker);
            Gmc = invertG(Gcm);

            vectorOfTransforms = getAllTransformsBetweenMarkers(frameMarkersData, Gcm, indexBaseMarker);

            Eigen::Vector3d camT = Gmc.block<3,1>(0,3);
            Eigen::Matrix3d camRot = Gmc.block<3,3>(0,0);
            Eigen::Quaterniond camQuat(camRot);
            fixQuatEigen(camQuat);

            int indexFirstBaseMarker = getBaseMarkerIndex(firstFrameMarkersData.markerIds, BASE_MARKER_ID);
            Eigen::Matrix4d firstGcm = getGFromFrameMarkersData(firstFrameMarkersData, indexFirstBaseMarker);
            Eigen::Matrix4d firstGmc = invertG(firstGcm);
            Eigen::Vector3d firstCamT = firstGmc.block<3,1>(0,3);
            Eigen::Matrix3d firstCamRot = firstGmc.block<3,3>(0,0);
            Eigen::Quaterniond firstCamQuat(firstCamRot);
            fixQuatEigen(firstCamQuat);

            float deltaTCam = tempCameraData.time - firstCamMeasurement.time;
            deltaTCam /= 1000;

            Eigen::Vector3d tempAngVel = getAngularVelocityFromTwoQuats(firstCamQuat, camQuat, deltaTCam);

            measurementCam(0) = camT.x(); // Traslation x
            measurementCam(1) = camT.y(); // Traslation y
            measurementCam(2) = camT.z(); // Traslation z
            measurementCam(3) = camQuat.w();
            measurementCam(4) = camQuat.x();
            measurementCam(5) = camQuat.y();
            measurementCam(6) = camQuat.z();
            measurementCam(7) = (camT.x() - firstCamT.x())/deltaTCam; // traslation speed (x)
            measurementCam(8) = (camT.y() - firstCamT.y())/deltaTCam; // traslation speed (y)
            measurementCam(9) = (camT.z() - firstCamT.z())/deltaTCam; // traslation speed (z)
            measurementCam(10) = tempAngVel.x(); // angular speed (x)
            measurementCam(11) = tempAngVel.y(); // angular speed (y)
            measurementCam(12) = tempAngVel.z(); // angular speed (z)

            KF.statePost.at<float>(0) = measurementCam(0);
            KF.statePost.at<float>(1) = measurementCam(1);
            KF.statePost.at<float>(2) = measurementCam(2);
            KF.statePost.at<float>(3) = measurementCam(3);
            KF.statePost.at<float>(4) = measurementCam(4);
            KF.statePost.at<float>(5) = measurementCam(5);
            KF.statePost.at<float>(6) = measurementCam(6);
            KF.statePost.at<float>(7) = measurementCam(7);
            KF.statePost.at<float>(8) = measurementCam(8);
            KF.statePost.at<float>(9) = measurementCam(9);
            KF.statePost.at<float>(10) = measurementCam(10);
            KF.statePost.at<float>(11) = measurementCam(11);
            KF.statePost.at<float>(12) = measurementCam(12);
            
            std::cout << "State Post: " << std::endl << KF.statePost << std::endl << std::endl;

            oldCamT = camT;
            oldCamQuat = camQuat;
            oldCamAngSpeed = measurementCam.block<3,1>(10,0);
            oldDeltaTCam = tempCameraData.time;
            oldGmc = Gmc;
            oldCamLinearSpeed = Eigen::Vector3d{ KF.statePost.at<float>(7),  KF.statePost.at<float>(8),  KF.statePost.at<float>(9)};
            
            deltaTImu = tempImuData.time - imuReadVector.at(indexImu - 1).time;
            deltaTImu /= 1000;

            Eigen::Matrix3d imuRot = imuQuat.toRotationMatrix();
            firstImuRot = imuRot;
            firstImuGyro = gyroFiltered;
            firstImuAcc = accFiltered;
            imuPreintegration(deltaTImu, accFiltered, gyroFiltered, deltaPos, deltaVel, imuRot);

            measurementImu(0,0) = gyroFiltered[0];
            measurementImu(1,0) = gyroFiltered[1];
            measurementImu(2,0) = gyroFiltered[2];
            measurementImu(3,0) = imuQuat.w();
            measurementImu(4,0) = imuQuat.x();
            measurementImu(5,0) = imuQuat.y();
            measurementImu(6,0) = imuQuat.z();
            measurementImu(7,0) = deltaVel[0];
            measurementImu(8,0) = deltaVel[1];
            measurementImu(9,0) = deltaVel[2];
            measurementImu(10,0) = deltaPos[0];
            measurementImu(11,0) = deltaPos[1];
            measurementImu(12,0) = deltaPos[2];

            oldDeltaTImu = tempImuData.time;
            
            firstRun = false;
        }

        Eigen::Vector3d PosOriginal{measurementCam(0), measurementCam(1), measurementCam(2)};
        Eigen::Vector3d PosKF{KF.statePost.at<float>(0), KF.statePost.at<float>(1), KF.statePost.at<float>(2)};
        Eigen::Vector3d PosKFPre{KF.statePre.at<float>(0), KF.statePre.at<float>(1), KF.statePre.at<float>(2)};

        Eigen::Vector4d quatMeasurement{measurementCam(3), measurementCam(4), measurementCam(5), measurementCam(6)};
        Eigen::Vector4d quatKF{KF.statePost.at<float>(3), KF.statePost.at<float>(4), KF.statePost.at<float>(5), KF.statePost.at<float>(6)};

        Eigen::Vector3d PosError = PosOriginal - PosKF;
        float errorMagnitude = PosError.norm();
        Eigen::Vector3d printPosError{printErrorX, 0, errorMagnitude};

        float errorMagnitudeQuat = (quatMeasurement - quatKF).norm();
        Eigen::Vector3d printQuatError{printErrorX, 0, errorMagnitudeQuat};

        printErrorX += 0.1;
        
        /* if (vectorOfPointsOne.size() > 50)
        {
            vectorOfPointsOne.erase(vectorOfPointsOne.begin());
            vectorOfPointsTwo.erase(vectorOfPointsTwo.begin());
            vectorOfMarkers.erase(vectorOfMarkers.begin());
        } */

       /*  vectorOfPointsOne.push_back(PosOriginal);
        vectorOfPointsTwo.push_back(PosKF);
        vectorOfMarkers.push_back(PosKFPre); */
        
        /* if(lastOneWasCamera)
            timeStamps.push_back(oldDeltaTCam);
        else
            timeStamps.push_back(oldDeltaTImu);*/

        vectorOfPointsOne.push_back(printQuatError);
        vectorOfPointsTwo.push_back(Eigen::Vector3d{0,0,0});
        
        //gnuPrintImuPreintegration(output, vectorOfPointsOne, vectorOfPointsTwo, vectorOfMarkers);

        //vectorCamMeasurenments.push_back(measurementCam);
        //vectorStates.push_back(convertOpencvMatToEigenMat(KF.statePost));

        if (indexCamera == (int)cameraData.size() - 1)
        {
            //pointsDataWrite(vectorCamMeasurenments, vectorStates, timeStamps, "cameraVsKalman.csv");
            //pointsDataWrite(vectorImuGyro, vectorIIRGyro, timeStamps, "imuGyroVsIIR.csv");
            //pointsDataWrite(vectorImuAcc, vectorIIRAcc, timeStamps, "imuAccVsIIR.csv");

            break;
        }   
    }

    float averageError = 0;

    for (size_t i = 0; i < vectorOfPointsOne.size(); i++)
    {
        averageError += vectorOfPointsOne[i].z();
    }

    averageError /= vectorOfPointsOne.size();

    std::cout << "averageError: " << averageError << std::endl;

    pclose(output);

    std::cout << "cameraIgnoredTimes: " << std::endl << cameraIgnoredTimes << std::endl << std::endl;

}

void runIMUPrediction()
{
    Eigen::Vector3d deltaPos;
    deltaPos.setZero();

    Eigen::Vector3d deltaVel;
    deltaVel.setZero();

    Eigen::Matrix3d imuRot;
    imuRot.setIdentity();

    Eigen::Matrix3d wHat;
    wHat.setIdentity();

    Eigen::Vector3d calVelocity;
    calVelocity.setZero();

    Eigen::Matrix4d chi;
    chi.setZero();

    Eigen::Matrix4d GImu;
    GImu.setIdentity();

    Eigen::Matrix4d GImuOld;
    GImuOld.setIdentity();

    Eigen::Vector3d oldAngularVelocity;
    oldAngularVelocity.setZero();

    Eigen::Vector3d oldLinealAcc;
    oldLinealAcc.setZero();

    Eigen::Quaterniond oldQuat;
    oldQuat.setIdentity();

    cv::KalmanFilter KF(23, 23, 0);

    bool firstRun = true;
    bool validGyro = true;
    bool validAcc = true;
    float deltaT = -1;
    float oldDeltaT = 0;
    int ignoredTimes = 0;
    int validKalmanItes = 0;
    float firstPositionMagnitude = 0;
    float lastPositionMagnitude = 0;

    Eigen::Matrix<double, 23, 1> measurement;
    measurement.setZero();

    initKalmanFilter(KF, 23);

    std::vector<CameraInput> cameraData = readDataCamera();
    std::vector<ImuInputJetson> imuReadVector = readDataIMUJetson();

    int imuIndex = getImuStartingIdexBaseOnCamera(cameraData, imuReadVector);

    std::vector<Eigen::Vector3d> vectorOfPointsOne;
    std::vector<Eigen::Vector3d> vectorOfPointsTwo;
    std::vector<Eigen::Vector3d> vectorOfMarkers;

    Eigen::Vector3d originPos;
    originPos.setZero();
    vectorOfMarkers.push_back(originPos);

    std::vector<float> vectorOfMagnitudesGyro;
    std::vector<float> vectorOfMagnitudesAcc;

    FILE *output;
    output = popen("gnuplot", "w");

    for (size_t i = imuIndex; i < imuReadVector.size(); i++)
    {
        ImuInputJetson tempImuData = imuReadVector.at(i);  

        Eigen::Vector3d gyro{tempImuData.gyroVect.x, tempImuData.gyroVect.y, tempImuData.gyroVect.z};
        Eigen::Vector3d acc{tempImuData.accVect.x, tempImuData.accVect.y, tempImuData.accVect.z};
        Eigen::Quaterniond imuQuat{
            tempImuData.rotQuat[0],
            tempImuData.rotQuat[1],
            tempImuData.rotQuat[2],
            tempImuData.rotQuat[3]
        };

        vectorOfMagnitudesGyro.push_back(gyro.norm());
        vectorOfMagnitudesAcc.push_back(acc.norm());

        if (!firstRun)
        {
            if(gyro.norm() < THRESHOLD_IMU_GYRO)
            {
                validGyro = false;
            }
            else
            {
                validGyro = true;
            }

            if(acc.norm() < THRESHOLD_IMU_ACC)
            {
                validAcc = false;
            }
            else
            {
                validAcc = true;
            }

            if (validGyro || validAcc)
            {
                /////////////////////////// Prediction ////////////////////////////////////
            
                updateTransitionMatrixIMU(KF, measurement, deltaT);

                predict(KF);

                Eigen::Quaterniond quatKFPre{
                    KF.statePre.at<float>(3),
                    KF.statePre.at<float>(4),
                    KF.statePre.at<float>(5),
                    KF.statePre.at<float>(6)
                };

                quatKFPre.normalize();
                KF.statePre.at<float>(3) = quatKFPre.w();
                KF.statePre.at<float>(4) = quatKFPre.x();
                KF.statePre.at<float>(5) = quatKFPre.y();
                KF.statePre.at<float>(6) = quatKFPre.z();

                std::cout << "KF.statePre:\n" << KF.statePre << endl << endl;

                deltaT = tempImuData.time - oldDeltaT;
                deltaT /= 1000;

                /////////////////////////// Measurenment ////////////////////////////////////

                Eigen::Matrix3d imuRot = imuQuat.toRotationMatrix();
                imuPreintegration(deltaT, acc, gyro, deltaPos, deltaVel, imuRot);

                measurement(0,0) = gyro[0];
                measurement(1,0) = gyro[1];
                measurement(2,0) = gyro[2];
                measurement(3,0) = imuQuat.w();
                measurement(4,0) = imuQuat.x();
                measurement(5,0) = imuQuat.y();
                measurement(6,0) = imuQuat.z();
                measurement(7,0) = deltaVel[0];
                measurement(8,0) = deltaVel[1];
                measurement(9,0) = deltaVel[2];
                measurement(10,0) = deltaPos[0];
                measurement(11,0) = deltaPos[1];
                measurement(12,0) = deltaPos[2];
                measurement(13,0) = (gyro[0] - oldAngularVelocity[0]) / deltaT;
                measurement(14,0) = (gyro[1] - oldAngularVelocity[1]) / deltaT;
                measurement(15,0) = (gyro[2] - oldAngularVelocity[2]) / deltaT;
                measurement(16,0) = acc[0];
                measurement(17,0) = acc[1];
                measurement(18,0) = acc[2];
                measurement(19,0) = (imuQuat.w() - oldQuat.w()) / deltaT;
                measurement(20,0) = (imuQuat.x() - oldQuat.x()) / deltaT;
                measurement(21,0) = (imuQuat.y() - oldQuat.y()) / deltaT;
                measurement(22,0) = (imuQuat.z() - oldQuat.z()) / deltaT;

                std::cout << "measurement:\n" << measurement << endl << endl;

                oldAngularVelocity = gyro;
                oldLinealAcc = acc;
                oldDeltaT = tempImuData.time;
                oldQuat = imuQuat;

                /////////////////////////// Update ////////////////////////////////////

                correctIMU(KF, measurement);

                Eigen::Quaterniond quatKFPost{
                    KF.statePost.at<float>(3),
                    KF.statePost.at<float>(4),
                    KF.statePost.at<float>(5),
                    KF.statePost.at<float>(6)
                };

                quatKFPost.normalize();
                KF.statePost.at<float>(3) = quatKFPost.w();
                KF.statePost.at<float>(4) = quatKFPost.x();
                KF.statePost.at<float>(5) = quatKFPost.y();
                KF.statePost.at<float>(6) = quatKFPost.z();

                Eigen::Vector3d tempPos{
                    KF.statePost.at<float>(10),
                    KF.statePost.at<float>(11),
                    KF.statePost.at<float>(12)
                };

                lastPositionMagnitude = tempPos.norm();

                validKalmanItes++;

                if (validKalmanItes == 10)
                {
                    std::cout << "First Position Magnitude: " << firstPositionMagnitude << std::endl;
                    std::cout << "Last Position Magnitude: " << lastPositionMagnitude << std::endl;
                    std::cout << "Position Difference: " << lastPositionMagnitude - firstPositionMagnitude << std::endl;
                }

                std::cout << "KF.statePost:\n" << KF.statePost << endl << endl;
            }
            else
            {
                ignoredTimes++;
            }
        }
        else
        {
            deltaT = tempImuData.time - imuReadVector.at(i-1).time;
            deltaT /= 1000;

            Eigen::Matrix3d imuRot = imuQuat.toRotationMatrix();
            imuPreintegration(deltaT, acc, gyro, deltaPos, deltaVel, imuRot);

            measurement(0,0) = gyro[0];
            measurement(1,0) = gyro[1];
            measurement(2,0) = gyro[2];
            measurement(3,0) = imuQuat.w();
            measurement(4,0) = imuQuat.x();
            measurement(5,0) = imuQuat.y();
            measurement(6,0) = imuQuat.z();
            measurement(7,0) = deltaVel[0];
            measurement(8,0) = deltaVel[1];
            measurement(9,0) = deltaVel[2];
            measurement(10,0) = deltaPos[0];
            measurement(11,0) = deltaPos[1];
            measurement(12,0) = deltaPos[2];
            measurement(13,0) = 0; // angular acc x
            measurement(14,0) = 0; // angular acc y
            measurement(15,0) = 0; // angular acc z
            measurement(16,0) = acc[0]; // lineal acc x
            measurement(17,0) = acc[1]; // lineal acc y
            measurement(18,0) = acc[2]; // lineal acc z
            measurement(19,0) = 0; // quat vel w
            measurement(20,0) = 0; // quat vel x
            measurement(21,0) = 0; // quat vel y
            measurement(22,0) = 0; // quat vel z

            KF.statePost.at<float>(0) = measurement(0,0);
            KF.statePost.at<float>(1) = measurement(1,0);
            KF.statePost.at<float>(2) = measurement(2,0);
            KF.statePost.at<float>(3) = measurement(3,0);
            KF.statePost.at<float>(4) = measurement(4,0);
            KF.statePost.at<float>(5) = measurement(5,0);
            KF.statePost.at<float>(6) = measurement(6,0);
            KF.statePost.at<float>(7) = measurement(7,0);
            KF.statePost.at<float>(8) = measurement(8,0);
            KF.statePost.at<float>(9) = measurement(9,0);
            KF.statePost.at<float>(10) = measurement(10,0);
            KF.statePost.at<float>(11) = measurement(11,0);
            KF.statePost.at<float>(12) = measurement(12,0);
            KF.statePost.at<float>(13) = measurement(13,0);
            KF.statePost.at<float>(14) = measurement(14,0);
            KF.statePost.at<float>(15) = measurement(15,0);
            KF.statePost.at<float>(16) = measurement(16,0);
            KF.statePost.at<float>(17) = measurement(17,0);
            KF.statePost.at<float>(18) = measurement(18,0);
            KF.statePost.at<float>(19) = measurement(19,0);
            KF.statePost.at<float>(20) = measurement(20,0);
            KF.statePost.at<float>(21) = measurement(21,0);
            KF.statePost.at<float>(22) = measurement(22,0);

            oldAngularVelocity = gyro;
            oldLinealAcc = acc;
            oldQuat = imuQuat;
            oldDeltaT = tempImuData.time;

            firstPositionMagnitude = deltaPos.norm();

            firstRun = false;
        }
        
        if (validGyro || validAcc)
        {
            GImu.block<3,3>(0,0) = imuRot;
            GImu.block<3,1>(0,3) = deltaPos;

            Eigen::Quaterniond tempOriginalQuat = {tempImuData.rotQuat[0], tempImuData.rotQuat[1], tempImuData.rotQuat[2], tempImuData.rotQuat[3]};
            Eigen::Vector3d rotationVectorOriginal = QuatToRotVectEigen(tempOriginalQuat);

            Eigen::Quaterniond tempQuat = {KF.statePost.at<float>(3), KF.statePost.at<float>(4), KF.statePost.at<float>(5), KF.statePost.at<float>(6)};
            Eigen::Vector3d rotationVector = QuatToRotVectEigen(tempQuat);

            float quatDiff = (tempOriginalQuat.coeffs() - tempQuat.coeffs()).norm();

            std::cout << "Quat Diff: " << quatDiff << std::endl;

            Eigen::Vector3d GyroOriginal{measurement(0,0), measurement(1,0), measurement(2,0)};
            Eigen::Vector3d GyroKF{KF.statePost.at<float>(0), KF.statePost.at<float>(1), KF.statePost.at<float>(2)};

            Eigen::Vector3d VelOriginal{measurement(7,0), measurement(8,0), measurement(9,0)};
            Eigen::Vector3d VelKF{KF.statePost.at<float>(7), KF.statePost.at<float>(8), KF.statePost.at<float>(9)};

            Eigen::Vector3d PosOriginal{measurement(10,0), measurement(11,0), measurement(12,0)};
            Eigen::Vector3d PosKF{KF.statePost.at<float>(10), KF.statePost.at<float>(11), KF.statePost.at<float>(12)};

            Eigen::Vector3d AccAngOriginal{measurement(13,0), measurement(14,0), measurement(15,0)};
            Eigen::Vector3d AccAngKF{KF.statePost.at<float>(13), KF.statePost.at<float>(14), KF.statePost.at<float>(15)};

            Eigen::Vector3d AccLinOriginal{measurement(16,0), measurement(17,0), measurement(18,0)};
            Eigen::Vector3d AccLinKF{KF.statePost.at<float>(16), KF.statePost.at<float>(17), KF.statePost.at<float>(18)};

            if (vectorOfPointsOne.size() > 50)
            {
                vectorOfPointsOne.erase(vectorOfPointsOne.begin());
                vectorOfPointsTwo.erase(vectorOfPointsTwo.begin());
            }

            vectorOfPointsOne.push_back(rotationVectorOriginal);
            vectorOfPointsTwo.push_back(rotationVector);

            gnuPrintImuPreintegration(output, vectorOfPointsOne, vectorOfPointsTwo, vectorOfMarkers);
        }
    }

    float maxGyro = -10;
    float maxAcc = -10;
    float minGyro = 10;
    float minAcc = 10;
    float sumGyro = 0;
    float sumAcc = 0;

    for (size_t i = 0; i < vectorOfMagnitudesGyro.size(); i++)
    {
        float tempGyro = vectorOfMagnitudesGyro[i];
        float tempAcc = vectorOfMagnitudesAcc[i];

        if (tempGyro > maxGyro)
        {
            maxGyro = tempGyro;
        }

        if (tempGyro < minGyro)
        {
            minGyro = tempGyro;
        }

        if (tempAcc > maxAcc)
        {
            maxAcc = tempAcc;
        }

        if (tempAcc < minAcc)
        {
            minAcc = tempAcc;
        }

        sumGyro += tempGyro;
        sumAcc += tempAcc;
    }

    std::cout << "Average Gyro: " << sumGyro / vectorOfMagnitudesGyro.size() << std::endl;
    std::cout << "Max Gyro: " << maxGyro << std::endl;
    std::cout << "Min Gyro: " << minGyro << std::endl;
    
    std::cout << "Average Acc: " << sumAcc / vectorOfMagnitudesAcc.size() << std::endl;
    std::cout << "Max Acc: " << maxAcc << std::endl;
    std::cout << "Min Acc: " << minAcc << std::endl;

    std::cout << "Ignored Times: " << ignoredTimes << std::endl;

    std::cout << "First Position Magnitude: " << firstPositionMagnitude << std::endl;
    std::cout << "Last Position Magnitude: " << lastPositionMagnitude << std::endl;
    std::cout << "Position Difference: " << lastPositionMagnitude - firstPositionMagnitude << std::endl;

    /*std::vector<float> normalizedOriginals;
    std::vector<float> normalizedPredictions;

    normalizeDataSet(vectorOfPointsOne, normalizedOriginals, 0);
    normalizeDataSet(vectorOfPointsTwo, normalizedPredictions, 0);

    std::vector<float> printProgressiveOriginals;
    std::vector<float> printProgressivePrediction;

    for (size_t i = 0; i < vectorOfPointsOne.size(); i++)
    {
        printProgressiveOriginals.push_back(normalizedOriginals[i]);
        printProgressivePrediction.push_back(normalizedPredictions[i]);
        gnuPrintImuCompareValues(output, printProgressiveOriginals, printProgressivePrediction);
    }*/

    sleep(10);
    pclose(output);
}

void runKalmanFilterCamera()
{
    bool firstRun = true;
    float deltaT = -1;
    cv::KalmanFilter KF(12, 12, 0);

    // Create measurement vector (traslation, quaternion, speeds).
    cv::Mat_<float> measurement(12, 1);
    measurement.setTo(cv::Scalar(0));

    // Create old measurement to calculate speeds.
    cv::Mat_<float> measurementOld(12, 1);
    measurementOld.setTo(cv::Scalar(0));

    // Initialize Kalman Filter.
    initKalmanFilter(KF, 12);

    std::vector<CameraInput> cameraData = readDataCamera();

    std::vector<Eigen::Vector3d> vectorOfPointsOne;
    std::vector<Eigen::Vector3d> vectorOfPointsTwo;
    std::vector<Eigen::Vector3d> vectorOfMarkers;

    Eigen::Vector3d originPos;
    originPos.setZero();
    vectorOfMarkers.push_back(originPos);

    FILE *output;
    output = popen("gnuplot", "w");

    for (size_t i = 0; i < cameraData.size(); i++)
    {
        CameraInput tempCameraData = cameraData.at(i);

        FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(tempCameraData,
         dictionary, cameraMatrix, distCoeffs);

        if (!firstRun)
        {
            predict(KF);

            deltaT = tempCameraData.time - cameraData.at(i-1).time;
            deltaT /= 1000;

            updateTransitionMatrix(KF, deltaT);

            doMeasurement(measurement, measurementOld, frameMarkersData, deltaT);
            measurementOld = measurement.clone();

            std::cout << "measurement:\n" << measurement << std::endl << std::endl;

            cv::Vec3d drawRvec = {cv::Vec3d(KF.statePost.at<float>(3), KF.statePost.at<float>(4),
                                             KF.statePost.at<float>(5))};

            cv::Mat rotMat;
            cv::Rodrigues(drawRvec, rotMat);

            std::cout << "rotMat:\n" << rotMat << std::endl << std::endl;

            correct(KF, measurement, KF.measurementNoiseCov);
            
            cv::Mat frameCopy = tempCameraData.frame.clone();

            drawAxisOnFrame(frameMarkersData.rvecs, frameMarkersData.tvecs,
                            tempCameraData.frame, cameraMatrix, distCoeffs, "Original");
            
            cv::Vec3d tempTvec = {cv::Vec3d(KF.statePost.at<float>(0), KF.statePost.at<float>(1),
                                             KF.statePost.at<float>(2))};
            std::vector<cv::Vec3d> tempTvecs;
            tempTvecs.push_back(tempTvec);

            cv::Vec3d tempRvec = {cv::Vec3d(KF.statePost.at<float>(3), KF.statePost.at<float>(4),
                                             KF.statePost.at<float>(5))};
            std::vector<cv::Vec3d> tempRvecs;
            tempRvecs.push_back(tempRvec);

            drawAxisOnFrame(tempRvecs, tempTvecs, frameCopy, cameraMatrix, distCoeffs, "EKF");

            Eigen::Vector3d graphTvecOriginal{frameMarkersData.rvecs[0][0], frameMarkersData.rvecs[0][1], frameMarkersData.rvecs[0][2]};
            Eigen::Vector3d graphTvecKF{tempRvec[0], tempRvec[1], tempRvec[2]};
            
            vectorOfPointsOne.push_back(graphTvecOriginal);
            vectorOfPointsTwo.push_back(graphTvecKF);

            gnuPrintImuPreintegration(output, vectorOfPointsOne, vectorOfPointsTwo, vectorOfMarkers);

            cv::waitKey(33);
        }
        else
        {   
            doMeasurement(measurement, measurementOld, frameMarkersData,  deltaT);
            measurementOld = measurement.clone();

            KF.statePost = measurement.clone();

            std::cout << "measurement:\n" << measurement << std::endl << std::endl;

            cv::Vec3d tempRvec = {cv::Vec3d(KF.statePost.at<float>(3), KF.statePost.at<float>(4),
                                             KF.statePost.at<float>(5))};

            cv::Mat rotMat;
            cv::Rodrigues(tempRvec, rotMat);

            std::cout << "rotMat:\n" << rotMat << std::endl << std::endl;

            firstRun = false;
        }
    }

    sleep(10);
    pclose(output);
}

// Main method that creates threads, writes and read data from files and displays data on console.
int main()
{
    bool generateNewData = false;
    bool preccessData = false;
    bool ifCalibrateIMUOnly = false;
    bool runKalmanFilterBool = true;

    if (ifCalibrateIMUOnly)
    {
        //imuCalibration();
        printIMUData();
    }
    else if (runKalmanFilterBool)
    {
        //runKalmanFilterCamera();
        //runIMUPrediction();
        /* Eigen::Vector3d accBiasVect;
        Eigen::Vector3d gyroBiasVect;

        accBiasVect.setZero();
        gyroBiasVect.setZero();
        calculateBiasAccAndGyro(accBiasVect, gyroBiasVect);

        std::cout << "Acc Bias: " << accBiasVect << std::endl;
        std::cout << "Gyro Bias: " << gyroBiasVect << std::endl;
 */
        runCameraAndIMUKalmanFilter();
    }
    else
    {
        if (generateNewData)
        {
            timeCameraStart = std::chrono::steady_clock::now();
            timeIMUStart = std::chrono::steady_clock::now();

            std::thread cameraCapture(cameraCaptureThread);
            std::thread imu(imuThreadJetson);

            cameraCapture.join();
            imu.join();

            cameraDataWrite(cameraFramesBuffer);
            //IMUDataWriteTestAxis();
            IMUDataJetsonWrite(imuDataJetsonBuffer);
        }

        if (preccessData)
        {
            std::vector<ImuInputJetson> imuReadVector = readDataIMUJetson();
            std::vector<CameraInput> cameraReadVector = readDataCamera();
            
            /*
            std::vector<glm::vec3> splinePoints = createSplinePoint(imuReadVector);
            std::vector<glm::quat> slerpPoints = createSlerpPoint(imuReadVector);

            std::vector<ImuInputJetson> imuReadVectorCopy = imuReadVector;
            std::vector<CameraInput> cameraReadVectorCopy = hardCopyCameraVector(cameraReadVector);

            std::vector<FrameMarkersData> frameMarkersDataVector = getRotationTraslationFromAllFrames(
                cameraReadVector,
                dictionary,
                cameraMatrix,
                distCoeffs);

            cameraRotationSlerpDataWrite(frameMarkersDataVector);
            std::vector<CameraInterpolatedData> interpolatedRotation = interpolateCameraRotation(imuReadVectorCopy, cameraReadVectorCopy, frameMarkersDataVector);

            cameraRotationSlerpDataWrite(interpolatedRotation);

            testInterpolateCamera(interpolatedRotation, cameraMatrix, distCoeffs);
            */

            WINDOW *win;
            char buff[512];

            win = initscr();
            clearok(win, TRUE);

            size_t imuIndex = 0;
            size_t cameraIndex = 0;

            int oldTimeIMU = 0;
            int oldTimeCamera = 0;

            while (imuIndex < imuReadVector.size() && cameraIndex < cameraReadVector.size())
            {
                if (imuIndex < imuReadVector.size())
                {
                    ImuInputJetson imuDataJetson = imuReadVector.at(imuIndex);

                    wmove(win, 3, 2);
                    snprintf(buff, 511, "Index = %0d", imuDataJetson.index);
                    waddstr(win, buff);

                    wmove(win, 5, 2);
                    snprintf(buff, 511, "Gyro = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuDataJetson.gyroVect.x, imuDataJetson.gyroVect.y, imuDataJetson.gyroVect.z);
                    waddstr(win, buff);

                    wmove(win, 7, 2);
                    snprintf(buff, 511, "Euler = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuDataJetson.eulerVect.x, imuDataJetson.eulerVect.y, imuDataJetson.eulerVect.z);
                    waddstr(win, buff);

                    wmove(win, 9, 2);
                    snprintf(buff, 511, "Quat = {W=%06.2f, X=%06.2f, Y=%06.2f, Z=%06.2f}", imuDataJetson.rotQuat.w, imuDataJetson.rotQuat.x,
                             imuDataJetson.rotQuat.y, imuDataJetson.rotQuat.z);
                    waddstr(win, buff);

                    wmove(win, 11, 3);
                    snprintf(buff, 511, "Acc = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuDataJetson.accVect.x, imuDataJetson.accVect.y, imuDataJetson.accVect.z);
                    waddstr(win, buff);

                    wmove(win, 13, 2);
                    snprintf(buff, 511, "Grav = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuDataJetson.gravVect.x, imuDataJetson.gravVect.y, imuDataJetson.gravVect.z);
                    waddstr(win, buff);

                    if (imuIndex != 0)
                    {
                        wmove(win, 15, 2);
                        snprintf(buff, 511, "Time between captures (IMU): %010d", imuDataJetson.time - oldTimeIMU);
                        waddstr(win, buff);

                        oldTimeIMU = imuDataJetson.time;
                    }
                    else
                    {
                        wmove(win, 15, 2);
                        snprintf(buff, 511, "Time between captures (IMU): %010d", 0);
                        waddstr(win, buff);
                    }
                }

                if (cameraIndex < cameraReadVector.size())
                {
                    CameraInput frame = cameraReadVector.at(cameraIndex);

                    wmove(win, 19, 2);
                    snprintf(buff, 511, "Index = %0d", frame.index);
                    waddstr(win, buff);

                    if (cameraIndex != 0)
                    {
                        wmove(win, 21, 2);
                        snprintf(buff, 511, "Time between captures (IMU): %010d", frame.time - oldTimeCamera);
                        waddstr(win, buff);

                        oldTimeCamera = frame.time;
                    }
                    else
                    {
                        wmove(win, 21, 2);
                        snprintf(buff, 511, "Time between captures (IMU): %010d", 0);
                        waddstr(win, buff);
                    }

                    FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(frame,
                     dictionary, cameraMatrix, distCoeffs);

                    drawAxisOnFrame(frameMarkersData.rvecs, frameMarkersData.tvecs,
                                     frame.frame, cameraMatrix, distCoeffs, "Jetson");
                }

                cv::waitKey(33);

                wrefresh(win);
                wclear(win);

                imuIndex++;
                cameraIndex++;
            }

            endwin();
        }
    }

    return 0;
}

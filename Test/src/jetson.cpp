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
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-2)); // R.
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

void correct(cv::KalmanFilter &KF, cv::Mat_<float> measurement)
{
    cv::Mat_<float> S = KF.measurementMatrix * KF.errorCovPre * KF.measurementMatrix.t() + KF.measurementNoiseCov;
    KF.gain = KF.errorCovPre * KF.measurementMatrix.t() * S.inv();

    int stateSize = KF.statePre.rows;

    KF.statePost = KF.statePre + KF.gain * (measurement - KF.statePre);
    KF.errorCovPost = (cv::Mat::eye(stateSize, stateSize, KF.statePost.type()) - KF.gain * KF.measurementMatrix) * KF.errorCovPre;
}

void correctIMU(cv::KalmanFilter &KF, Eigen::Matrix<double, 19, 1> measurement)
{
    cv::Mat_<float> S = KF.measurementMatrix * KF.errorCovPre * KF.measurementMatrix.t() + KF.measurementNoiseCov;
    KF.gain = KF.errorCovPre * KF.measurementMatrix.t() * S.inv();

    int stateSize = KF.statePre.rows;

    cv::Mat tempMeasurement = convertEigenMatToOpencvMat(measurement);

    KF.statePost = KF.statePre + KF.gain * (tempMeasurement - KF.statePre);
    KF.errorCovPost = (cv::Mat::eye(stateSize, stateSize, KF.statePost.type()) - KF.gain * KF.measurementMatrix) * KF.errorCovPre;
}

void updateTransitionMatrix(cv::KalmanFilter &KF, float deltaT)
{
    KF.transitionMatrix =
        (cv::Mat_<float>(12, 12) << 
        1, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, deltaT, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, deltaT, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, deltaT, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, deltaT, 
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
}

void updateTransitionMatrixIMU(cv::KalmanFilter &KF, Eigen::Matrix<double, 19, 1> measurenment, float deltaT)
{
    float dT2 = deltaT / 2;
    float w1 = KF.statePost.at<float>(0);
    float w2 = KF.statePost.at<float>(1);
    float w3 = KF.statePost.at<float>(2);

    //float w1 = measurenment(0,0);
    //float w2 = measurenment(1,0);
    //float w3 = measurenment(2,0);

    KF.transitionMatrix =
        (cv::Mat_<float>(19, 19) << 
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0,
        0, 0, 0, 1, -dT2*w1, -dT2*w2, -dT2*w3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, dT2*w1, 1, dT2*w3, -dT2*w2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, dT2*w2, -dT2*w3, 1, dT2*w1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, dT2*w3, dT2*w3, -dT2*w1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, deltaT,
        0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
}

void updateMeasurementMatrix(cv::KalmanFilter &KF)
{
    KF.measurementMatrix =
        (cv::Mat_<float>(12, 12) <<
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
}

void initStatePostFirstTime(cv::KalmanFilter &KF, cv::Mat_<float> measurement)
{
    KF.statePost = KF.statePre;
    /*
    KF.statePost.at<float>(0) = measurement.at<float>(0);
    KF.statePost.at<float>(1) = measurement.at<float>(1);
    KF.statePost.at<float>(2) = measurement.at<float>(2);
    KF.statePost.at<float>(3) = quaternion.w;
    KF.statePost.at<float>(4) = quaternion.x;
    KF.statePost.at<float>(5) = quaternion.y;
    KF.statePost.at<float>(6) = quaternion.z;*/
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

    Eigen::Matrix4d identity4x4;
    identity4x4.setIdentity();

    bool firstRun = true;
    float deltaT = -1;
    cv::KalmanFilter KF(19, 19, 0);
    float cumulativeDeltaT = 0;

    Eigen::Matrix<double, 19, 1> measurement;
    measurement.setZero();

    initKalmanFilter(KF, 19);

    std::vector<CameraInput> cameraData = readDataCamera();
    std::vector<ImuInputJetson> imuReadVector = readDataIMUJetson();

    int imuIndex = getImuStartingIdexBaseOnCamera(cameraData, imuReadVector);

    std::vector<Eigen::Vector3d> vectorOfPointsOne;
    std::vector<Eigen::Vector3d> vectorOfPointsTwo;

    FILE *output;
    output = popen("gnuplot", "w");

    for (size_t i = imuIndex; i < imuReadVector.size(); i++)
    {
        ImuInputJetson tempImuData = imuReadVector.at(i);
        
        if (!firstRun)
        {
            /////////////////////////// Prediction ////////////////////////////////////
            
            updateTransitionMatrixIMU(KF, measurement, deltaT);

            std::cout << "KF.transitionMatrix:\n" << KF.transitionMatrix << endl << endl;

            predict(KF);

            std::cout << "KF.statePre:\n" << KF.statePre << endl << endl;

            deltaT = tempImuData.time - imuReadVector.at(i-1).time;
            cumulativeDeltaT += deltaT;

            /////////////////////////// Measurenment ////////////////////////////////////

            Eigen::Vector3d gyro{tempImuData.gyroVect.x, tempImuData.gyroVect.y, tempImuData.gyroVect.z};
            Eigen::Vector3d acc{tempImuData.accVect.x, tempImuData.accVect.y, tempImuData.accVect.z};
            Eigen::Quaterniond imuQuat{
                tempImuData.rotQuat[0],
                tempImuData.rotQuat[1],
                tempImuData.rotQuat[2],
                tempImuData.rotQuat[3]
                };

            imuQuat.normalize();
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

            std::cout << "measurement:\n" << measurement << endl << endl;

            oldAngularVelocity = gyro;
            oldLinealAcc = acc;

            /////////////////////////// Update ////////////////////////////////////

            correctIMU(KF, measurement);

            std::cout << "KF.statePost:\n" << KF.statePost << endl << endl;
        }
        else
        {
            deltaT = tempImuData.time - imuReadVector.at(i-1).time;
            cumulativeDeltaT += deltaT;

            Eigen::Vector3d gyro{tempImuData.gyroVect.x, tempImuData.gyroVect.y, tempImuData.gyroVect.z};
            Eigen::Vector3d acc{tempImuData.accVect.x, tempImuData.accVect.y, tempImuData.accVect.z};
            Eigen::Quaterniond imuQuat{
                tempImuData.rotQuat[0],
                tempImuData.rotQuat[1],
                tempImuData.rotQuat[2],
                tempImuData.rotQuat[3]
                };

            imuQuat.normalize();
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
            measurement(13,0) = 0;
            measurement(14,0) = 0;
            measurement(15,0) = 0;
            measurement(16,0) = acc[0];
            measurement(17,0) = acc[1];
            measurement(18,0) = acc[2];

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

            oldAngularVelocity = gyro;
            oldLinealAcc = acc;

            firstRun = false;
        }
        
        /*wHat = getWHat(gyro);
        std::cout << "wHat:\n"<< wHat << std::endl << std::endl;

        calVelocity = deltaVel - wHat * deltaPos;
        std::cout << "calVelocity:\n"<< calVelocity << std::endl << std::endl;
        
        chi <<
        wHat(0,0), wHat(0,1), wHat(0,2), calVelocity(0),
        wHat(1,0), wHat(1,1), wHat(1,2), calVelocity(1),
        wHat(2,0), wHat(2,1), wHat(2,2), calVelocity(2),
        0, 0, 0, 0;

        std::cout << "chi:\n"<< chi << std::endl << std::endl;
        
        GImu = (identity4x4 + chi * deltaT) * GImuOld;*/

        GImu.block<3,3>(0,0) = imuRot;
        GImu.block<3,1>(0,3) = deltaPos;

        Eigen::Quaterniond tempOriginalQuat = {tempImuData.rotQuat[0], tempImuData.rotQuat[1], tempImuData.rotQuat[2], tempImuData.rotQuat[3]};
        tempOriginalQuat.normalize();

        Eigen::Vector3d rotationVectorOriginal = QuatToRotVectEigen(tempOriginalQuat);

        Eigen::Quaterniond tempQuat = {KF.statePost.at<float>(3), KF.statePost.at<float>(4), KF.statePost.at<float>(5), KF.statePost.at<float>(6)};
        tempQuat.normalize();

        Eigen::Vector3d rotationVector = QuatToRotVectEigen(tempQuat);

        Eigen::Vector3d GyroOriginal{measurement(0,0), measurement(1,0), measurement(2,0)};
        Eigen::Vector3d GyroKF{KF.statePost.at<float>(0), KF.statePost.at<float>(1), KF.statePost.at<float>(2)};

        Eigen::Vector3d VelOriginal{measurement(7,0), measurement(8,0), measurement(9,0)};
        Eigen::Vector3d VelKF{KF.statePost.at<float>(7), KF.statePost.at<float>(8), KF.statePost.at<float>(9)};
        
        Eigen::Vector3d PosOriginal{measurement(10,0), measurement(11,0), measurement(12,0)};
        Eigen::Vector3d PosKF{KF.statePost.at<float>(10), KF.statePost.at<float>(11), KF.statePost.at<float>(12)};

        Eigen::Vector3d AccOriginal{measurement(13,0), measurement(14,0), measurement(15,0)};
        Eigen::Vector3d AccKF{KF.statePost.at<float>(13), KF.statePost.at<float>(14), KF.statePost.at<float>(15)};

        if (vectorOfPointsOne.size() > 50)
        {
            vectorOfPointsOne.erase(vectorOfPointsOne.begin());
            vectorOfPointsTwo.erase(vectorOfPointsTwo.begin());
        }

        vectorOfPointsOne.push_back(rotationVectorOriginal);
        vectorOfPointsTwo.push_back(rotationVector);

        gnuPrintImuPreintegration(output, vectorOfPointsOne, vectorOfPointsTwo);

        //GImuOld = GImu;
    }

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

            updateTransitionMatrix(KF, deltaT);
            updateMeasurementMatrix(KF);

            doMeasurement(measurement, measurementOld, frameMarkersData, deltaT);
            measurementOld = measurement.clone();

            correct(KF, measurement);
            
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

            drawAxisOnFrame(tempRvecs, tempTvecs, tempCameraData.frame, cameraMatrix, distCoeffs, "EKF");

            Eigen::Vector3d graphTvecOriginal{frameMarkersData.rvecs[0][0], frameMarkersData.rvecs[0][1], frameMarkersData.rvecs[0][2]};
            Eigen::Vector3d graphTvecKF{tempRvec[0], tempRvec[1], tempRvec[2]};
            
            vectorOfPointsOne.push_back(graphTvecOriginal);
            vectorOfPointsTwo.push_back(graphTvecKF);

            gnuPrintImuPreintegration(output, vectorOfPointsOne, vectorOfPointsTwo);

            cv::waitKey(33);
        }
        else
        {   
            doMeasurement(measurement, measurementOld, frameMarkersData,  deltaT);
            measurementOld = measurement.clone();
            //initStatePostFirstTime(KF, measurement);
            KF.statePost = measurement.clone();

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
        imuCalibration();
        //printIMUData();
    }
    else if (runKalmanFilterBool)
    {
        //runKalmanFilterCamera();
        runIMUPrediction();
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

            while (imuIndex < imuReadVector.size() || cameraIndex < cameraReadVector.size())
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

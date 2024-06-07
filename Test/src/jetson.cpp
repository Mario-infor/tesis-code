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

void correct(cv::KalmanFilter &KF, cv::Mat_<float> measurement, cv::Mat measurementNoiseCov)
{
    cv::Mat_<float> S = KF.measurementMatrix * KF.errorCovPre * KF.measurementMatrix.t() + measurementNoiseCov;
    KF.gain = KF.errorCovPre * KF.measurementMatrix.t() * S.inv();
    
    int stateSize = KF.statePre.rows;

    KF.statePost = KF.statePre + KF.gain * (measurement - KF.statePre);
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

void updateTransitionMatrixFusion(cv::KalmanFilter &KF, float deltaT, int stateSize, Eigen::Vector3d w)
{
    float dT2 = deltaT / 2;

    float w1 = KF.statePost.at<float>(10);
    float w2 = KF.statePost.at<float>(11);
    float w3 = KF.statePost.at<float>(12);

    /*float w1 = w(0);
    float w2 = w(1);
    float w3 = w(2);*/
    
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

void imuPreintegration(
    const float deltaT,
    const Eigen::Vector3d imuAccWorld,
    Eigen::Vector3d &deltaPos,
    Eigen::Vector3d &deltaVel)
{
    deltaPos += deltaVel * deltaT + 0.5 * imuAccWorld * deltaT * deltaT;
    deltaVel += imuAccWorld * deltaT;
}

void runCameraAndIMUKalmanFilter()
{
    std::vector<Eigen::Vector3d> vectorOfPointsOne;
    std::vector<Eigen::Vector3d> vectorOfPointsTwo;
    std::vector<Eigen::Vector3d> vectorOfMarkers;
    std::vector<TransformBetweenMarkers> vectorOfTransforms;
    std::vector<Eigen::Vector3d> vectorErrorPoints;
    std::vector<Eigen::VectorXd> vectorCamMeasurenments;
    std::vector<Eigen::Quaterniond> vectorImuMeasurenments;
    std::vector<Eigen::VectorXd> vectorStates;

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

    float accelerometer_noise_density = 0.015665911;
    float accelerometer_random_walk = 0.00082064132;

    float gyroscope_noise_density = 0.001576344;
    float gyroscope_random_walk = 1.637381e-05;

    //Eigen::Vector3d accBias{1.35588e-05, 0.000179604, 0.000180296};
    //Eigen::Vector3d gyroBias{0.000333556, 0.000200133, 0.000593729};

    Eigen::Vector3d accBias{0, 0, 0};
    Eigen::Vector3d gyroBias{0, 0, 0};

    Eigen::Vector3d accNoise{0, 0, 0};
    Eigen::Vector3d gyroNoise{0, 0, 0};



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
    /*Gci << 
    -0.99787874, 0.05596833, -0.03324997, 0.09329806,
    0.03309321, -0.00372569, -0.99944533, 0.01431868,
    -0.05606116, -0.99842559, 0.00186561, -0.12008699,
    0.0, 0.0, 0.0, 1.0;*/

    Gci << 
    -0.99787874, 0.05596833, -0.03324997, 0.09329806,
    0.03309321, -0.00372569, -0.99944533, 0.01431868,
    -0.05606116, -0.99842559, 0.00186561, -0.06008699,
    0.0, 0.0, 0.0, 1.0;

    Eigen::Matrix4d Gmc;
    Eigen::Matrix4d oldGmc;
    Eigen::Matrix4d Gcm;
    Eigen::Matrix4d Gwi;
    Eigen::Matrix4d Gmi;
    Eigen::Matrix4d Gci_inv;
    Eigen::Matrix4d Gcm_second;
    Eigen::Matrix4d Gmc_second;
    Eigen::Matrix4d Gmm_secondToBase;
    Eigen::Matrix4d Gmw;
    Eigen::Matrix4d Gmw_inv;


    Eigen::Matrix4d Gti; // Constant that converts IMU measurenment to IMU pos in camera world.
    Eigen::Matrix4d Gni; // Constant used to convert camera Ghi to IMU Ghi.

    Gmc.setIdentity();
    oldGmc.setIdentity();
    Gcm.setIdentity();
    Gwi.setIdentity();
    Gmi.setIdentity();
    Gti.setIdentity();
    Gni.setIdentity();
    Gcm_second.setIdentity();
    Gmc_second.setIdentity();
    Gmm_secondToBase.setIdentity();
    Gmw.setIdentity();
    Gmw_inv.setIdentity();

    Gci_inv = invertG(Gci);

    CameraInput tempCameraData = cameraData.at(indexCamera);
    ImuInputJetson tempImuData = imuReadVector.at(indexImu);

    Eigen::Matrix3d firstImuRot;
    Eigen::Vector3d firstImuGyro;
    Eigen::Vector3d firstImuAcc;

    FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(tempCameraData,
         dictionary, cameraMatrix, distCoeffs);

    Eigen::Vector3d gyro{tempImuData.gyroVect.x, tempImuData.gyroVect.y, tempImuData.gyroVect.z};
    Eigen::Vector3d acc{tempImuData.accVect.x, tempImuData.accVect.y, tempImuData.accVect.z};

    deltaTImu = tempImuData.time - imuReadVector.at(indexImu - 1).time;

    accBias = accBias + accelerometer_random_walk * sqrt(deltaTImu) * acc;
    gyroBias = gyroBias + gyroscope_random_walk * sqrt(deltaTImu) * gyro;
    
    //accNoise = (accelerometer_noise_density / sqrt(deltaTImu)) * acc;
    //gyroNoise = (gyroscope_noise_density / sqrt(deltaTImu)) * gyro;
    
    gyro = gyro - gyroBias;
    acc = acc - accBias;

    Eigen::Quaterniond imuQuat{
        tempImuData.rotQuat[0],
        tempImuData.rotQuat[1],
        tempImuData.rotQuat[2],
        tempImuData.rotQuat[3]
    };

    Eigen::Quaterniond originalQuat = imuQuat;

    fixQuatEigen(imuQuat);

    for (size_t i = 0; i < ites; i++)
    {
        std::cout << "Index: " << i << std::endl;
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
                FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(tempCameraData, dictionary, cameraMatrix, distCoeffs);

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
                    //Eigen::Vector3d markerPos = Gcm.block<3, 1>(0, 3);
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
                    fixQuatEigen(camQuat);

                    w = getAngularVelocityFromTwoQuats(oldCamQuat, camQuat, deltaTCamMeasurement);

                    measurementCam(0) = camT.x();
                    measurementCam(1) = camT.y();
                    measurementCam(2) = camT.z();
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

                // Reset IMU information.

                Eigen::Quaterniond stateQuat(KF.statePost.at<float>(3),  KF.statePost.at<float>(4),  KF.statePost.at<float>(5), KF.statePost.at<float>(6));

                Eigen::Matrix3d stateRot = stateQuat.toRotationMatrix();
                Eigen::Vector3d statePos( KF.statePost.at<float>(0),  KF.statePost.at<float>(1), KF.statePost.at<float>(2));
               
                Eigen::Matrix4d resetGmi;
                resetGmi.setIdentity();
                resetGmi.block<3, 3>(0, 0) = stateRot;
                resetGmi.block<3, 1>(0, 3) = statePos;
                
                Gwi = Gci * resetGmi * Gmw_inv;
                
                Eigen::Matrix4d stateGhi;
                stateGhi << 
                    0,                              -KF.statePost.at<float>(12),     KF.statePost.at<float>(11),      KF.statePost.at<float>(7),
                    KF.statePost.at<float>(12),     0,                              -KF.statePost.at<float>(10),     KF.statePost.at<float>(8),
                    -KF.statePost.at<float>(11),     KF.statePost.at<float>(10),     0,                               KF.statePost.at<float>(9),
                    0,0,0,0;
                
                Eigen::Matrix4d imuGhiMarker = Gci * stateGhi * Gci_inv;
                Eigen::Matrix4d imuGhiWorld = invertG(Gwi) * imuGhiMarker * Gwi;

                deltaPos = Eigen::Vector3d(Gwi.block<3, 1>(0, 3));
                deltaVel = Eigen::Vector3d(imuGhiWorld.block<3, 1>(0, 3));
                //firstImuRot = Gci.block<3, 3>(3, 3) * camRot * Gmi.block<3, 3>(0, 0);

                lastOneWasCamera = true;                                  
            }
            else
            {
                ImuInputJetson tempImuData = imuReadVector.at(indexImu);

                if (lastOneWasCamera)
                    deltaTImu = tempImuData.time - oldDeltaTCam;
                else
                    deltaTImu = tempImuData.time - oldDeltaTImu;

                deltaTImu /= 1000;

                Eigen::Vector3d gyro{tempImuData.gyroVect.x, tempImuData.gyroVect.y, tempImuData.gyroVect.z};
                Eigen::Vector3d acc{tempImuData.accVect.x, tempImuData.accVect.y, tempImuData.accVect.z};
                
                accBias = accBias + accelerometer_random_walk * sqrt(deltaTImu) * acc;
                gyroBias = gyroBias + gyroscope_random_walk * sqrt(deltaTImu) * gyro;

                gyro = gyro - gyroBias;
                acc = acc - accBias;

                imuQuat = Eigen::Quaterniond{
                    tempImuData.rotQuat[0],
                    tempImuData.rotQuat[1],
                    tempImuData.rotQuat[2],
                    tempImuData.rotQuat[3]
                };
                originalQuat = imuQuat;
                fixQuatEigen(imuQuat);

                /////////////////////////// Measurenment ////////////////////////////////////

                Eigen::Matrix3d imuRot = imuQuat.toRotationMatrix();

                //Eigen::Matrix3d Rmi = invertG(Gmc).block<3, 3>(0, 0) * Gci_inv.block<3, 3>(0, 0) * imuRot;

                //Eigen::Matrix3d imuRotFromNewOrigen = imuRot * firstImuRot.transpose();

                Eigen::Vector3d imuAccFromWorld = imuRot.transpose() * acc;

                float deltaTForIntegration = tempImuData.time - imuReadVector.at(indexImu - 1).time;
                deltaTForIntegration /= 1000;
                
                imuPreintegration(deltaTForIntegration, imuAccFromWorld, deltaPos, deltaVel);

                Gwi.block<3, 3>(0, 0) = imuRot;
                Gwi.block<3, 1>(0, 3) = deltaPos;

                Gmi = Gwi * Gmw;

                Eigen::Matrix3d Rmi = Gmi.block<3, 3>(0, 0);
                Eigen::Vector3d Tmi = Gmi.block<3, 1>(0, 3);

                Eigen::Quaterniond imuQuatMarker(Rmi);
                fixQuatEigen(imuQuatMarker);

                Eigen::Matrix3d wHatWorld = getWHat(imuRot.transpose() * gyro);
                Eigen::Matrix4d Ghi_wi;
                Ghi_wi.setZero();
                Ghi_wi.block<3, 3>(0, 0) = wHatWorld;
                Ghi_wi.block<3, 1>(0, 3) = deltaVel;

                Eigen::Matrix4d Ghi_mi = Gmw_inv * Ghi_wi * Gmw;

                measurementImu(0) = Ghi_mi(2, 1);
                measurementImu(1) = Ghi_mi(0, 2);
                measurementImu(2) = Ghi_mi(1, 0);
                measurementImu(3) = imuQuatMarker.w();
                measurementImu(4) = imuQuatMarker.x();
                measurementImu(5) = imuQuatMarker.y();
                measurementImu(6) = imuQuatMarker.z();
                measurementImu(7) = Ghi_mi(0, 3);
                measurementImu(8) = Ghi_mi(1, 3);
                measurementImu(9) = Ghi_mi(2, 3);
                measurementImu(10) = Tmi.x();
                measurementImu(11) = Tmi.y();
                measurementImu(12) = Tmi.z();

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

                Eigen::Vector3d gyroInCameraFrame = multiplyVectorByG(Gci_inv, gyro);                

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

            Gmi = Gci * Gmc;

            Eigen::Matrix3d imuRot = imuQuat.toRotationMatrix();

            //Eigen::Matrix3d Rmi = invertG(Gmc).block<3, 3>(0, 0) * Gci_inv.block<3, 3>(0, 0) * imuRot;
            Eigen::Matrix3d Rmi = Gmi.block<3, 3>(0, 0);
            Eigen::Vector3d Tmi = Gmi.block<3, 1>(0, 3);
            //firstImuRot = Rmi;

            Eigen::Vector3d imuAccFromWorld = imuRot.transpose() * acc;

            imuPreintegration(deltaTImu, imuAccFromWorld, deltaPos, deltaVel);

            Gwi.block<3, 3>(0, 0) = imuRot;
            Gwi.block<3, 1>(0, 3) = deltaPos;

            Gmw = invertG(Gwi) * Gci * Gmc;
            Gmw_inv = invertG(Gmw);

            Eigen::Quaterniond imuQuatMarker(Rmi);
            fixQuatEigen(imuQuatMarker);

            Eigen::Matrix3d wHatWorld = getWHat(imuRot.transpose() * gyro);
            Eigen::Matrix4d Ghi_wi;
            Ghi_wi.setZero();
            Ghi_wi.block<3, 3>(0, 0) = wHatWorld;
            Ghi_wi.block<3, 1>(0, 3) = deltaVel;

            Eigen::Matrix4d Ghi_mi = Gmw_inv * Ghi_wi * Gmw;

            measurementImu(0) = Ghi_mi(2, 1);
            measurementImu(1) = Ghi_mi(0, 2);
            measurementImu(2) = Ghi_mi(1, 0);
            measurementImu(3) = imuQuatMarker.w();
            measurementImu(4) = imuQuatMarker.x();
            measurementImu(5) = imuQuatMarker.y();
            measurementImu(6) = imuQuatMarker.z();
            measurementImu(7) = Ghi_mi(0, 3);
            measurementImu(8) = Ghi_mi(1, 3);
            measurementImu(9) = Ghi_mi(2, 3);
            measurementImu(10) = Tmi.x();
            measurementImu(11) = Tmi.y();
            measurementImu(12) = Tmi.z();

            oldDeltaTImu = tempImuData.time;
            
            firstRun = false;
        }

        Eigen::Vector3d PosOriginal{measurementCam(0), measurementCam(1), measurementCam(2)};
        Eigen::Vector3d PosKF{KF.statePost.at<float>(0), KF.statePost.at<float>(1), KF.statePost.at<float>(2)};
        //Eigen::Vector3d PosKFPre{KF.statePre.at<float>(0), KF.statePre.at<float>(1), KF.statePre.at<float>(2)};

        //Eigen::Quaterniond quatMeasurement{measurementCam(3), measurementCam(4), measurementCam(5), measurementCam(6)};
        //Eigen::Quaterniond quatKF{KF.statePost.at<float>(3), KF.statePost.at<float>(4), KF.statePost.at<float>(5), KF.statePost.at<float>(6)};

        Eigen::Vector3d PosError = PosOriginal - PosKF;
        float errorMagnitude = PosError.norm();
        Eigen::Vector3d printPosError{printErrorX, 0, errorMagnitude};

        /*if(quatMeasurement.norm() != 1)
            fixQuatEigen(quatMeasurement);

        if(quatKF.norm() != 1)
            fixQuatEigen(quatKF);

        float errorMagnitudeQuat = (quatMeasurement*quatKF.inverse()).norm();
        
        Eigen::Vector3d printQuatError{printErrorX, 0, errorMagnitudeQuat};*/

        printErrorX += 0.1;
        
        /*if (vectorOfPointsOne.size() > 50)
        {
            vectorOfPointsOne.erase(vectorOfPointsOne.begin());
            vectorOfPointsTwo.erase(vectorOfPointsTwo.begin());
            vectorOfMarkers.erase(vectorOfMarkers.begin());
        }*/

        //vectorOfPointsOne.push_back(PosOriginal);
        //vectorOfPointsTwo.push_back(PosKF);
        //vectorOfMarkers.push_back(PosKFPre);
        
        if(lastOneWasCamera)
            timeStamps.push_back(oldDeltaTCam);
        else
            timeStamps.push_back(oldDeltaTImu);

        vectorOfPointsOne.push_back(printPosError);
        vectorOfPointsTwo.push_back(Eigen::Vector3d{0,0,0});
        
        //gnuPrintImuPreintegration(output, vectorOfPointsOne, vectorOfPointsTwo, vectorOfMarkers);

        vectorCamMeasurenments.push_back(measurementCam);
        vectorStates.push_back(convertOpencvMatToEigenMat(KF.statePost));

        vectorImuMeasurenments.push_back(originalQuat);

        if (indexCamera == (int)cameraData.size() - 1)
        {
            pointsDataWrite(vectorCamMeasurenments, vectorStates, timeStamps, "cameraVsKalman.csv");
            quatDataWrite(vectorImuMeasurenments, timeStamps, "imuQuats.csv");
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
            IMUDataJetsonWrite(imuDataJetsonBuffer);
        }

        if (preccessData)
        {
            std::vector<ImuInputJetson> imuReadVector = readDataIMUJetson();
            std::vector<CameraInput> cameraReadVector = readDataCamera();

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

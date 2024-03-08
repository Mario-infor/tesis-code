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

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/spline.hpp>
#include <glm/gtc/quaternion.hpp>

// Buffer to store camera structs.
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(RING_BUFFER_LENGTH_CAMERA);

// Buffer to store IMU structs.
RingBuffer<ImuInputJetson> imuDataJetsonBuffer = RingBuffer<ImuInputJetson>(RING_BUFFER_LENGTH_IMU);

void cameraCaptureThread()
{
    int captureWidth = 800 ;
    int captureHeight = 600 ;
    int displayWidth = 800 ;
    int displayHeight = 600 ;
    int frameRate = 30 ;
    int flipMethod = 0 ;

    std::string pipeline = gstreamerPipeline(captureWidth,
            captureHeight,
            displayWidth,
            displayHeight,
            frameRate,
            flipMethod);
    
    cv::VideoCapture cap;
    cap.open(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened())
        std::cerr << "Error openning the camera." << std::endl;
    else
    {
        int index = 0;

        while (index < RING_BUFFER_LENGTH_CAMERA)
        {
            std::cout << "Camera: " << index << std::endl;
            if (doneCalibrating)
            {
                timeCameraStart = std::chrono::steady_clock::now();
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
                    //capture.frame = frame.clone();
                    capture.time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeCameraStart).count();

                    cameraFramesBuffer.Queue(capture);
                    index++;
                }
            }
        }
    }
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
}

void initKalmanFilter(cv::KalmanFilter &KF)
{
    KF.statePre.at<float>(0) = 0;  // x traslation.
    KF.statePre.at<float>(1) = 0;  // y traslation.
    KF.statePre.at<float>(2) = 0;  // z traslation.
    KF.statePre.at<float>(3) = 0;  // w quat rotation.
    KF.statePre.at<float>(4) = 0;  // x quat rotation.
    KF.statePre.at<float>(5) = 0;  // y quat rotation.
    KF.statePre.at<float>(6) = 0;  // z quat rotation.
    KF.statePre.at<float>(7) = 0;  // x traslation velocity.
    KF.statePre.at<float>(8) = 0;  // y traslation velocity.
    KF.statePre.at<float>(9) = 0;  // z traslation velocity.
    KF.statePre.at<float>(10) = 0; // w rotation velocity.
    KF.statePre.at<float>(11) = 0; // x rotation velocity.
    KF.statePre.at<float>(12) = 0; // y rotation velocity.
    KF.statePre.at<float>(13) = 0; // z rotation velocity.

    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));     // Q.
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-2)); // R.
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));           // P'.
    cv::setIdentity(KF.transitionMatrix, cv::Scalar::all(1));       // A.
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
    measurement(3) = frameMarkersData.qvecs[0].val[0]; // quaternion (w)
    measurement(4) = frameMarkersData.qvecs[0].val[1]; // quaternion (x)
    measurement(5) = frameMarkersData.qvecs[0].val[2]; // quaternion (y)
    measurement(6) = frameMarkersData.qvecs[0].val[3]; // quaternion (z)

    measurement(7) = (measurement(0) - measurementOld(0)) / deltaT; // traslation speed (x)
    measurement(8) = (measurement(1) - measurementOld(1)) / deltaT; // traslation speed (y)
    measurement(9) = (measurement(2) - measurementOld(2)) / deltaT; // traslation speed (z)
    measurement(10) = (measurement(3) - measurementOld(3)) / deltaT; // quaternion speed (w)
    measurement(11) = (measurement(4) - measurementOld(4)) / deltaT; // quaternion speed (x)
    measurement(12) = (measurement(5) - measurementOld(5)) / deltaT; // quaternion speed (y)
    measurement(13) = (measurement(6) - measurementOld(6)) / deltaT; // quaternion speed (z)    
}

void correct(cv::KalmanFilter &KF, cv::Mat_<float> measurement)
{
    //cv::Mat_<float> y = measurement - KF.statePre;
    cv::Mat_<float> S = KF.measurementMatrix * KF.errorCovPre * KF.measurementMatrix.t() + KF.measurementNoiseCov;
    KF.gain = KF.errorCovPre * KF.measurementMatrix.t() * S.inv();

    int stateSize = KF.statePre.rows;

    KF.statePost = KF.statePre + KF.gain * (measurement - KF.statePre);
    KF.errorCovPost = (cv::Mat::eye(stateSize, stateSize, KF.statePost.type()) - KF.gain * KF.measurementMatrix) * KF.errorCovPre;
}

void updateTransitionMatrix(cv::KalmanFilter &KF, float deltaT)
{
    KF.transitionMatrix =
        (cv::Mat_<float>(14, 14) << 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT, 0,
         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, deltaT,
         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);
}

void updateMeasurementMatrix(cv::KalmanFilter &KF)
{
    KF.measurementMatrix =
        (cv::Mat_<float>(14, 14) <<
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
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

void runKalmanFilter()
{
    bool firstRun = true;
    float deltaT = 10;
    cv::KalmanFilter KF(14, 14, 0);

    // Create measurement vector (traslation, quaternion, speeds).
    cv::Mat_<float> measurement(14, 1);
    measurement.setTo(cv::Scalar(0));

    // Create old measurement to calculate speeds.
    cv::Mat_<float> measurementOld(14, 1);
    measurementOld.setTo(cv::Scalar(0));

    // Initialize Kalman Filter.
    initKalmanFilter(KF);

    std::vector<CameraInput> cameraData = readDataCamera();

    for (size_t i = 0; i < cameraData.size(); i++)
    {
        CameraInput tempCameraData = cameraData.at(i);

        FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(tempCameraData,
         dictionary, cameraMatrix, distCoeffs);

        cv::Vec3d rotVect = cv::Vec3d(frameMarkersData.rvecs[0][0],
                                    frameMarkersData.rvecs[0][1],
                                    frameMarkersData.rvecs[0][2]);

        glm::quat quaternion = convertOpencvRotVectToQuat(rotVect);

        frameMarkersData.qvecs.push_back(cv::Vec4d(quaternion.w, quaternion.x, quaternion.y, quaternion.z));

        if (!firstRun)
        {
            predict(KF);

            std::cout << "statePre: " << KF.statePre << std::endl << std::endl;

            deltaT = tempCameraData.time;
            updateTransitionMatrix(KF, deltaT);
            updateMeasurementMatrix(KF);

            doMeasurement(measurement, measurementOld, frameMarkersData, deltaT);
            measurementOld = measurement.clone();

            std::cout << "measurement: " << measurement << std::endl << std::endl;

            correct(KF, measurement);
            
            std::cout << "statePost: " << KF.statePost << std::endl << std::endl;
            
            drawAxisOnFrame(frameMarkersData.rvecs, frameMarkersData.tvecs,
                             tempCameraData.frame, cameraMatrix, distCoeffs);
            

            glm::quat tempQuat = glm::quat(KF.statePost.at<float>(3), KF.statePost.at<float>(4),
                                           KF.statePost.at<float>(5), KF.statePost.at<float>(6));

            cv::Vec3d tempRvec = convertQuatToOpencvRotVect(tempQuat);
            cv::Vec3d tempTvec = cv::Vec3d(KF.statePost.at<float>(0), KF.statePost.at<float>(1),
                                             KF.statePost.at<float>(2));

            std::vector<cv::Vec3d> tempRvecs = {tempRvec};
            std::vector<cv::Vec3d> tempTvecs = {tempTvec};

            drawAxisOnFrame(tempRvecs, tempTvecs, tempCameraData.frame, cameraMatrix, distCoeffs);

            cv::waitKey(33);
        }
        else
        {   
            doMeasurement(measurement, measurementOld, frameMarkersData, deltaT);
            measurementOld = measurement.clone();
            //initStatePostFirstTime(KF, measurement);
            KF.statePost = measurement.clone();
            firstRun = false;
        }
    }
}

// Main method that creates threads, writes and read data from files and displays data on console.
int main()
{
    bool generateNewData = false;
    bool preccessData = true;
    bool stopProgram = false;
    bool ifCalibrateIMUOnly = false;
    bool runKalmanFilterBool = true;

    timeIMUStart = std::chrono::steady_clock::now();

    if (ifCalibrateIMUOnly)
    {
        imuCalibration();
        //printIMUData();
    }
    else if (runKalmanFilterBool)
    {
        runKalmanFilter();
    }
    else
    {
        if (generateNewData)
        {
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

            myMutex.lock();
            bool stop = stopProgram;
            myMutex.unlock();

            win = initscr();
            clearok(win, TRUE);

            size_t imuIndex = 0;
            size_t cameraIndex = 0;

            int oldTimeIMU = 0;
            int oldTimeCamera = 0;

            while ((imuIndex < imuReadVector.size() || cameraIndex < cameraReadVector.size()) && !stop)
            {
                if (cv::waitKey(1) == 'q')
                {
                    stopProgram = stop = true;
                }

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
                                     frame.frame, cameraMatrix, distCoeffs);
                }

                cv::waitKey(33);

                wrefresh(win);
                wclear(win);

                imuIndex++;
                cameraIndex++;
                stop = stopProgram;
            }

            endwin();
        }
    }

    return 0;
}

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
#include <readWriteData.h>
#include <structsFile.h>
#include <RingBuffer.h>
#include <BNO055-BBB_driver.h>
#include <jetson.h>

// Buffer to store camera structs.
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(RING_BUFFER_LENGTH_CAMERA);

// Buffer to store IMU structs.
RingBuffer<ImuInputJetson> imuDataJetsonBuffer = RingBuffer<ImuInputJetson>(RING_BUFFER_LENGTH_IMU);


cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 493.02975478, 0, 310.67004724,
                        0, 495.25862058, 166.53292108,
                        0, 0, 1);

cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.12390713, 0.17792574, -0.00934536, -0.01052198, -1.13104202);

void cameraCaptureThread()
{
    int captureWidth = 800 ;
        int captureHeight = 600 ;
        int displayWidth = 800 ;
        int displayHeight = 600 ;
        int frameRate = 30 ;
        int flipMethod = 0 ;

    std::string pipeline = gstreamer_pipeline(captureWidth,
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

// Convert rotation vector to quaternion.
glm::quat convertOpencvRotVectToQuat(cv::Vec3d rotVect)
{
    float vecNorm = cv::norm(rotVect);
    float w = cos(vecNorm / 2);
    cv::Vec3d xyz = sin(vecNorm / 2) * rotVect / vecNorm;

    glm::quat quaternion = glm::quat(w, xyz[0], xyz[1], xyz[2]);

    return quaternion;
}

// Convert quaternion to rotation vector.
cv::Vec3d convertQuatToOpencvRotVect(glm::quat quaternion)
{
    cv::Vec3d rotVect;

    float w = quaternion.w;
    float x = quaternion.x;
    float y = quaternion.y;
    float z = quaternion.z;

    float vecNorm = acos(w * 2);

    rotVect[0] = x * vecNorm / sin(vecNorm / 2);
    rotVect[1] = y * vecNorm / sin(vecNorm / 2);
    rotVect[2] = z * vecNorm / sin(vecNorm / 2);

    return rotVect;
}

// Create spline points.
std::vector<glm::vec3> createSplinePoint(std::vector<ImuInputJetson> imuReadVector)
{
    std::vector<glm::vec3> points;

    for (size_t i = 0; i < imuReadVector.size() - 4; i++)
    {
        // Create a for loop for t values from 0 to 1 with a step of 0.1.
        for (float t = 0; t < 1; t += 0.1)
        {
            glm::vec3 tempPoint = glm::catmullRom(imuReadVector[i].accVect, imuReadVector[i + 1].accVect,
                                                  imuReadVector[i + 2].accVect, imuReadVector[i + 3].accVect, t);

            points.push_back(tempPoint);
        }
    }

    return points;
}

// Create slerp points for quaternions (tests at home).
std::vector<glm::quat> createSlerpPoint(std::vector<ImuInputJetson> imuReadVector)
{
    std::vector<glm::quat> points;

    for (size_t i = 0; i < imuReadVector.size() - 1; i++)
    {
        // Create a for loop for t values from 0 to 1 with a step of 0.1.
        for (float t = 0; t < 1; t += 0.1)
        {
            glm::quat tempPoint = glm::slerp(imuReadVector.at(i).rotQuat, imuReadVector.at(i + 1).rotQuat, t);
            points.push_back(tempPoint);
        }
    }

    return points;
}

// Tests Slerp and Spline methods.
void testSlerpAndSpline(std::vector<ImuInputJetson> imuReadVector, std::vector<CameraInput> cameraReadVector)
{
    std::vector<glm::vec3> splinePoints = createSplinePoint(imuReadVector);
    std::vector<glm::quat> slerpPoints = createSlerpPoint(imuReadVector);

    std::cout << "Spline points: " << std::endl;
    for (size_t i = 0; i < splinePoints.size(); i++)
    {
        std::cout << "X: " << splinePoints[i].x << " Y: " << splinePoints[i].y << " Z: " << splinePoints[i].z << std::endl;
    }

    std::cout << "Slerp points: " << std::endl;
    for (size_t i = 0; i < slerpPoints.size(); i++)
    {
        std::cout << "W: " << slerpPoints[i].w << " X: " << slerpPoints[i].x << " Y: " << slerpPoints[i].y << " Z: " << slerpPoints[i].z << std::endl;
    }
}

// Get rotation and translation from frame.
FrameMarkersData getRotationTraslationFromFrame(CameraInput frame)
{
    FrameMarkersData frameMarkersData;

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;

    cv::aruco::detectMarkers(frame.frame, dictionary, markerCorners, markerIds);

    if (markerIds.size() > 0)
    {
        cv::aruco::drawDetectedMarkers(frame.frame, markerCorners, markerIds);

        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
        frameMarkersData.rvecs = rvecs;
        frameMarkersData.tvecs = tvecs;
    }

    return frameMarkersData;
}

// Get rotation and translation from all frames.
std::vector<FrameMarkersData> getRotationTraslationFromAllFrames(std::vector<CameraInput> cameraReadVector)
{
    std::vector<FrameMarkersData> frameMarkersDataVector;

    for (size_t i = 0; i < cameraReadVector.size(); i++)
    {
        FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(cameraReadVector[i]);
        frameMarkersDataVector.push_back(frameMarkersData);
    }

    return frameMarkersDataVector;
}

// Interpolate camera rotation to fit IMU data.
std::vector<CameraInterpolatedData> interpolateCameraRotation(const std::vector<ImuInputJetson> imuReadVectorCopy,
                                                              const std::vector<CameraInput> cameraReadVectorCopy,
                                                              std::vector<FrameMarkersData> frameMarkersDataVector)
{
    std::vector<CameraInterpolatedData> interpolatedPoints;

    int indexCamera = 0;
    int indexIMU = 0;

    while (indexCamera != (int)(cameraReadVectorCopy.size() - 1))
    {
        CameraInput tempCameraInput = cameraReadVectorCopy[indexCamera];
        FrameMarkersData tempFrameMarkersData = frameMarkersDataVector[indexCamera];
        CameraInterpolatedData tempCameraInterpolatedData;

        tempCameraInterpolatedData.originalOrNot = 0;
        tempCameraInterpolatedData.frame = tempCameraInput;
        tempCameraInterpolatedData.frameMarkersData = tempFrameMarkersData;

        interpolatedPoints.push_back(tempCameraInterpolatedData);

        for (size_t i = indexIMU; i < imuReadVectorCopy.size(); i++)
        {
            if (imuReadVectorCopy[i].time > cameraReadVectorCopy[indexCamera].time && imuReadVectorCopy[i].time < cameraReadVectorCopy[indexCamera + 1].time)
            {
                tempCameraInput.index = -1;
                tempCameraInput.time = imuReadVectorCopy[i].time;
                tempCameraInput.frame = cameraReadVectorCopy[indexCamera].frame;

                for (size_t j = 0; j < frameMarkersDataVector[indexCamera].rvecs.size(); j++)
                {
                    cv::Vec3d rotVect0 = frameMarkersDataVector[indexCamera].rvecs[j];
                    cv::Vec3d rotVect1 = frameMarkersDataVector[indexCamera + 1].rvecs[j];

                    glm::quat quaternion0 = convertOpencvRotVectToQuat(rotVect0);
                    glm::quat quaternion1 = convertOpencvRotVectToQuat(rotVect1);

                    float relativePos = (float)(imuReadVectorCopy[i].time - cameraReadVectorCopy[indexCamera].time) /
                                        (float)(cameraReadVectorCopy[indexCamera + 1].time - cameraReadVectorCopy[indexCamera].time);

                    glm::quat interpolatedPoint = glm::slerp(quaternion0, quaternion1, relativePos);
                    cv::Vec3d rotacionVec3 = convertQuatToOpencvRotVect(interpolatedPoint);

                    tempFrameMarkersData.rvecs.push_back(rotacionVec3);
                    tempFrameMarkersData.qvecs.push_back(cv::Vec4d(interpolatedPoint.w, interpolatedPoint.x, interpolatedPoint.y, interpolatedPoint.z));

                    tempFrameMarkersData.tvecs.push_back(frameMarkersDataVector[indexCamera].tvecs[j]);
                    tempFrameMarkersData.markerIds.push_back(frameMarkersDataVector[indexCamera].markerIds[j]);
                }

                tempCameraInterpolatedData.originalOrNot = 1;
                tempCameraInterpolatedData.frame = tempCameraInput;
                tempCameraInterpolatedData.frameMarkersData = tempFrameMarkersData;

                interpolatedPoints.push_back(tempCameraInterpolatedData);

                indexIMU = i + 1;
            }
            else if (imuReadVectorCopy[i].time > cameraReadVectorCopy[indexCamera + 1].time)
                break;
        }
        indexCamera++;
    }

    return interpolatedPoints;
}

// Test interpolate camera rotation to fit IMU data.
void testInterpolateCamera(std::vector<CameraInterpolatedData> interpolatedPoints)
{
    for (int i = 0; i < (int)interpolatedPoints.size(); i++)
    {
        for (size_t j = 0; j < interpolatedPoints[i].frameMarkersData.rvecs.size(); j++)
        {
            if (!std::isnan(interpolatedPoints[i].frameMarkersData.rvecs[j][0]))
            {
                std::cout << "Rvec: " << interpolatedPoints[i].frameMarkersData.rvecs[j] << std::endl;

                cv::aruco::drawAxis(interpolatedPoints[i].frame.frame, cameraMatrix, distCoeffs, interpolatedPoints[i].frameMarkersData.rvecs[j],
                                    interpolatedPoints[i].frameMarkersData.tvecs[j], 0.1);
            }
        }

        cv::imshow("Test", interpolatedPoints[i].frame.frame);

        cv::waitKey(1);
    }
    cv::destroyAllWindows();
}

// Create a hard copy of camera vector.
std::vector<CameraInput> hardCopyCameraVector(std::vector<CameraInput> cameraReadVector)
{
    std::vector<CameraInput> cameraReadVectorCopy;

    std::vector<CameraInput>::iterator it = cameraReadVector.begin();
    CameraInput tempCamera;

    for (; it != cameraReadVector.end(); it++)
    {
        tempCamera = *it;
        cameraReadVectorCopy.push_back(tempCamera);
    }

    return cameraReadVectorCopy;
}

void printIMUData()
{
    char filename[] = IMU_ADDRESS;
    BNO055 sensors;
    sensors.openDevice(filename);

    WINDOW *win;
    char buff[512];

    win = initscr();
    clearok(win, TRUE);

    float maxX = 0.0;
    float maxY = 0.0;
    float maxZ = 0.0;

    while (true)
    {
        sensors.readAll();
        wmove(win, 5, 2);


        float tempMaxX = sensors.accelVect.vi[0] * sensors.Scale;
        float tempMaxY = sensors.accelVect.vi[1] * sensors.Scale;
        float tempMaxZ = sensors.accelVect.vi[2] * sensors.Scale;

        if (tempMaxX > maxX)
            maxX = tempMaxX;
        
        if (tempMaxY > maxY)
            maxY = tempMaxY;

        if (tempMaxZ > maxZ)
            maxZ = tempMaxZ;
        

        //snprintf(buff, 511, "Acc = {X=%f, Y=%f, Z=%f}", maxX, maxY, maxZ);

        snprintf(buff, 79, "EULER=[%07.5lf, %07.5lf, %07.3lf]",
            sensors.eOrientation.vi[0] * sensors.Scale * MATH_RAD_TO_DEGREE,
            sensors.eOrientation.vi[1] * sensors.Scale * MATH_RAD_TO_DEGREE,
            sensors.eOrientation.vi[2] * sensors.Scale * MATH_RAD_TO_DEGREE);
      
        waddstr(win, buff);

        //sleep(1);

        wrefresh(win);
        wclear(win);
    }
    endwin();
} 

// Main method that creates threads, writes and read data from files and displays data on console.
int main()
{
    
    bool ifCalibrateIMUOnly = true;
    timeIMUStart = std::chrono::steady_clock::now();

    if (ifCalibrateIMUOnly)
    {
        //imuCalibration();
        printIMUData();
    }
    else
    {
        if (generateNewData)
        {
            std::thread cameraCapture(cameraCaptureThread);
            std::thread imu(imuThreadJetson);

            cameraCapture.join();
            imu.join();

            cameraDataWrite();
            IMUDataWriteTestAxis();
            //IMUDataJetsonWrite();
        }

        if (preccessData)
        {
            std::vector<ImuInputJetson> imuReadVector = readDataIMUJetson();
            std::vector<CameraInput> cameraReadVector = readDataCamera();

            std::vector<glm::vec3> splinePoints = createSplinePoint(imuReadVector);
            std::vector<glm::quat> slerpPoints = createSlerpPoint(imuReadVector);

            std::vector<ImuInputJetson> imuReadVectorCopy = imuReadVector;
            std::vector<CameraInput> cameraReadVectorCopy = hardCopyCameraVector(cameraReadVector);

            std::vector<FrameMarkersData> frameMarkersDataVector = getRotationTraslationFromAllFrames(cameraReadVectorCopy);

            cameraRotationSlerpDataWrite(frameMarkersDataVector);

            std::vector<CameraInterpolatedData> interpolatedRotation = interpolateCameraRotation(imuReadVectorCopy, cameraReadVectorCopy, frameMarkersDataVector);

            cameraRotationSlerpDataWrite(interpolatedRotation);

            // testInterpolateCamera(interpolatedRotation);

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

                    FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(frame);

                    for (int i = 0; i < (int)frameMarkersData.rvecs.size(); i++)
                    {
                        cv::aruco::drawAxis(frame.frame, cameraMatrix, distCoeffs, frameMarkersData.rvecs[i], frameMarkersData.tvecs[i], 0.1);
                    }

                    cv::imshow("draw axis", frame.frame);
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

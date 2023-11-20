#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
//#include <BNO055-BBB_driver.h>
#include <chrono>
#include <curses.h>
#include <vector>
#include <thread>
#include <mutex>
#include <fstream>
#include "RingBuffer.h"
#include <glm/glm.hpp>
#include <glm/gtx/spline.hpp>
#include <glm/gtc/quaternion.hpp>

// Amount of IMU data and frames to read from devices.
#define RINGBUFFERLENGTHCAMERA 1875
#define RINGBUFFERLENGTHIMU 3750

// Struct to store information about each frame saved.
struct CameraInput
{
    int index;
    int time;
    cv::Mat frame;
};

// Struct to store information about each IMU data saved (Jetson Board).
struct ImuInputJetson
{
    int index;
    int time;
    glm::vec3 gyroVect;
    glm::vec3 eulerVect;
    glm::quat rotQuat;
    glm::vec3 accVect;
    glm::vec3 gravVect;
};

// Buffer to store camera structs.
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(RINGBUFFERLENGTHCAMERA);

// Buffer to store IMU structs.

RingBuffer<ImuInputJetson> imuDataJetsonBuffer = RingBuffer<ImuInputJetson>(RINGBUFFERLENGTHIMU);

// Global variables that need to be accessed from different threads or methods.
std::mutex myMutex;
std::chrono::time_point<std::chrono::steady_clock> timeCameraStart;
std::chrono::time_point<std::chrono::steady_clock> timeIMUStart;
std::string dirCameraFolder = "./Data/Camera/";
std::string dirIMUFolder = "./Data/IMU/";
bool stopProgram = false;
bool doneCalibrating = false;
bool generateNewData = true;
bool preccessData = false;

// Pipeline for camera on JEtson Board.

std::string get_tegra_pipeline(int width, int height, int fps)
{
    return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)I420, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// Thread in charge of readng data from camera and store it on camera buffer.
void cameraCaptureThread()
{
    int WIDTH = 640;
    int HEIGHT = 360;
    int FPS = 60;
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);
    cv::VideoCapture cap;
    cap.open(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened())
        std::cerr << "Error al abrir la cámara." << std::endl;
    else
    {
        int index = 0;
        timeCameraStart = std::chrono::steady_clock::now();

        while (index < RINGBUFFERLENGTHCAMERA)
        {
            std::cout << "Camera: " << index << std::endl;
            if (doneCalibrating)
            {
                cv::Mat frame, grayscale;
                cap.read(frame);

                if (frame.empty())
                {
                    std::cerr << "No se pudo capturar el frame." << std::endl;
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

// Thead in charge of reading data from the IMU.
void imuThreadJetson()
{
    int cont = 0;
    char filename[] = "/dev/i2c-1";
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

    while (index < RINGBUFFERLENGTHIMU)
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

// Write IMU data to files.
void IMUDataJetsonWrite()
{
    std::ofstream IMUTimeFile(dirIMUFolder + "IMUTime", std::ios::out);
    std::ofstream IMUDataFile(dirIMUFolder + "IMUData", std::ios::out);

    if (IMUTimeFile.is_open() && IMUDataFile.is_open())
    {
        while (!imuDataJetsonBuffer.QueueIsEmpty())
        {
            ImuInputJetson tempIMU;
            imuDataJetsonBuffer.Dequeue(tempIMU);

            IMUTimeFile << tempIMU.time << std::endl;

            IMUDataFile << tempIMU.index << std::endl;
            IMUDataFile << tempIMU.gyroVect.x << std::endl;
            IMUDataFile << tempIMU.gyroVect.y << std::endl;
            IMUDataFile << tempIMU.gyroVect.z << std::endl;
            IMUDataFile << tempIMU.eulerVect.x << std::endl;
            IMUDataFile << tempIMU.eulerVect.y << std::endl;
            IMUDataFile << tempIMU.eulerVect.z << std::endl;
            IMUDataFile << tempIMU.rotQuat.w << std::endl;
            IMUDataFile << tempIMU.rotQuat.x << std::endl;
            IMUDataFile << tempIMU.rotQuat.y << std::endl;
            IMUDataFile << tempIMU.rotQuat.z << std::endl;
            IMUDataFile << tempIMU.accVect.x << std::endl;
            IMUDataFile << tempIMU.accVect.y << std::endl;
            IMUDataFile << tempIMU.accVect.z << std::endl;
            IMUDataFile << tempIMU.gravVect.x << std::endl;
            IMUDataFile << tempIMU.gravVect.y << std::endl;
            IMUDataFile << tempIMU.gravVect.z << std::endl;
        }
    }
}

// Read IMU data from files.
std::vector<ImuInputJetson> readDataIMUJetson()
{
    std::vector<ImuInputJetson> IMUData;
    std::ifstream fileTime(dirIMUFolder + "IMUTime");
    std::ifstream fileData(dirIMUFolder + "IMUData");

    if (!fileTime || !fileData)
        std::cerr << "Files not found." << std::endl;
    else
    {
        int value;
        while (fileTime >> value)
        {
            ImuInputJetson tempIMUInput;

            tempIMUInput.time = value;

            fileData >> tempIMUInput.index;
            fileData >> tempIMUInput.gyroVect.x;
            fileData >> tempIMUInput.gyroVect.y;
            fileData >> tempIMUInput.gyroVect.z;
            fileData >> tempIMUInput.eulerVect.x;
            fileData >> tempIMUInput.eulerVect.y;
            fileData >> tempIMUInput.eulerVect.z;
            fileData >> tempIMUInput.rotQuat.w;
            fileData >> tempIMUInput.rotQuat.x;
            fileData >> tempIMUInput.rotQuat.y;
            fileData >> tempIMUInput.rotQuat.z;
            fileData >> tempIMUInput.accVect.x;
            fileData >> tempIMUInput.accVect.y;
            fileData >> tempIMUInput.accVect.z;
            fileData >> tempIMUInput.gravVect.x;
            fileData >> tempIMUInput.gravVect.y;
            fileData >> tempIMUInput.gravVect.z;

            IMUData.push_back(tempIMUInput);
        }
    }

    return IMUData;
}

// Write camera time data to file and store all frams as .png files.
void cameraDataWrite()
{
    std::ofstream cameraTimeFile(dirCameraFolder + "cameraTime", std::ios::out);

    if (cameraTimeFile.is_open())
    {
        while (!cameraFramesBuffer.QueueIsEmpty())
        {
            char buff[256];

            CameraInput tempFrame;
            cameraFramesBuffer.Dequeue(tempFrame);
            snprintf(buff, 255, "frame_%06d.png", tempFrame.index);
            std::string imageName(buff);
            cv::imwrite(dirCameraFolder + imageName, tempFrame.frame);

            cameraTimeFile << tempFrame.time << std::endl;
        }
    }
}

// Read camera data and frames from files.
std::vector<CameraInput> readDataCamera()
{
    std::vector<CameraInput> cameraData;
    std::ifstream fileTime(dirCameraFolder + "cameraTime");

    int index = 0;
    std::string imageName = "";
    cv::Mat image;

    char buff[256];

    if (!fileTime)
        std::cerr << "File not found." << std::endl;
    else
    {
        int value;
        while (fileTime >> value)
        {
            CameraInput tempCameraInput;
            tempCameraInput.time = value;
            tempCameraInput.index = index;

            snprintf(buff, 255, "frame_%06d.png", tempCameraInput.index);
            // imageName = "frame_" + std::to_string(index) + ".png";
            std::string imageName(buff);
            image = cv::imread(dirCameraFolder + imageName, cv::IMREAD_GRAYSCALE);
            image.copyTo(tempCameraInput.frame);

            cameraData.push_back(tempCameraInput);
            index++;
        }
    }

    return cameraData;
}

// Create spline points (tests at home).
/*std::vector<glm::vec3> createSplinePoint(std::vector<ImuInputJetson> imuReadVector)
{
    std::vector<glm::vec3> points;

    for (size_t i = 0; i < imuReadVector.size() - 4; i++)
    {
        std::vector<glm::vec3> controlPoints = {
            glm::vec3(imuReadVector[i].accX, imuReadVector[i].accY, imuReadVector[i].accZ),
            glm::vec3(imuReadVector[i + 1].accX, imuReadVector[i + 1].accY, imuReadVector[i + 1].accZ),
            glm::vec3(imuReadVector[i + 2].accX, imuReadVector[i + 2].accY, imuReadVector[i + 2].accZ),
            glm::vec3(imuReadVector[i + 3].accX, imuReadVector[i + 3].accY, imuReadVector[i + 3].accZ),
        };

        // Create a for loop for t values from 0 to 1 with a step of 0.1.
        for (float t = 0; t < 1; t += 0.1)
        {
            glm::vec3 tempPoint = glm::catmullRom(controlPoints[0], controlPoints[1], controlPoints[2], controlPoints[3], t);
            points.push_back(tempPoint);
        }
    }

    return points;
}*/

// Create slerp points for quaternions (tests at home).
/*std::vector<glm::quat> createSlerpPoint(std::vector<ImuInputJetson> imuReadVector)
{
    std::vector<glm::quat> points;

    for (size_t i = 0; i < imuReadVector.size() - 1; i++)
    {
        std::vector<glm::quat> controlPoints = {
            glm::quat(imuReadVector[i].quatW, imuReadVector[i].quatX, imuReadVector[i].quatY, imuReadVector[i].quatZ),
            glm::quat(imuReadVector[i + 1].quatW, imuReadVector[i + 1].quatX, imuReadVector[i + 1].quatY, imuReadVector[i + 1].quatZ),
        };

        // Create a for loop for t values from 0 to 1 with a step of 0.1.
        for (float t = 0; t < 1; t += 0.1)
        {
            glm::quat tempPoint = glm::slerp(controlPoints[0], controlPoints[1], t);
            points.push_back(tempPoint);
        }
    }

    return points;
}*/

// Main method that creates threads, writes and read data from files and displays data on console.
int main()
{
    timeIMUStart = std::chrono::steady_clock::now();

    if (generateNewData)
    {
        std::thread cameraCapture(cameraCaptureThread);
        std::thread imu(imuThreadJetson);

        cameraCapture.join();
        imu.join();

        cameraDataWrite();
        IMUDataJetsonWrite();
    }

    if (preccessData)
    {
        std::vector<ImuInputJetson> imuReadVector = readDataIMUJetson();
        std::vector<CameraInput> cameraReadVector = readDataCamera();

        /*std::vector<glm::vec3> splinePoints = createSplinePoint(imuReadVector);
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
        }*/

        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

        cv::Mat cameraMatrix, distCoeffs;

        cameraMatrix = (cv::Mat_<double>(3, 3) << 1.4149463861018060e+03, 0.0, 9.6976370017096372e+02,
                        0.0, 1.4149463861018060e+03, 5.3821002771506880e+02,
                        0.0, 0.0, 1.0);

        distCoeffs = (cv::Mat_<double>(1, 5) << 1.8580734579482813e-01, -5.5388292695096419e-01, 1.9639104707396063e-03,
                      4.5272274621161552e-03, 5.3671862979121965e-01);

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
                stopProgram = true;
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

                std::vector<int> markerIds;
                std::vector<std::vector<cv::Point2f>> markerCorners;

                cv::aruco::detectMarkers(frame.frame, dictionary, markerCorners, markerIds);

                if (markerIds.size() > 0)
                {
                    cv::aruco::drawDetectedMarkers(frame.frame, markerCorners, markerIds);

                    std::vector<cv::Vec3d> rvecs, tvecs;
                    cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

                    for (int i = 0; i < (int)rvecs.size(); i++)
                    {
                        cv::aruco::drawAxis(frame.frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
                    }
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

    return 0;
}

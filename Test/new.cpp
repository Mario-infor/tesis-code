#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <boost/asio.hpp>
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
#define RINGBUFFERLENGTHCAMERA 50
#define RINGBUFFERLENGTHIMU 100

// Struct to store information about each frame saved.
struct CameraInput
{
    int index;
    int time;
    cv::Mat frame;

    CameraInput &operator=(const CameraInput &other)
    {
        if (this != &other)
        {
            index = other.index;
            time = other.time;
            frame = other.frame.clone();
        }
        return *this;
    }
};

// Struct to store information about each IMU data saved (tests at home).
struct ImuInput
{
    int index;
    int time;
    glm::vec3 acc;
    glm::quat rotQuat;
};

// Struct to store a list of rvects and tvects.
struct FrameMarkersData
{
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;
};

struct CameraInterpolatedData
{
    CameraInput frame;
    FrameMarkersData frameMarkersData;
};

// Buffer to store camera structs.
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(RINGBUFFERLENGTHCAMERA);

// Buffer to store IMU structs.
RingBuffer<ImuInput> imuDataBuffer = RingBuffer<ImuInput>(RINGBUFFERLENGTHIMU);

// Global variables that need to be accessed from different threads or methods.
std::mutex myMutex;
std::chrono::time_point<std::chrono::steady_clock> timeCameraStart;
std::chrono::time_point<std::chrono::steady_clock> timeIMUStart;
std::string dirCameraFolder = "./Data/Camera/";
std::string dirIMUFolder = "./Data/IMU/";
std::string dirRotationsFolder = "./Data/Rotations/";
bool stopProgram = false;
bool doneCalibrating = false;
bool generateNewData = false;

cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 493.02975478, 0, 310.67004724,
                        0, 495.25862058, 166.53292108,
                        0, 0, 1);

cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.12390713, 0.17792574, -0.00934536, -0.01052198, -1.13104202);

// Thread in charge of readng data from camera and store it on camera buffer.
void cameraCaptureThread()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
        std::cerr << "Error opening the camera." << std::endl;
    else
    {
        int index = 0;

        while (index < RINGBUFFERLENGTHCAMERA)
        {
            std::cout << "Camera: " << index << std::endl;
            if (doneCalibrating)
            {
                timeCameraStart = std::chrono::steady_clock::now();
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

// Convert string recived from IMU to float data (Tests at home).
void parseImuData(std::string data, std::vector<float> &parsedData)
{
    std::stringstream ss(data);
    std::vector<std::string> splitData;
    std::string temp;

    while (std::getline(ss, temp, ','))
    {
        splitData.push_back(temp);
    }

    for (const std::string &item : splitData)
    {
        try
        {
            float number = std::stof(item);
            parsedData.push_back(number);
        }
        catch (const std::invalid_argument &e)
        {
            std::cerr << "Error: Could not convert string to float. " << e.what() << std::endl;
        }
        catch (const std::out_of_range &e)
        {
            std::cerr << "Error: Value is out of float range. " << e.what() << std::endl;
        }
    }
}

// Thread in charge of reading data from the IMU (tests at home).
void imuThread()
{
    boost::asio::io_service io;
    boost::asio::serial_port serial(io);

    try
    {
        serial.open("/dev/ttyACM0");
        serial.set_option(boost::asio::serial_port_base::baud_rate(9600));
        boost::asio::streambuf buffer;

        int index = 0;

        while (index < RINGBUFFERLENGTHIMU)
        {
            std::cout << "IMU: " << index << std::endl;
            boost::system::error_code ec;
            boost::asio::read_until(serial, buffer, '\n', ec);

            ImuInput imuInput;
            imuInput.time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeIMUStart).count();

            if (ec)
            {
                std::cout << ec.what();
            }
            else
            {
                std::string receivedData;
                std::istream is(&buffer);
                std::getline(is, receivedData);

                std::vector<float> parsedData;
                parseImuData(receivedData, parsedData);

                if (parsedData.size() == 7)
                {
                    imuInput.index = index;
                    imuInput.acc = glm::vec3(parsedData[0], parsedData[1], parsedData[2]);
                    imuInput.rotQuat = glm::quat(parsedData[3], parsedData[4], parsedData[5], parsedData[6]);

                    doneCalibrating = true;
                }
                else
                {
                    imuInput.index = index;
                    imuInput.acc = glm::vec3(0, 0, 0);
                    imuInput.rotQuat = glm::quat(1, 0, 0, 0);
                }
                index++;
                imuDataBuffer.Queue(imuInput);
            }
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    serial.close();
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

// Write data from IMU to files (tests at home).
void IMUDataWrite()
{
    std::ofstream IMUTimeFile(dirIMUFolder + "IMUTime", std::ios::out);
    std::ofstream IMUDataFile(dirIMUFolder + "IMUData", std::ios::out);

    if (IMUTimeFile.is_open() && IMUDataFile.is_open())
    {
        while (!imuDataBuffer.QueueIsEmpty())
        {
            ImuInput tempIMU;
            imuDataBuffer.Dequeue(tempIMU);

            IMUTimeFile << tempIMU.time << std::endl;
            IMUDataFile << tempIMU.index << std::endl;
            IMUDataFile << tempIMU.acc.x << std::endl;
            IMUDataFile << tempIMU.acc.y << std::endl;
            IMUDataFile << tempIMU.acc.z << std::endl;
            IMUDataFile << tempIMU.rotQuat.w << std::endl;
            IMUDataFile << tempIMU.rotQuat.x << std::endl;
            IMUDataFile << tempIMU.rotQuat.y << std::endl;
            IMUDataFile << tempIMU.rotQuat.z << std::endl;
        }
    }
}

// Read IMU data from files.
std::vector<ImuInput> readDataIMU()
{
    std::vector<ImuInput> IMUData;
    std::ifstream fileTime(dirIMUFolder + "IMUTime");
    std::ifstream fileData(dirIMUFolder + "IMUData");

    if (!fileTime || !fileData)
        std::cerr << "Files not found." << std::endl;
    else
    {
        int value;
        while (fileTime >> value)
        {
            ImuInput tempIMUInput;

            tempIMUInput.time = value;
            fileData >> tempIMUInput.index;
            fileData >> tempIMUInput.acc.x;
            fileData >> tempIMUInput.acc.y;
            fileData >> tempIMUInput.acc.z;
            fileData >> tempIMUInput.rotQuat.w;
            fileData >> tempIMUInput.rotQuat.x;
            fileData >> tempIMUInput.rotQuat.y;
            fileData >> tempIMUInput.rotQuat.z;

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
std::vector<glm::vec3> createSplinePoint(std::vector<ImuInput> imuReadVector)
{
    std::vector<glm::vec3> points;

    for (size_t i = 0; i < imuReadVector.size() - 4; i++)
    {
        // Create a for loop for t values from 0 to 1 with a step of 0.1.
        for (float t = 0; t < 1; t += 0.1)
        {
            glm::vec3 tempPoint = glm::catmullRom(imuReadVector[i].acc, imuReadVector[i + 1].acc,
                                                  imuReadVector[i + 2].acc, imuReadVector[i + 3].acc, t);

            points.push_back(tempPoint);
        }
    }

    return points;
}

// Create slerp points for quaternions (tests at home).
std::vector<glm::quat> createSlerpPoint(std::vector<ImuInput> imuReadVector)
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

void testSlerpAndSpline(std::vector<ImuInput> imuReadVector, std::vector<CameraInput> cameraReadVector)
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
std::vector<CameraInterpolatedData> interpolateCameraRotation(const std::vector<ImuInput> imuReadVectorCopy, const std::vector<CameraInput> cameraReadVectorCopy)
{
    std::vector<CameraInterpolatedData> interpolatedPoints;
    std::vector<FrameMarkersData> frameMarkersDataVector = getRotationTraslationFromAllFrames(cameraReadVectorCopy);

    int indexCamera = 0;
    int indexIMU = 0;

    while (indexCamera != (int)(cameraReadVectorCopy.size() - 1))
    {
        for (size_t i = indexIMU; i < imuReadVectorCopy.size(); i++)
        {
            if (imuReadVectorCopy[i].time > cameraReadVectorCopy[indexCamera].time && imuReadVectorCopy[i].time < cameraReadVectorCopy[indexCamera + 1].time)
            {
                CameraInterpolatedData tempCameraInterpolatedData;
                CameraInput tempCameraInput;
                FrameMarkersData tempFrameMarkersData;

                tempCameraInput.index = -1;
                tempCameraInput.time = imuReadVectorCopy[i].time;
                tempCameraInput.frame = cameraReadVectorCopy[indexCamera].frame;

                for (size_t j = 0; j < frameMarkersDataVector[indexCamera].rvecs.size(); j++)
                {
                    cv::Vec3d rotVect0 = frameMarkersDataVector[indexCamera].rvecs[j];
                    cv::Vec3d rotVect1 = frameMarkersDataVector[indexCamera + 1].rvecs[j];

                    /* glm::vec3 eulerAngles0(rotVect0[0] * 180 / M_PI, rotVect0[1] * 180 / M_PI, rotVect0[2] * 180 / M_PI);
                    glm::vec3 eulerAngles1(rotVect1[0] * 180 / M_PI, rotVect1[1] * 180 / M_PI, rotVect1[2] * 180 / M_PI);

                    glm::quat quaternion0 = glm::quat(eulerAngles0);
                    glm::quat quaternion1 = glm::quat(eulerAngles1); */

                    glm::quat quaternion0 = convertOpencvRotVectToQuat(rotVect0);
                    glm::quat quaternion1 = convertOpencvRotVectToQuat(rotVect1);

                    float relativePos = (float)(imuReadVectorCopy[i].time - cameraReadVectorCopy[indexCamera].time) /
                                        (float)(cameraReadVectorCopy[indexCamera + 1].time - cameraReadVectorCopy[indexCamera].time);

                    glm::quat interpolatedPoint = glm::slerp(quaternion0, quaternion1, relativePos);

                    // glm::vec3 rotacionVec3 = glm::eulerAngles(interpolatedPoint);
                    cv::Vec3d rotacionVec3 = convertQuatToOpencvRotVect(interpolatedPoint);

                    tempFrameMarkersData.rvecs.push_back(rotacionVec3);

                    /* tempFrameMarkersData.rvecs.push_back(cv::Vec3d(
                        rotacionVec3.x * M_PI / 180,
                        rotacionVec3.y * M_PI / 180,
                        rotacionVec3.z * M_PI / 180)); */

                    tempFrameMarkersData.tvecs.push_back(frameMarkersDataVector[indexCamera].tvecs[j]);
                }

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
            cv::aruco::drawAxis(interpolatedPoints[i].frame.frame, cameraMatrix, distCoeffs, interpolatedPoints[i].frameMarkersData.rvecs[j],
                                interpolatedPoints[i].frameMarkersData.tvecs[j], 0.1);
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

// Main method that creates threads, writes and read data from files and displays data on console.
int main()
{
    timeIMUStart = std::chrono::steady_clock::now();

    if (generateNewData)
    {
        std::thread cameraCapture(cameraCaptureThread);
        std::thread imu(imuThread);

        cameraCapture.join();
        imu.join();

        cameraDataWrite();
        IMUDataWrite();
    }

    std::vector<ImuInput> imuReadVector = readDataIMU();
    std::vector<CameraInput> cameraReadVector = readDataCamera();

    std::vector<ImuInput> imuReadVectorCopy = imuReadVector;
    std::vector<CameraInput> cameraReadVectorCopy = hardCopyCameraVector(cameraReadVector);

    std::vector<CameraInterpolatedData> interpolatedRotation = interpolateCameraRotation(imuReadVectorCopy, cameraReadVectorCopy);

    testInterpolateCamera(interpolatedRotation);

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
            ImuInput imuData = imuReadVector.at(imuIndex);

            wmove(win, 3, 2);
            snprintf(buff, 511, "Index = %0d", imuData.index);
            waddstr(win, buff);

            wmove(win, 5, 3);
            snprintf(buff, 511, "Acc = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuData.acc.x, imuData.acc.y, imuData.acc.z);
            waddstr(win, buff);

            wmove(win, 7, 2);
            snprintf(buff, 511, "Quat = {W=%06.2f, X=%06.2f, Y=%06.2f, Z=%06.2f}", imuData.rotQuat.w, imuData.rotQuat.x,
                     imuData.rotQuat.y, imuData.rotQuat.z);
            waddstr(win, buff);

            if (imuIndex != 0)
            {
                wmove(win, 9, 2);
                snprintf(buff, 511, "Time between captures (IMU): %010d", imuData.time - oldTimeIMU);
                waddstr(win, buff);

                oldTimeIMU = imuData.time;
            }
            else
            {
                wmove(win, 9, 2);
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

        cv::waitKey(1);

        wrefresh(win);
        wclear(win);

        imuIndex++;
        cameraIndex++;
        stop = stopProgram;
    }

    endwin();

    return 0;
}

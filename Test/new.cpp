#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
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
};

// Struct to store information about each IMU data saved (tests at home).
struct ImuInput
{
    int index;
    int time;
    float accX;
    float accY;
    float accZ;
    float quatX;
    float quatY;
    float quatZ;
    float quatW;
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
bool stopProgram = false;
bool doneCalibrating = false;
bool generateNewData = false;

// Thread in charge of readng data from camera and store it on camera buffer.
void cameraCaptureThread()
{
    cv::VideoCapture cap(0);

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
                    imuInput.accX = parsedData[0];
                    imuInput.accY = parsedData[1];
                    imuInput.accZ = parsedData[2];
                    imuInput.quatW = parsedData[3];
                    imuInput.quatX = parsedData[4];
                    imuInput.quatY = parsedData[5];
                    imuInput.quatZ = parsedData[6];

                    doneCalibrating = true;
                }
                else
                {
                    imuInput.index = index;
                    imuInput.accX = 0;
                    imuInput.accY = 0;
                    imuInput.accZ = 0;
                    imuInput.quatW = 0;
                    imuInput.quatX = 0;
                    imuInput.quatY = 0;
                    imuInput.quatZ = 0;
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
            IMUDataFile << tempIMU.accX << std::endl;
            IMUDataFile << tempIMU.accY << std::endl;
            IMUDataFile << tempIMU.accZ << std::endl;
            IMUDataFile << tempIMU.quatW << std::endl;
            IMUDataFile << tempIMU.quatX << std::endl;
            IMUDataFile << tempIMU.quatY << std::endl;
            IMUDataFile << tempIMU.quatZ << std::endl;
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
            fileData >> tempIMUInput.accX;
            fileData >> tempIMUInput.accY;
            fileData >> tempIMUInput.accZ;
            fileData >> tempIMUInput.quatW;
            fileData >> tempIMUInput.quatX;
            fileData >> tempIMUInput.quatY;
            fileData >> tempIMUInput.quatZ;

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
}

// Create slerp points for quaternions (tests at home).
std::vector<glm::quat> createSlerpPoint(std::vector<ImuInput> imuReadVector)
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

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    cv::Mat cameraMatrix, distCoeffs;

    cameraMatrix = (cv::Mat_<double>(3, 3) << 661.30425, 0, 323.69932,
                    0, 660.76768, 242.771412,
                    0, 0, 1);

    distCoeffs = (cv::Mat_<double>(1, 5) << 0.18494665, -0.76514154, -0.00064337, -0.00251164, 0.79249157);

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
            ImuInput imuData = imuReadVector.at(imuIndex);

            wmove(win, 3, 2);
            snprintf(buff, 511, "Index = %0d", imuData.index);
            waddstr(win, buff);

            wmove(win, 5, 3);
            snprintf(buff, 511, "Acc = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuData.accX, imuData.accY, imuData.accZ);
            waddstr(win, buff);

            wmove(win, 7, 2);
            snprintf(buff, 511, "Quat = {X=%06.2f, Y=%06.2f, Z=%06.2f, W=%06.2f}", imuData.quatX, imuData.quatY, imuData.quatZ, imuData.quatW);
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

    return 0;
}

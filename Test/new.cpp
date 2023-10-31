#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

#include <chrono>
#include <curses.h>
#include <vector>
#include <thread>
#include <mutex>
#include <fstream>
#include "RingBuffer.h"

// Amount of IMU data and frames to read from devices.
#define RINGBUFFERLENGTHCAMERA 100
#define RINGBUFFERLENGTHIMU 200
#define JETSON

#ifdef JETSON
#include <BNO055-BBB_driver.h>
#else
#include <boost/asio.hpp>
#endif

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


// Struct to store information about each IMU data saved (Jetson Board).
struct ImuInputJetson
{
    int index;
    int time;
    float gyroX;
    float gyroY;
    float gyroZ;
    float eulerX;
    float eulerY;
    float eulerZ;
    float quatX;
    float quatY;
    float quatZ;
    float quatW;
    float accX;
    float accY;
    float accZ;
    float gravX;
    float gravY;
    float gravZ;
};

// Buffer to store camera structs.
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(RINGBUFFERLENGTHCAMERA);

// Buffer to store IMU structs.
#ifdef JETSON
RingBuffer<ImuInputJetson> imuDataJetsonBuffer = RingBuffer<ImuInputJetson>(RINGBUFFERLENGTHIMU);
#else
RingBuffer<ImuInput> imuDataBuffer = RingBuffer<ImuInput>(RINGBUFFERLENGTHIMU);
#endif

// Global variables that need to be accessed from different threads or methods.
std::mutex myMutex;
std::chrono::time_point<std::chrono::steady_clock> timeCameraStart;
std::chrono::time_point<std::chrono::steady_clock> timeIMUStart;
std::string dirCameraFolder = "./Data/Camera/";
std::string dirIMUFolder = "./Data/IMU/";
bool stopProgram = false;
bool doneCalibrating = false;

// Pipeline for camera on JEtson Board.
#ifdef JETSON
std::string get_tegra_pipeline(int width, int height, int fps)
{
    return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)I420, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}
#endif

// Thread in charge of readng data from camera and store it on camera buffer.
void cameraCaptureThread()
{
#ifdef JETSON
    int WIDTH = 640;
    int HEIGHT = 360;
    int FPS = 60;
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);
    cv::VideoCapture cap;
    cap.open(pipeline, cv::CAP_GSTREAMER);
#else
    cv::VideoCapture cap(0);
#endif

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
                cv::Mat frame;
                cap.read(frame);

                if (frame.empty())
                {
                    std::cerr << "No se pudo capturar el frame." << std::endl;
                    break;
                }
                else
                {
                    CameraInput capture;
                    capture.index = index;
                    capture.frame = frame.clone();
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

#ifdef JETSON

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

        imuInputJetson.gyroX = sensors.gyroVect.vi[0] * 0.01;
        imuInputJetson.gyroY = sensors.gyroVect.vi[1] * 0.01;
        imuInputJetson.gyroZ = sensors.gyroVect.vi[2] * 0.01;
        imuInputJetson.eulerX = sensors.eOrientation.vi[0] * sensors.Scale;
        imuInputJetson.eulerY = sensors.eOrientation.vi[1] * sensors.Scale;
        imuInputJetson.eulerZ = sensors.eOrientation.vi[2] * sensors.Scale;
        imuInputJetson.quatX = sensors.qOrientation.vi[0] * sensors.Scale;
        imuInputJetson.quatY = sensors.qOrientation.vi[1] * sensors.Scale;
        imuInputJetson.quatZ = sensors.qOrientation.vi[2] * sensors.Scale;
        imuInputJetson.quatW = sensors.qOrientation.vi[3] * sensors.Scale;
        imuInputJetson.accX = sensors.accelVect.vi[0] * sensors.Scale;
        imuInputJetson.accY = sensors.accelVect.vi[1] * sensors.Scale;
        imuInputJetson.accZ = sensors.accelVect.vi[2] * sensors.Scale;
        imuInputJetson.gravX = sensors.gravVect.vi[0] * 0.01;
        imuInputJetson.gravY = sensors.gravVect.vi[1] * 0.01;
        imuInputJetson.gravZ = sensors.gravVect.vi[2] * 0.01;

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
            imuDataBuffer.Dequeue(tempIMU);

            IMUTimeFile << tempIMU.time << std::endl;

            IMUDataFile << tempIMU.index << std::endl;
            IMUDataFile << tempIMU.gyroX << std::endl;
            IMUDataFile << tempIMU.gyroY << std::endl;
            IMUDataFile << tempIMU.gyroZ << std::endl;
            IMUDataFile << tempIMU.eulerX << std::endl;
            IMUDataFile << tempIMU.eulerY << std::endl;
            IMUDataFile << tempIMU.eulerZ << std::endl;
            IMUDataFile << tempIMU.quatX << std::endl;
            IMUDataFile << tempIMU.quatY << std::endl;
            IMUDataFile << tempIMU.quatZ << std::endl;
            IMUDataFile << tempIMU.quatW << std::endl;
            IMUDataFile << tempIMU.accX << std::endl;
            IMUDataFile << tempIMU.accY << std::endl;
            IMUDataFile << tempIMU.accZ << std::endl;
            IMUDataFile << tempIMU.gravX << std::endl;
            IMUDataFile << tempIMU.gravY << std::endl;
            IMUDataFile << tempIMU.gravZ << std::endl;
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
            fileData >> tempIMUInput.gyroX;
            fileData >> tempIMUInput.gyroY;
            fileData >> tempIMUInput.gyroZ;
            fileData >> tempIMUInput.eulerX;
            fileData >> tempIMUInput.eulerY;
            fileData >> tempIMUInput.eulerZ;
            fileData >> tempIMUInput.quatX;
            fileData >> tempIMUInput.quatY;
            fileData >> tempIMUInput.quatZ;
            fileData >> tempIMUInput.quatW;
            fileData >> tempIMUInput.accX;
            fileData >> tempIMUInput.accY;
            fileData >> tempIMUInput.accZ;
            fileData >> tempIMUInput.gravX;
            fileData >> tempIMUInput.gravY;
            fileData >> tempIMUInput.gravZ;

            IMUData.push_back(tempIMUInput);
        }
    }

    return IMUData;
}

#else

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

#endif

// Write camera time data to file and store all frams as .png files.
void cameraDataWrite()
{
    std::ofstream cameraTimeFile(dirCameraFolder + "cameraTime", std::ios::out);

    if (cameraTimeFile.is_open())
    {
        while (!cameraFramesBuffer.QueueIsEmpty())
        {
            CameraInput tempFrame;
            cameraFramesBuffer.Dequeue(tempFrame);
            std::string imageName = "frame_" + std::to_string(tempFrame.index) + ".png";
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

            imageName = "frame_" + std::to_string(index) + ".png";
            image = cv::imread(dirCameraFolder + imageName);
            image.copyTo(tempCameraInput.frame);

            cameraData.push_back(tempCameraInput);
            index++;
        }
    }

    return cameraData;
}

// Main method that creates threads, writes and read data from files and displays data on console.
int main(int argc, char **argv)
{

    timeIMUStart = std::chrono::steady_clock::now();

    std::thread cameraCapture(cameraCaptureThread);

#ifdef JETSON
    std::thread imu(imuThreadJetson);
#else
    std::thread imu(imuThread);
#endif

    cameraCapture.join();
    imu.join();

    cameraDataWrite();
    IMUDataWrite();

    std::vector<ImuInputJetson> imuReadVector = readDataIMU();
    std::vector<CameraInput> cameraReadVector = readDataCamera();

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

#ifdef JETSON
        if (imuIndex < imuReadVector.size())
        {
            ImuInputJetson imuDataJetson = imuReadVector.at(imuIndex);

            wmove(win, 3, 2);
            snprintf(buff, 511, "Index = %0d", imuDataJetson.index);
            waddstr(win, buff);

            wmove(win, 5, 2);
            snprintf(buff, 511, "Gyro = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuDataJetson.gyroX, imuDataJetson.gyroY, imuDataJetson.gyroZ);
            waddstr(win, buff);

            wmove(win, 7, 2);
            snprintf(buff, 511, "Euler = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuDataJetson.eulerX, imuDataJetson.eulerY, imuDataJetson.eulerZ);
            waddstr(win, buff);

            wmove(win, 9, 2);
            snprintf(buff, 511, "Quat = {X=%06.2f, Y=%06.2f, Z=%06.2f, W=%06.2f}", imuDataJetson.quatX,
                     imuDataJetson.quatY, imuDataJetson.quatZ, imuDataJetson.quatW);
            waddstr(win, buff);

            wmove(win, 11, 3);
            snprintf(buff, 511, "Acc = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuDataJetson.accX, imuDataJetson.accY, imuDataJetson.accZ);
            waddstr(win, buff);

            wmove(win, 13, 2);
            snprintf(buff, 511, "Grav = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuDataJetson.gravX, imuDataJetson.gravY, imuDataJetson.gravZ);
            waddstr(win, buff);

            if (imuIndex != 0)
            {
                wmove(win, 15, 2);
                snprintf(buff, 511, "Time between captures (IMU): %010ld", imuDataJetson.time - oldTimeIMU);
                waddstr(win, buff);

                oldTimeIMU = imuDataJetson.time;
            }
            else
            {
                wmove(win, 15, 2);
                snprintf(buff, 511, "Time between captures (IMU): %010ld", 0);
                waddstr(win, buff);
            }
        }

#else
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
                snprintf(buff, 511, "Time between captures (IMU): %010ld", imuData.time - oldTimeIMU);
                waddstr(win, buff);

                oldTimeIMU = imuData.time;
            }
            else
            {
                wmove(win, 9, 2);
                snprintf(buff, 511, "Time between captures (IMU): %010ld", 0);
                waddstr(win, buff);
            }
        }

#endif
        if (cameraIndex < cameraReadVector.size())
        {
            CameraInput frame = cameraReadVector.at(cameraIndex);

            wmove(win, 19, 2);
            snprintf(buff, 511, "Index = %0d", frame.index);
            waddstr(win, buff);

            if (cameraIndex != 0)
            {
                wmove(win, 21, 2);
                snprintf(buff, 511, "Time between captures (IMU): %010ld", frame.time - oldTimeCamera);
                waddstr(win, buff);

                oldTimeCamera = frame.time;
            }
            else
            {
                wmove(win, 21, 2);
                snprintf(buff, 511, "Time between captures (IMU): %010ld", 0);
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

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <boost/asio.hpp>
#include <chrono>
#include <curses.h>
#include <vector>
#include <mutex>
#include "RingBuffer.h"

#define MAXLEN 512 // maximum buffer size
#define LOOPLENGTH 1000

struct CameraInput
{
    cv::Mat frame;
    std::chrono::time_point<std::chrono::steady_clock> timeStamp;
};

struct ImuInput
{
    float accX;
    float accY;
    float accZ;
    float quatX;
    float quatY;
    float quatZ;
    float quatW;
    std::chrono::time_point<std::chrono::steady_clock> timeStamp;
};

std::mutex mutex;
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(LOOPLENGTH);
RingBuffer<ImuInput> imuDataBuffer = RingBuffer<ImuInput>(LOOPLENGTH);

bool capturedNewFrame = false;
bool cameraThreadIsRunning = true;

bool capturedNewImuData = false;
bool imuThreadIsRunning = true;
bool stopProgram = false;

void cameraCaptureThread()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
        std::cerr << "Error al abrir la cámara." << std::endl;
    else
    {
        mutex.lock();
        bool stop = stopProgram;
        mutex.unlock();

        while (!stop)
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
                capture.frame = frame.clone();
                capture.timeStamp = std::chrono::steady_clock::now();
                cameraFramesBuffer.Queue(capture);
                capturedNewFrame = true;
            }
            mutex.lock();
            stop = stopProgram;
            mutex.unlock();
        }
    }
    mutex.lock();
    cameraThreadIsRunning = false;
    mutex.unlock();
}

void cameraDisplayThread()
{
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    cv::Mat cameraMatrix, distCoeffs;

    cameraMatrix = (cv::Mat_<double>(3, 3) << 661.30425, 0, 323.69932,
                    0, 660.76768, 242.771412,
                    0, 0, 1);

    distCoeffs = (cv::Mat_<double>(1, 5) << 0.18494665, -0.76514154, -0.00064337, -0.00251164, 0.79249157);

    mutex.lock();
    bool keepLooping = cameraThreadIsRunning;
    mutex.unlock();

    // auto tempTime = std::chrono::steady_clock::now();

    while (keepLooping)
    {
        if (capturedNewFrame)
        {
            CameraInput frame;
            cameraFramesBuffer.Dequeue(frame);

            /*auto timePassedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(frame.timeStamp - tempTime);
            std::cout << "Time between captures (Camera): " << timePassedMilliseconds.count() << std::endl;
            tempTime = frame.timeStamp;*/

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
            capturedNewFrame = false;

            cv::waitKey(33);
        }
        mutex.lock();
        keepLooping = cameraThreadIsRunning;
        mutex.unlock();
    }
    cv::destroyAllWindows();
}

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

void imuThread()
{
    boost::asio::io_service io;
    boost::asio::serial_port serial(io);

    try
    {
        serial.open("/dev/ttyACM0");
        serial.set_option(boost::asio::serial_port_base::baud_rate(9600));
        boost::asio::streambuf buffer;

        mutex.lock();
        bool stop = stopProgram;
        mutex.unlock();

        while (!stop)
        {
            boost::system::error_code ec;
            boost::asio::read_until(serial, buffer, '\n', ec);

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

                ImuInput imuInput;
                imuInput.timeStamp = std::chrono::steady_clock::now();

                if (parsedData.size() == 7)
                {
                    imuInput.accX = parsedData[0];
                    imuInput.accY = parsedData[1];
                    imuInput.accZ = parsedData[2];
                    imuInput.quatW = parsedData[3];
                    imuInput.quatX = parsedData[4];
                    imuInput.quatY = parsedData[5];
                    imuInput.quatZ = parsedData[6];
                }
                else
                {
                    imuInput.accX = 0;
                    imuInput.accY = 0;
                    imuInput.accZ = 0;
                    imuInput.quatW = 0;
                    imuInput.quatX = 0;
                    imuInput.quatY = 0;
                    imuInput.quatZ = 0;
                }

                imuDataBuffer.Queue(imuInput);
                mutex.lock();
                capturedNewImuData = true;
                mutex.unlock();
            }
            mutex.lock();
            stop = stopProgram;
            mutex.unlock();
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    serial.close();

    mutex.lock();
    imuThreadIsRunning = false;
    mutex.unlock();
}

void imuDisplayThread()
{
    WINDOW *win;
    char buff[512];

    mutex.lock();
    bool keepLooping = imuThreadIsRunning;
    mutex.unlock();

    win = initscr();
    clearok(win, TRUE);

    auto tempTime = std::chrono::steady_clock::now();

    while (keepLooping)
    {
        if (capturedNewImuData)
        {
            ImuInput imuData;
            imuDataBuffer.Dequeue(imuData);

            auto timePassedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(imuData.timeStamp - tempTime);

            wmove(win, 5, 3);
            snprintf(buff, 511, "Acc[%010ld] = {%06.2f, %06.2f, %06.2f}", timePassedMilliseconds.count(), imuData.accX, imuData.accY, imuData.accZ);
            waddstr(win, buff);

            wmove(win, 7, 2);
            snprintf(buff, 511, "Quat[%010ld] = {%06.2f, %06.2f, %06.2f, %06.2f}", timePassedMilliseconds.count(), imuData.quatX, imuData.quatY, imuData.quatZ, imuData.quatW);
            waddstr(win, buff);

            wmove(win, 9, 2);
            snprintf(buff, 511, "Time between captures (IMU): %010ld", timePassedMilliseconds.count());
            waddstr(win, buff);
            tempTime = imuData.timeStamp;

            capturedNewImuData = false;
            wrefresh(win);
        }
        mutex.lock();
        keepLooping = imuThreadIsRunning;
        mutex.unlock();
    }
    endwin();
}

int main(int argc, char **argv)
{

    std::thread cameraCapture(cameraCaptureThread);
    std::thread cameraDisplay(cameraDisplayThread);
    std::thread imu(imuThread);
    // std::thread imuDisplay(imuDisplayThread);

    mutex.lock();
    bool stop = stopProgram;
    mutex.unlock();

    WINDOW *win;
    char buff[512];

    win = initscr();
    clearok(win, TRUE);

    while (!stop)
    {
        if (cv::waitKey(1) == 'q')
        {
            stopProgram = true;
        }

        usleep(100);
        mutex.lock();
        stop = stopProgram;
        mutex.unlock();

        auto tempTime = std::chrono::steady_clock::now();

        mutex.lock();
        if (capturedNewImuData)
        {
            ImuInput imuData;
            imuDataBuffer.Dequeue(imuData);

            auto timePassedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(imuData.timeStamp - tempTime);

            wmove(win, 5, 3);
            snprintf(buff, 511, "Acc[%010ld] = {%06.2f, %06.2f, %06.2f}", timePassedMilliseconds.count(), imuData.accX, imuData.accY, imuData.accZ);
            waddstr(win, buff);

            wmove(win, 7, 2);
            snprintf(buff, 511, "Quat[%010ld] = {%06.2f, %06.2f, %06.2f, %06.2f}", timePassedMilliseconds.count(), imuData.quatX, imuData.quatY, imuData.quatZ, imuData.quatW);
            waddstr(win, buff);

            wmove(win, 9, 2);
            snprintf(buff, 511, "Time between captures (IMU): %010ld", timePassedMilliseconds.count());
            waddstr(win, buff);

            tempTime = imuData.timeStamp;
            capturedNewImuData = false;
            wrefresh(win);
        }
        mutex.unlock();
    }

    cameraCapture.join();
    cameraDisplay.join();
    imu.join();
    // imuDisplay.join();

    endwin();

    return 0;
}
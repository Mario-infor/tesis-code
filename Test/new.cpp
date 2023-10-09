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

size_t cameraCaptureSize = 1000;
size_t indexCamera = 0;
std::mutex mutex;
// std::vector<CameraInput> cameraFramesList;
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(cameraCaptureSize);
RingBuffer<ImuInput> imuDataBuffer = RingBuffer<ImuInput>(cameraCaptureSize);
bool cameraRun = true;
bool imuRun = true;
bool capturedNewFrame = false;
bool cameraThreadIsRunning = true;

void cameraCaptureThread()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
        std::cerr << "Error al abrir la cámara." << std::endl;
    else
    {
        for (size_t i = 0; i < cameraCaptureSize; i++)
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

    size_t localCameraIndex = 0;

    mutex.lock();
    bool keepLooping = cameraThreadIsRunning;
    mutex.unlock();

    while (keepLooping)
    {
        if (capturedNewFrame)
        {
            CameraInput frame;
            cameraFramesBuffer.Dequeue(frame);

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
            localCameraIndex += 1;
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

        cv::waitKey(3000);

        for (size_t i = 0; i < cameraCaptureSize; i++)
        {
            std::cout << "Ite: " << i << std::endl;

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
            }
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    serial.close();
}

int main(int argc, char **argv)
{
    // WINDOW *win;
    // std::thread cameraCapture(cameraCaptureThread);
    // std::thread cameraDisplay(cameraDisplayThread);
    std::thread imu(imuThread);

    // win = initscr();
    // clearok(win, TRUE);

    // cameraCapture.join();
    // cameraDisplay.join();
    imu.join();

    /*while (true)
    {
        if (cv::waitKey(0) == 'q')
        {
            imuRun = false;
        }
    }*/

    // auto start = cameraFramesList[0].timeStamp;
    /*CameraInput start;
    cameraFramesBuffer.Dequeue(start);

    for (size_t i = 1; i < cameraFramesBuffer.getT(); i++)
    {
        CameraInput end;
        cameraFramesBuffer.Dequeue(end);
        auto timePassedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end.timeStamp - start.timeStamp);
        std::cout << "Time between captures: " << timePassedMilliseconds.count() << std::endl;
        start = end;
    }*/

    for (size_t i = 0; i < imuDataBuffer.getT(); i++)
    {
        ImuInput temp;
        imuDataBuffer.Dequeue(temp);
        // X:0.01,Y:0.00,Z:0.19@W:0.9999,X:-0.0130,Y:0.0082,Z:0.0000
        std::cout << "Ite: " << i << " "
                  << "X:" << temp.accX << ","
                  << "Y:" << temp.accY << ","
                  << "Z:" << temp.accZ << "@"
                  << "W:" << temp.quatW << ","
                  << "X:" << temp.quatX << ","
                  << "Y:" << temp.quatY << ","
                  << "Z:" << temp.quatZ << std::endl;
    }

    // endwin();
    return 0;
}
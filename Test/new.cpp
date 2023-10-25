#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <chrono>
#include <curses.h>
#include <vector>
#include <thread>
#include <mutex>
#include "RingBuffer.h"

#define RINGBUFFERLENGTH 100

#ifdef JETSON
    #include <BNO055-BBB_driver.h>
#else
    #include <boost/asio.hpp>
#endif

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

struct ImuInputJetson
{
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
    std::chrono::time_point<std::chrono::steady_clock> timeStamp;
};

std::mutex myMutex;
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(RINGBUFFERLENGTH);

#ifdef JETSON
    RingBuffer<ImuInputJetson> imuDataJetsonBuffer = RingBuffer<ImuInputJetson>(RINGBUFFERLENGTH);
#else
    RingBuffer<ImuInput> imuDataBuffer = RingBuffer<ImuInput>(RINGBUFFERLENGTH);
#endif

bool capturedNewFrame = false;
bool capturedNewImuData = false;
bool imuThreadIsRunning = true;
bool stopProgram = false;


#ifdef JETSON
std::string get_tegra_pipeline(int width, int height, int fps)
{
    return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)I420, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}
#endif

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
        myMutex.lock();
        bool stop = stopProgram;
        myMutex.unlock();

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
            myMutex.lock();
            stop = stopProgram;
            myMutex.unlock();
        }
    }
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

#ifdef JETSON

void imuThreadJetson()
{
    int cont = 0;
    bool flag;
    char filename[] = "/dev/i2c-1";
    BNO055 sensors;
    sensors.openDevice(filename);

    do
    {
        sensors.readCalibVals();
        flag = sensors.calSys == 3 && sensors.calMag == 3 && sensors.calGyro == 3 && sensors.calAcc == 3;
    } while (cont++ < 2000 && !flag);

    myMutex.lock();
    bool stop = stopProgram;
    myMutex.unlock();

    while (!stop)
    {
        sensors.readAll();

        ImuInputJetson imuInputJetson;
        imuInputJetson.timeStamp = std::chrono::steady_clock::now();

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
        myMutex.lock();
        capturedNewImuData = true;
        stop = stopProgram;
        myMutex.unlock();
    }
    myMutex.lock();
    stop = stopProgram;
    myMutex.unlock();
}

#else
void imuThread()
{
    boost::asio::io_service io;
    boost::asio::serial_port serial(io);

    try
    {
        serial.open("/dev/ttyACM0");
        serial.set_option(boost::asio::serial_port_base::baud_rate(9600));
        boost::asio::streambuf buffer;

        myMutex.lock();
        bool stop = stopProgram;
        myMutex.unlock();

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
                myMutex.lock();
                capturedNewImuData = true;
                myMutex.unlock();
            }
            myMutex.lock();
            stop = stopProgram;
            myMutex.unlock();
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    serial.close();

    myMutex.lock();
    imuThreadIsRunning = false;
    myMutex.unlock();
}

#endif

int main(int argc, char **argv)
{
    std::thread cameraCapture(cameraCaptureThread);

    #ifdef JETSON
        std::thread imu(imuThreadJetson);
    #else
        std::thread imu(imuThread);
    #endif
    

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

    auto tempTimeImu = std::chrono::steady_clock::now();
    auto tempTimeCamera = std::chrono::steady_clock::now();
    
    win = initscr();
    clearok(win, TRUE);

    while (!stop)
    {
        if (cv::waitKey(1) == 'q')
        {
            stopProgram = true;
        }

        myMutex.lock();
        stop = stopProgram;
        myMutex.unlock();

        myMutex.lock();
        if (capturedNewImuData)
        {
            #ifdef JETSON
                ImuInputJetson imuDataJetson;
                imuDataJetsonBuffer.Dequeue(imuDataJetson);

                auto timePassedMillisecondsImuJetson = std::chrono::duration_cast<std::chrono::milliseconds>(imuDataJetson.timeStamp - tempTimeImu);

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

                wmove(win, 15, 2);
                snprintf(buff, 511, "Time between captures (IMU): %010ld", timePassedMillisecondsImuJetson.count());
                waddstr(win, buff);

                tempTimeImu = imuDataJetson.timeStamp;

            #else
                ImuInput imuData;
                imuDataBuffer.Dequeue(imuData);

                auto timePassedMillisecondsImu = std::chrono::duration_cast<std::chrono::milliseconds>(imuData.timeStamp - tempTimeImu);

                wmove(win, 5, 3);
                snprintf(buff, 511, "Acc = {X=%06.2f, Y=%06.2f, Z=%06.2f}", imuData.accX, imuData.accY, imuData.accZ);
                waddstr(win, buff);

                wmove(win, 7, 2);
                snprintf(buff, 511, "Quat = {X=%06.2f, Y=%06.2f, Z=%06.2f, W=%06.2f}", imuData.quatX, imuData.quatY, imuData.quatZ, imuData.quatW);
                waddstr(win, buff);

                wmove(win, 9, 2);
                snprintf(buff, 511, "Time between captures (IMU): %010ld", timePassedMillisecondsImu.count());
                waddstr(win, buff);

                tempTimeImu = imuData.timeStamp;
            #endif
            capturedNewImuData = false;
        }
        myMutex.unlock();

        myMutex.lock();
        if (capturedNewFrame)
        {
            CameraInput frame;
            cameraFramesBuffer.Dequeue(frame);

            auto timePassedMillisecondsCamera = std::chrono::duration_cast<std::chrono::milliseconds>(frame.timeStamp - tempTimeCamera);

            wmove(win, 19, 2);
            snprintf(buff, 511, "Time between captures (Camera): %010ld", timePassedMillisecondsCamera.count());
            waddstr(win, buff);

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

            //cv::imshow("draw axis", frame.frame);

            tempTimeCamera = frame.timeStamp;
            capturedNewFrame = false;

            cv::waitKey(33);
        }
        myMutex.unlock();
        wrefresh(win);
	wclear(win);
    }

    cameraCapture.join();
    imu.join();

    endwin();

    return 0;
}

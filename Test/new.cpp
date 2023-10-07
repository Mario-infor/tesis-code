#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <boost/asio.hpp>
#include <chrono>
#include <curses.h>
#include <vector>
#include "RingBuffer.h"

#define MAXLEN 512 // maximum buffer size

struct CameraInput
{
    cv::Mat frame;
    std::chrono::time_point<std::chrono::steady_clock> timeStamp;
};

size_t cameraCaptureSize = 100;

std::vector<CameraInput> cameraFramesList;
RingBuffer<CameraInput> cameraFramesBuffer = RingBuffer<CameraInput>(cameraCaptureSize);
bool cameraRun = true;
bool imuRun = true;

void cameraCaptureThread()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
        std::cerr << "Error al abrir la cámara." << std::endl;
    else
    {
        for (size_t i = 0; i < cameraCaptureSize; i++)
        {
            std::cout << "Ite: " << i << std::endl;
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
                //cameraFramesList.push_back(capture);
                cameraFramesBuffer.Queue(capture);
            }
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
        //int nbytes = -1;
        boost::asio::streambuf buffer;

        while(imuRun)
        {
            auto start = std::chrono::steady_clock::now();
            //nbytes = -1;

            boost::system::error_code ec;
            // Lee hasta encontrar un '\n'
            boost::asio::read_until(serial, buffer, '\n', ec);
            // Convierte el contenido del buffer en una cadena de texto
            if (ec)
            {
                std::cout << ec.what();
            }
            else
            {
                std::string receivedData;
                std::istream is(&buffer);
                std::getline(is, receivedData);

                // std::cout << "Ite: " << i << " Datos recibidos: " << nbytes << " |" << receivedData
                //           << "|" << std::endl;
            }
            auto end = std::chrono::steady_clock::now();
            auto timePassedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "IMU time passed: " << timePassedMilliseconds.count() << std::endl;
            std::cout.flush();
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    serial.close();
    std::cout << "Imu Finished!!" << std::endl;
}

int main(int argc, char **argv)
{

    //WINDOW *win;
    std::thread camera(cameraCaptureThread);
    //thread imu(imuThread);


    //win = initscr();
    //clearok(win, TRUE);
    

    
    camera.join();
    //imu.join();

    //auto start = cameraFramesList[0].timeStamp;
    CameraInput start;
    cameraFramesBuffer.Dequeue(start);

    for (size_t i = 1; i < cameraFramesBuffer.getT(); i++)
    {
        CameraInput end;
        cameraFramesBuffer.Dequeue(end);
        auto timePassedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end.timeStamp - start.timeStamp);
        std::cout << "Time between captures: " << timePassedMilliseconds.count() << std::endl;
        start = end;
    }
    

    //endwin();
    return 0;
}
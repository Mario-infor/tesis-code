#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <boost/asio.hpp>
#include <chrono>

using namespace std;
using namespace boost;

#define MAXLEN 512 // maximum buffer size

void cameraThread()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
        std::cerr << "Error al abrir la cámara." << std::endl;
    else
    {
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

        cv::Mat cameraMatrix, distCoeffs;

        cameraMatrix = (cv::Mat_<double>(3, 3) << 661.30425, 0, 323.69932,
                        0, 660.76768, 242.771412,
                        0, 0, 1);

        distCoeffs = (cv::Mat_<double>(1, 5) << 0.18494665, -0.76514154, -0.00064337, -0.00251164, 0.79249157);

        while (true)
        {
            auto start = chrono::steady_clock::now();
            cv::Mat frame;
            cap.read(frame);

            if (frame.empty())
            {
                std::cerr << "No se pudo capturar el frame." << std::endl;
                break;
            }

            std::vector<int> markerIds;
            std::vector<std::vector<cv::Point2f>> markerCorners;

            cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds);

            if (markerIds.size() > 0)
            {
                cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

                std::vector<cv::Vec3d> rvecs, tvecs;
                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

                for (int i = 0; i < (int)rvecs.size(); i++)
                {
                    cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
                }
            }

            cv::imshow("draw axis", frame);

            auto end = chrono::steady_clock::now();
            auto timePassedMilliseconds = chrono::duration_cast<chrono::milliseconds>(end - start);

            cout << "Camera time passed: " << timePassedMilliseconds.count() << endl;

            if (cv::waitKey(1) == 'q')
                break;
        }
    }
    cout << "Camera Finished!!" << endl;
}

void imuThread()
{
    asio::io_service io;
    asio::serial_port serial(io);

    try
    {
        serial.open("/dev/ttyACM0");
        serial.set_option(asio::serial_port_base::baud_rate(9600));
        int nbytes = -1;
        asio::streambuf buffer;

        for (size_t i = 0; i < 100; i++)
        {
            auto start = chrono::steady_clock::now();
            nbytes = -1;

            boost::system::error_code ec;
            // Lee hasta encontrar un '\n'
            nbytes = asio::read_until(serial, buffer, '\n', ec);
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

                //std::cout << "Ite: " << i << " Datos recibidos: " << nbytes << " |" << receivedData
                //          << "|" << std::endl;
            }
            auto end = chrono::steady_clock::now();
            auto timePassedMilliseconds = chrono::duration_cast<chrono::milliseconds>(end - start);

            cout << "IMU time passed: " << timePassedMilliseconds.count() << endl;
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    serial.close();
    cout << "Imu Finished!!" << endl;
}

int main(int argc, char **argv)
{
    //thread camera(cameraThread);
    thread imu(imuThread);

    //camera.join();
    imu.join();

    cout << "Main Finished!!" << endl;

    return 0;
}
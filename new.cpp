#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <boost/asio.hpp>
#include "SimpleSerial.h"

using namespace std;
using namespace boost;

#define MAXLEN 512 // maximum buffer size

int main(int argc, char **argv)
{
    /*cv::Mat markerImage;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::drawMarker(dictionary, 23, 200, markerImage, 1);
    cv::imwrite("marker23.png", markerImage);

    cv::Mat inputImage = cv::imread("singlemarkersoriginal.jpg", cv::IMREAD_COLOR);
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    cv::Mat outputImage = inputImage.clone();
    cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);

    cv::namedWindow("draw markers", cv::WINDOW_NORMAL);
    cv::imshow("draw markers", outputImage);
    cv::imwrite("drawMarkers.png", outputImage);
    // cv::waitKey(0);

    cv::Mat cameraMatrix, distCoeffs;

    cameraMatrix = (cv::Mat_<double>(3, 3) << 661.30425, 0, 323.69932,
                    0, 660.76768, 242.771412,
                    0, 0, 1);

    distCoeffs = (cv::Mat_<double>(1, 5) << 0.18494665, -0.76514154, -0.00064337, -0.00251164, 0.79249157);

    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

    inputImage.copyTo(outputImage);
    for (int i = 0; i < (int)rvecs.size(); ++i)
    {
        auto rvec = rvecs[i];
        auto tvec = tvecs[i];
        cv::aruco::drawAxis(outputImage, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
    }
    cv::waitKey(0);

    cv::namedWindow("draw axis", cv::WINDOW_NORMAL);
    cv::imshow("draw axis", outputImage);
    cv::imwrite("drawAxis.png", outputImage);
    cv::waitKey(0);*/

    asio::io_service io;
    // create a serial port object
    asio::serial_port serial(io);

    try 
    {
        serial.open("/dev/ttyACM0"); // Reemplaza con el nombre de tu puerto serial
        serial.set_option(asio::serial_port_base::baud_rate(9600)); // Configura la velocidad de baudios

        for (int i = 0; i < 10000; i++)
        {
            asio::streambuf buffer;
            asio::read_until(serial, buffer, '$'); // Lee hasta encontrar un '\n'

            // Convierte el contenido del buffer en una cadena de texto
            std::istream is(&buffer);
            std::string receivedData;
            std::getline(is, receivedData);

            // Procesa y muestra los datos leídos
            std::cout << "Datos recibidos: " << receivedData << std::endl;
        }

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    serial.close();

    /*
    try
    {
        // open the platform specific device name
        // windows will be COM ports, linux will use /dev/ttyS* or /dev/ttyUSB*, etc
        serial.open("/dev/ttyACM0");
        serial.set_option(asio::serial_port_base::baud_rate(9600)); // Configura la velocidad de baudios

        char data[256]; // Un buffer para almacenar los datos leídos

        for (int i = 0; i < 10; i++)
        {
            size_t bytesRead = asio::read(serial, asio::buffer(data, sizeof(data)));

            // Procesa y muestra los datos leídos
            std::string receivedData(data, bytesRead);
            std::cout << "Datos recibidos: " << receivedData << std::endl;

            // Buffer para almacenar los datos leídos
            boost::asio::streambuf buffer; 

            // Función de lectura asincrónica
            boost::asio::async_read_until(serial, buffer, '@',
                                          [&](const boost::system::error_code &error, std::size_t bytes_transferred)
                                          {
                                              if (!error)
                                              {
                                                  std::istream input_stream(&buffer);
                                                  std::string line;
                                                  std::getline(input_stream, line); // Leer una línea hasta el '\n'
                                                  std::cout << "Cadena recibida: " << line << std::endl;
                                              }
                                              else
                                              {
                                                  std::cerr << "Error de lectura: " << error.message() << std::endl;
                                              }
                                          });
        }

        serial.close();
    }
    catch (boost::system::system_error &e)
    {
        std::cerr << e.what() << std::endl;
    }*/

    return 0;
}
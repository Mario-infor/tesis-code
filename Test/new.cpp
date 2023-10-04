#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <boost/asio.hpp>

using namespace std;
using namespace boost;

#define MAXLEN 512 // maximum buffer size

void cameraThread()
{
    /*cv::Mat markerImage;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::drawMarker(dictionary, 23, 200, markerImage, 1);
    cv::imwrite("marker23.png", markerImage);
    cv::Mat inputImage = cv::imread("singlemarkersoriginal.jpg", cv::IMREAD_COLOR);*/

    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "Error al abrir la cámara." << std::endl;
    }
    else
    {
        while (true)
        {
            cv::Mat frame;
            cap.read(frame);

            if (frame.empty())
            {
                std::cerr << "No se pudo capturar el frame." << std::endl;
                break;
            }

            cv::imshow("Video de la cámara", frame);

            // Presiona la tecla 'q' para salir del bucle
            if (cv::waitKey(1) == 'q')
            {
                cap.release();           // Libera la captura de video
                cv::destroyAllWindows(); // Cierra todas las ventanas

                break;
            }
        }
    }

    /*std::vector<int> markerIds;
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
}

void imuThread()
{
    asio::io_service io;
    // create a serial port object
    asio::serial_port serial(io);

    try
    {
        serial.open("/dev/ttyACM0");                                // Reemplaza con el nombre de tu puerto serial
        serial.set_option(asio::serial_port_base::baud_rate(9600)); // Configura la velocidad de baudios

        int nbytes = -1;
        asio::streambuf buffer;

        for (int i = 0; i < 10000; i++)
        {
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

                // Procesa y muestra los datos leídos

                std::cout << "Datos recibidos: " << nbytes << " |" << receivedData
                          << "|" << std::endl;
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
    thread camera(cameraThread);
    // thread imu(imuThread);

    camera.join();
    // imu.join();

    cout << "Camera Finished!!" << endl;
    return 0;
}
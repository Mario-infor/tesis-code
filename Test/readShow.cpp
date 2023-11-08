#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#ifdef JETSON
#include <BNO055-BBB_driver.h>
#else
#include <boost/asio.hpp>
#endif

// Struct to store information about each frame saved.
struct CameraInput
{
    cv::Mat frame;
};

// Main method that creates threads, writes and read data from files and displays data on console.
int main(int argc, char **argv)
{
    bool stopProgram = false;

    while (!stopProgram)
    {
        if (cv::waitKey(1) == 'q')
        {
            stopProgram = true;
        }

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

            cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

            cv::Mat cameraMatrix, distCoeffs;

            cameraMatrix = (cv::Mat_<double>(3, 3) << 661.30425, 0, 323.69932,
                            0, 660.76768, 242.771412,
                            0, 0, 1);

            distCoeffs = (cv::Mat_<double>(1, 5) << 0.18494665, -0.76514154, -0.00064337, -0.00251164, 0.79249157);

            while (!stopProgram)
            {
                if (cv::waitKey(1) == 'q')
                {
                    stopProgram = true;
                }
                
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
                    capture.frame = grayscale.clone();

                    std::vector<int> markerIds;
                    std::vector<std::vector<cv::Point2f>> markerCorners;

                    cv::aruco::detectMarkers(capture.frame, dictionary, markerCorners, markerIds);

                    if (markerIds.size() > 0)
                    {
                        cv::aruco::drawDetectedMarkers(capture.frame, markerCorners, markerIds);

                        std::vector<cv::Vec3d> rvecs, tvecs;
                        cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

                        for (int i = 0; i < (int)rvecs.size(); i++)
                        {
                            cv::aruco::drawAxis(capture.frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
                        }
                    }

                    cv::imshow("draw axis", capture.frame);
                }
            }
        }
    }

    return 0;
}

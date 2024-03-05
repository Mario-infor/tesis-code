#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <BNO055-BBB_driver.h>

// Struct to store information about each frame saved.
struct CameraInput
{
    cv::Mat frame;
};

// Pipeline for camera on JEtson Board.
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

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

        int capture_width = 800 ;
        int capture_height = 600 ;
        int display_width = 800 ;
        int display_height = 600 ;
        int framerate = 30 ;
        int flip_method = 0 ;

        std::string pipeline = gstreamer_pipeline(capture_width,
            capture_height,
            display_width,
            display_height,
            framerate,
            flip_method);

        cv::VideoCapture cap;
        cap.open(pipeline, cv::CAP_GSTREAMER);

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

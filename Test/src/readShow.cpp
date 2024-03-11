#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <readShow.h>
#include <cameraInfo.h>

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

        std::string pipeline = gstreamerPipeline(FRAME_WIDTH,
            FRAME_HEIGHT,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            FRAME_RATE,
            FLIP_METHOD);

        cv::VideoCapture cap;
        cap.open(pipeline, cv::CAP_GSTREAMER);

        if (!cap.isOpened())
            std::cerr << "Error al abrir la cámara." << std::endl;
        else
        {
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

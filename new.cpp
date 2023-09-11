#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <boost/asio.hpp>
#include "SimpleSerial.h"

using namespace std;
using namespace boost;

int main(int argc, char **argv)
{
    cv::Mat markerImage;
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
    //cv::waitKey(0);

    cv::Mat cameraMatrix, distCoeffs;

    cameraMatrix = (cv::Mat_<double>(3, 3) <<
    661.30425, 0, 323.69932,
    0, 660.76768, 242.771412,
    0, 0, 1);

    distCoeffs = (cv::Mat_<double>(1, 5) <<
    0.18494665, -0.76514154, -0.00064337, -0.00251164, 0.79249157);

    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);


    inputImage.copyTo(outputImage);
    for (int i = 0; i < (int)rvecs.size(); ++i) {
        auto rvec = rvecs[i];
        auto tvec = tvecs[i];
        cv::aruco::drawAxis(outputImage, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
    }cv::waitKey(0);

    cv::namedWindow("draw axis", cv::WINDOW_NORMAL);
    cv::imshow("draw axis", outputImage);
    cv::imwrite("drawAxis.png", outputImage);
    cv::waitKey(0);

    
    try
    {

        SimpleSerial serial("/dev/ttyACM0", 9600);

        serial.writeString("Hello world\n");

        cout << "Received : " << serial.readLine() << " : end" << endl;
    }
    catch (boost::system::system_error &e)
    {
        cout << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
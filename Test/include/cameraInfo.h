#ifndef CAMERAINFO_H
#define CAMERAINFO_H

#define FRAME_WIDTH 800
#define FRAME_HEIGHT 600
#define FRAME_RATE 30
#define FLIP_METHOD 0

/*cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    595.8977808485818,   0,                 409.12509669936657,
    0,                   791.330070013496,  295.52098484167175,
    0,                   0,                 1);

cv::Mat distCoeffs = (cv::Mat_<double>(1, 4) << 0.12912260003585463, \
                                                -0.25388754628599064, \
                                                0.000943128326254423, \
                                                0.0018654081762782874);*/

cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    592.1816253510633,   0,                 406.12406274169103,
    0,                   786.637131732552,  297.12988589591276,
    0,                   0,                 1);

cv::Mat distCoeffs = (cv::Mat_<double>(1, 4) << 0.1040194778195083, \
                                                -0.20497642380015912, \
                                                0.0022836648701707583, \
                                                5.6217918934802215e-05);

cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
cv::Ptr<cv::aruco::DetectorParameters>  detectorParams = cv::aruco::DetectorParameters::create();

// Pipeline for camera on Jetson Board.
std::string gstreamerPipeline (int captureWidth, int captureHeight, int displayWidth, int displayHeight, int framerate, int flipMethod) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(captureWidth) + ", height=(int)" +
           std::to_string(captureHeight) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flipMethod) + " ! video/x-raw, width=(int)" + std::to_string(displayWidth) + ", height=(int)" +
           std::to_string(displayHeight) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

/*
cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    493.02975478,   0,             310.67004724,
    0,              495.25862058,  166.53292108,
    0,              0,             1);

cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.12390713, 0.17792574, -0.00934536, -0.01052198, -1.13104202);
*/
#endif // CAMERAINFO_H
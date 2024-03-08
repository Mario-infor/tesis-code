#ifndef CAMERAINFO_H
#define CAMERAINFO_H

cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    791.330070013496,   0,                  409.12509669936657,
    0,                  595.8977808485818,  295.52098484167175,
    0,                  0,                  1);

cv::Mat distCoeffs = (cv::Mat_<double>(1, 4) << 0.12912260003585463, \
                                                -0.25388754628599064, \
                                                0.000943128326254423, \
                                                0.0018654081762782874);

cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
/*
cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    493.02975478,   0,             310.67004724,
    0,              495.25862058,  166.53292108,
    0,              0,             1);

cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.12390713, 0.17792574, -0.00934536, -0.01052198, -1.13104202);
*/
#endif // CAMERAINFO_H
#ifndef CAMERAINFO_H
#define CAMERAINFO_H

cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 493.02975478, 0, 310.67004724, \
                        0, 495.25862058, 166.53292108, \
                        0, 0, 1);

cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.12390713, 0.17792574, -0.00934536, -0.01052198, -1.13104202);
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

#endif // CAMERAINFO_H
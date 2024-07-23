#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <utils.h>
#include <iostream>
#include <BNO055-BBB_driver.h>
#include <curses.h>
#include <iterator>
#include <cmath>
#include <readWriteData.h>
#include <map>


void printIMUData()
{
    
    char filename[] = IMU_ADDRESS;
    BNO055 sensors;
    sensors.openDevice(filename);

    WINDOW *win;
    char buff[512];

    win = initscr();
    clearok(win, TRUE);

    while (true)
    {
        sensors.readAll();
        sensors.readAxisRemap();

        wmove(win, 5, 2);

        wmove(win, 5, 2);
        snprintf(buff, 512, "Gravity=[%07.5lf, %07.5lf, %07.3lf]",
                 sensors.gravVect.vi[0] * 0.01,
                 sensors.gravVect.vi[1] * 0.01,
                 sensors.gravVect.vi[2] * 0.01);
        waddstr(win, buff);

        wmove(win, 7, 2);
        snprintf(buff, 512, "axis=[axisConfig=%d, axisSign=%d]",
                 sensors.axisConfig,
                 sensors.axisSign);
        waddstr(win, buff);

        wmove(win, 9, 2);
        snprintf(buff, 512, "quat=[q=%07.5lf, x=%07.5lf, y=%07.5lf, z=%07.5lf]",
                 sensors.qOrientation.vi[3] * sensors.Scale,
                 sensors.qOrientation.vi[0] * sensors.Scale,
                 sensors.qOrientation.vi[1] * sensors.Scale,
                 sensors.qOrientation.vi[2] * sensors.Scale);
        waddstr(win, buff);

        wmove(win, 11, 2);
        snprintf(buff, 512, "operationModeRead=[operationModeRead=%d]",
                 sensors.operationModeRead);
        waddstr(win, buff);

        wrefresh(win);
        wclear(win);
    }
    endwin();
}

FrameMarkersData getRotationTraslationFromFrame(
    CameraInput frame,
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs,
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams)
{
    FrameMarkersData frameMarkersData;

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    std::vector<cv::Mat> rejectedCandidates, cameraMatrixVector, distCoeffsVector;
    cameraMatrixVector.push_back(cameraMatrix);
    distCoeffsVector.push_back(distCoeffs);

    cv::aruco::detectMarkers(
                frame.frame,
                dictionary,
                markerCorners,
                markerIds,
                detectorParams,
                rejectedCandidates,
                cameraMatrixVector,
                distCoeffsVector);

    if (markerIds.size() > 0)
    {
        cv::aruco::drawDetectedMarkers(frame.frame, markerCorners, markerIds);

        std::vector<cv::Vec3d> rvecs, tvecs;

        cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.17, cameraMatrix, distCoeffs, rvecs, tvecs);

        frameMarkersData.markerIds = markerIds;
        frameMarkersData.rvecs = rvecs;
        frameMarkersData.tvecs = tvecs;
    }

    return frameMarkersData;
}

void drawAxisOnFrame(
    std::vector<cv::Vec3d> rvecs,
    std::vector<cv::Vec3d> tvecs,
    cv::Mat frame,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs,
    std::string windowTitle)
{
    for (int i = 0; i < (int)rvecs.size(); i++)
    {
        cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.02);
    }

    cv::imshow(windowTitle, frame);
}

Eigen::Matrix3d getWHat(const Eigen::Vector3d w)
{
    Eigen::Matrix3d wHat;
    wHat << 
    0,      -w.z(),  w.y(),
    w.z(),   0,      -w.x(),
    -w.y(),  w.x(),   0;

    return wHat;
}

Eigen::Matrix4d getGhi(const Eigen::Vector3d w, const Eigen::Vector3d v)
{
    Eigen::Matrix3d wHat = getWHat(w);
    Eigen::Matrix4d ghi;
    ghi.setZero();

    ghi.block<3,3>(0,0) = wHat;
    ghi.block<3,1>(0,3) = v;

    return ghi;
}

int getImuStartingIndexBaseOnCamera(std::vector<CameraInput> cameraReadVector,
 std::vector<ImuInputJetson> imuReadVector)
{
    int imuIndex = 0;
    int tempImuTime = imuReadVector.at(0).time;
    int cameraTimeStamp = cameraReadVector.at(0).time;

    while (tempImuTime < cameraTimeStamp)
    {
        imuIndex++;
        tempImuTime = imuReadVector.at(imuIndex).time;
    }

    return imuIndex;
}

Eigen::Quaterniond normalizeQuaternion(Eigen::Quaterniond quat)
{
    Eigen::Quaterniond temp(quat);

    if(temp.w() < 0)
    {
        temp.coeffs() *= -1;
    }
    return temp.normalized();
}

cv::Mat convertEigenMatToOpencvMat(Eigen::MatrixXd eigenMat)
{
    cv::Mat opencvMat = cv::Mat::zeros(eigenMat.rows(), eigenMat.cols(), CV_32F);

    for (int i = 0; i < eigenMat.rows(); i++)
    {
        for (int j = 0; j < eigenMat.cols(); j++)
        {
            opencvMat.at<float>(i, j) = eigenMat(i, j);
        }
    }

    return opencvMat;
}

Eigen::MatrixXd convertOpencvMatToEigenMat(cv::Mat cvMat)
{
    Eigen::MatrixXd eigenMat(cvMat.rows, cvMat.cols);

    for (int i = 0; i < cvMat.rows; i++)
    {
        for (int j = 0; j < cvMat.cols; j++)
        {
            eigenMat(i, j) = cvMat.at<float>(i,j);
        }
    }

    return eigenMat;
}

Eigen::Matrix<double, 4, 4> getGFromFrameMarkersData(FrameMarkersData frameMarkersData, int index)
{
    cv::Mat camRotMat;
    cv::Rodrigues(frameMarkersData.rvecs[index], camRotMat);

    std::cout << "camRotMat: " << camRotMat << std::endl << std::endl;

    Eigen::Matrix<double, 3, 3> camRot;
    camRot <<
    camRotMat.at<double>(0, 0), camRotMat.at<double>(0, 1), camRotMat.at<double>(0, 2),
    camRotMat.at<double>(1, 0), camRotMat.at<double>(1, 1), camRotMat.at<double>(1, 2),
    camRotMat.at<double>(2, 0), camRotMat.at<double>(2, 1), camRotMat.at<double>(2, 2);

    std::cout << "camRot: " << std::endl << camRot << std::endl << std::endl;

    Eigen::Vector3d camT{frameMarkersData.tvecs[index].val[0], frameMarkersData.tvecs[index].val[1], frameMarkersData.tvecs[index].val[2]};
    
    std::cout << "camT: " << std::endl << camT << std::endl << std::endl;

    Eigen::Matrix<double, 4, 4> g;
    g.setIdentity();

    g.block<3,3>(0,0) = camRot;
    g.block<3,1>(0,3) = camT;

    std::cout << "g: " << std::endl << g << std::endl << std::endl;
    
    return g;
}

Eigen::Vector3d getAngularVelocityFromTwoQuats(Eigen::Quaterniond q1, Eigen::Quaterniond q2, float deltaT)
{
    Eigen::Vector3d w;

    w.x() = 2/deltaT * (q1.w()*q2.x() - q1.x()*q2.w() - q1.y()*q2.z() + q1.z()*q2.y());
    w.y() = 2/deltaT * (q1.w()*q2.y() + q1.x()*q2.z() - q1.y()*q2.w() - q1.z()*q2.x());
    w.z() = 2/deltaT * (q1.w()*q2.z() - q1.x()*q2.y() + q1.y()*q2.x() - q1.z()*q2.w());

    return w;
}

void calculateHAndJacobian(
    cv::KalmanFilter KF,
    Eigen::Matrix<double, 4, 4> Gci,
    Eigen::Matrix<double, 4, 4> Gci_inv,
    Eigen::MatrixXd &h,
    Eigen::MatrixXd &H
    )
{
    float t1_gmc = KF.statePost.at<float>(0);
    float t2_gmc = KF.statePost.at<float>(1);
    float t3_gmc = KF.statePost.at<float>(2);

    float q0 = KF.statePost.at<float>(3);
    float q1 = KF.statePost.at<float>(4);
    float q2 = KF.statePost.at<float>(5);
    float q3 = KF.statePost.at<float>(6);

    float q0_2 = q0*q0;
    float q1_2 = q1*q1;
    float q2_2 = q2*q2;
    float q3_2 = q3*q3;

    float v1 = KF.statePost.at<float>(7);
    float v2 = KF.statePost.at<float>(8);
    float v3 = KF.statePost.at<float>(9);

    float w0 = KF.statePost.at<float>(10);
    float w1 = KF.statePost.at<float>(11);
    float w2 = KF.statePost.at<float>(12);

    float r00_gci_inv = Gci_inv(0,0);
    float r01_gci_inv = Gci_inv(0,1);
    float r02_gci_inv = Gci_inv(0,2);
    float r10_gci_inv = Gci_inv(1,0);
    float r11_gci_inv = Gci_inv(1,1);
    float r12_gci_inv = Gci_inv(1,2);
    float r20_gci_inv = Gci_inv(2,0);
    float r21_gci_inv = Gci_inv(2,1);
    float r22_gci_inv = Gci_inv(2,2);

    float t1_gci_inv = Gci_inv(0,3);
    float t2_gci_inv = Gci_inv(1,3);
    float t3_gci_inv = Gci_inv(2,3);

    float r00_gci = Gci(0,0);
    float r01_gci = Gci(0,1);
    float r02_gci = Gci(0,2);
    float r10_gci = Gci(1,0);
    float r11_gci = Gci(1,1);
    float r12_gci = Gci(1,2);
    float r20_gci = Gci(2,0);
    float r21_gci = Gci(2,1);
    float r22_gci = Gci(2,2);

    float t1_gci = Gci(0,3);
    float t2_gci = Gci(1,3);
    float t3_gci = Gci(2,3);

               
    float h0 = r20_gci_inv*(r21_gci*w1-r11_gci*w2)+r21_gci_inv*(r01_gci*w2-r21_gci*w0)+r22_gci_inv*(r11_gci*w0-r01_gci*w1);
    float h1 = r00_gci_inv*(r22_gci*w1-r12_gci*w2)+r01_gci_inv*(r02_gci*w2-r22_gci*w0)+r02_gci_inv*(r12_gci*w0-r02_gci*w1);
    float h2 = r10_gci_inv*(r20_gci*w1-r10_gci*w2)+r11_gci_inv*(r00_gci*w2-r20_gci*w0)+r12_gci_inv*(r10_gci*w0-r00_gci*w1);

    float temp10 = (2*(q3_2+q0_2)-1)*r22_gci+2*(q2*q3-q0*q1)*r21_gci+2*(q1*q3+q0*q2)*r20_gci+2*(q2*q3+q0*q1)*r12_gci+(2*(q2_2+q0_2)-1)*r11_gci+2*(q1*q2-q0*q3)*r10_gci+2*(q1*q3-q0*q2)*r02_gci+2*(q0*q3+q1*q2)*r01_gci+(2*(q1_2+q0_2)-1)*r00_gci+1;

    if(temp10 == 0)
    {
        temp10 = 0.001;
    }

    float h3 = sqrt(temp10)/2;    
    float h4 = (-2*(q2*q3-q0*q1)*r22_gci+(2*(q3_2+q0_2)-1)*r21_gci-(2*(q2_2+q0_2)-1)*r12_gci+2*(q2*q3+q0*q1)*r11_gci-2*(q0*q3+q1*q2)*r02_gci+2*(q1*q3-q0*q2)*r01_gci)/(2*sqrt(temp10));
    float h5 = (2*(q1*q3+q0*q2)*r22_gci-(2*(q3_2+q0_2)-1)*r20_gci+2*(q1*q2-q0*q3)*r12_gci-2*(q2*q3+q0*q1)*r10_gci+(2*(q1_2+q0_2)-1)*r02_gci-2*(q1*q3-q0*q2)*r00_gci)/(2*sqrt(temp10));
    float h6 = (-2*(q1*q3+q0*q2)*r21_gci+2*(q2*q3-q0*q1)*r20_gci-2*(q1*q2-q0*q3)*r11_gci+(2*(q2_2+q0_2)-1)*r10_gci-(2*(q1_2+q0_2)-1)*r01_gci+2*(q0*q3+q1*q2)*r00_gci)/(2*sqrt(temp10));
    float h7 = r00_gci_inv*(-t2_gci*w2+t3_gci*w1+v1)+r01_gci_inv*(t1_gci*w2-t3_gci*w0+v2)+r02_gci_inv*(-t1_gci*w1+t2_gci*w0+v3);
    float h8 = r10_gci_inv*(-t2_gci*w2+t3_gci*w1+v1)+r11_gci_inv*(t1_gci*w2-t3_gci*w0+v2)+r12_gci_inv*(-t1_gci*w1+t2_gci*w0+v3);
    float h9 = r20_gci_inv*(-t2_gci*w2+t3_gci*w1+v1)+r21_gci_inv*(t1_gci*w2-t3_gci*w0+v2)+r22_gci_inv*(-t1_gci*w1+t2_gci*w0+v3);
    float h10 = 2*(q1*q3+q0*q2)*t3_gci+2*(q1*q2-q0*q3)*t2_gci+t1_gmc+(2*(q1_2+q0_2)-1)*t1_gci;
    float h11 = 2*(q2*q3-q0*q1)*t3_gci+t2_gmc+(2*(q2_2+q0_2)-1)*t2_gci+2*(q0*q3+q1*q2)*t1_gci;
    float h12 = t3_gmc+(2*(q3_2+q0_2)-1)*t3_gci+2*(q2*q3+q0*q1)*t2_gci+2*(q1*q3-q0*q2)*t1_gci;

    h << h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;

    H(0,0) = r11_gci*r22_gci_inv-r21_gci*r21_gci_inv;
    H(1,0) = r02_gci_inv*r12_gci-r01_gci_inv*r22_gci;   
    H(2,0) = r10_gci*r12_gci_inv-r11_gci_inv*r20_gci;
    H(7,0) = r02_gci_inv*t2_gci-r01_gci_inv*t3_gci;
    H(8,0) = r12_gci_inv*t2_gci-r11_gci_inv*t3_gci;
    H(9,0) = r22_gci_inv*t2_gci-r21_gci_inv*t3_gci;

    H(0,1) = r20_gci_inv*r21_gci-r01_gci*r22_gci_inv;
    H(1,1) = r00_gci_inv*r22_gci-r02_gci*r02_gci_inv;   
    H(2,1) = r10_gci_inv*r20_gci-r00_gci*r12_gci_inv;
    H(7,1) = r00_gci_inv*t3_gci-r02_gci_inv*t1_gci;
    H(8,1) = r10_gci_inv*t3_gci-r12_gci_inv*t1_gci;
    H(9,1) = r20_gci_inv*t3_gci-r22_gci_inv*t1_gci;

    H(0,2) = r01_gci*r21_gci_inv-r11_gci*r20_gci_inv;
    H(1,2) = r01_gci_inv*r02_gci-r00_gci_inv*r12_gci; 
    H(2,2) = r00_gci*r11_gci_inv-r10_gci*r10_gci_inv;
    H(7,2) = r01_gci_inv*t1_gci-r00_gci_inv*t2_gci;
    H(8,2) = r11_gci_inv*t1_gci-r10_gci_inv*t2_gci;
    H(9,2) = r21_gci_inv*t1_gci-r20_gci_inv*t2_gci;

    H(3,3) = (4*q0*r22_gci-2*q1*r21_gci+2*q2*r20_gci+2*q1*r12_gci+4*q0*r11_gci-2*q3*r10_gci-2*q2*r02_gci+2*q3*r01_gci+4*q0*r00_gci)/(4*sqrt(temp10));
    H(4,3) = (2*q1*r22_gci+4*q0*r21_gci-4*q0*r12_gci+2*q1*r11_gci-2*q3*r02_gci-2*q2*r01_gci)/(2*sqrt(temp10))-((4*q0*r22_gci-2*q1*r21_gci+2*q2*r20_gci+2*q1*r12_gci+4*q0*r11_gci-2*q3*r10_gci-2*q2*r02_gci+2*q3*r01_gci+4*q0*r00_gci)*(-2*(q2*q3-q0*q1)*r22_gci+(2*(q3_2+q0_2)-1)*r21_gci-(2*(q2_2+q0_2)-1)*r12_gci+2*(q2*q3+q0*q1)*r11_gci-2*(q0*q3+q1*q2)*r02_gci+2*(q1*q3-q0*q2)*r01_gci))/(4*pow(temp10, 3/2));
    H(5,3) = (2*q2*r22_gci-4*q0*r20_gci-2*q3*r12_gci-2*q1*r10_gci+4*q0*r02_gci+2*q2*r00_gci)/(2*sqrt(temp10))-((4*q0*r22_gci-2*q1*r21_gci+2*q2*r20_gci+2*q1*r12_gci+4*q0*r11_gci-2*q3*r10_gci-2*q2*r02_gci+2*q3*r01_gci+4*q0*r00_gci)*(2*(q1*q3+q0*q2)*r22_gci-(2*(q3_2+q0_2)-1)*r20_gci+2*(q1*q2-q0*q3)*r12_gci-2*(q2*q3+q0*q1)*r10_gci+(2*(q1_2+q0_2)-1)*r02_gci-2*(q1*q3-q0*q2)*r00_gci))/(4*pow(temp10, 3/2));
    H(6,3) = (-2*q2*r21_gci-2*q1*r20_gci+2*q3*r11_gci+4*q0*r10_gci-4*q0*r01_gci+2*q3*r00_gci)/(2*sqrt(temp10))-((-2*(q1*q3+q0*q2)*r21_gci+2*(q2*q3-q0*q1)*r20_gci-2*(q1*q2-q0*q3)*r11_gci+(2*(q2_2+q0_2)-1)*r10_gci-(2*(q1_2+q0_2)-1)*r01_gci+2*(q0*q3+q1*q2)*r00_gci)*(4*q0*r22_gci-2*q1*r21_gci+2*q2*r20_gci+2*q1*r12_gci+4*q0*r11_gci-2*q3*r10_gci-2*q2*r02_gci+2*q3*r01_gci+4*q0*r00_gci))/(4*pow(temp10, 3/2));
    H(10,3) = 2*q2*t3_gci-2*q3*t2_gci+4*q0*t1_gci;
    H(11,3) = -2*q1*t3_gci+4*q0*t2_gci+2*q3*t1_gci;
    H(12,3) = 4*q0*t3_gci+2*q1*t2_gci-2*q2*t1_gci;


    H(3,4) = (-2*q0*r21_gci+2*q3*r20_gci+2*q0*r12_gci+2*q2*r10_gci+2*q3*r02_gci+2*q2*r01_gci+4*q1*r00_gci)/(4*sqrt(temp10));
    H(4,4) = (2*q0*r22_gci+2*q0*r11_gci-2*q2*r02_gci+2*q3*r01_gci)/(2*sqrt(temp10))-((-2*q0*r21_gci+2*q3*r20_gci+2*q0*r12_gci+2*q2*r10_gci+2*q3*r02_gci+2*q2*r01_gci+4*q1*r00_gci)*(-2*(q2*q3-q0*q1)*r22_gci+(2*(q3_2+q0_2)-1)*r21_gci-(2*(q2_2+q0_2)-1)*r12_gci+2*(q2*q3+q0*q1)*r11_gci-2*(q0*q3+q1*q2)*r02_gci+2*(q1*q3-q0*q2)*r01_gci))/(4*pow(temp10, 3/2));
    H(5,4) = (2*q3*r22_gci+2*q2*r12_gci-2*q0*r10_gci+4*q1*r02_gci-2*q3*r00_gci)/(2*sqrt(temp10))-((-2*q0*r21_gci+2*q3*r20_gci+2*q0*r12_gci+2*q2*r10_gci+2*q3*r02_gci+2*q2*r01_gci+4*q1*r00_gci)*(2*(q1*q3+q0*q2)*r22_gci-(2*(q3_2+q0_2)-1)*r20_gci+2*(q1*q2-q0*q3)*r12_gci-2*(q2*q3+q0*q1)*r10_gci+(2*(q1_2+q0_2)-1)*r02_gci-2*(q1*q3-q0*q2)*r00_gci))/(4*pow(temp10, 3/2));
    H(6,4) = (-2*q3*r21_gci-2*q0*r20_gci-2*q2*r11_gci-4*q1*r01_gci+2*q2*r00_gci)/(2*sqrt(temp10))-((-2*q0*r21_gci+2*q3*r20_gci+2*q0*r12_gci+2*q2*r10_gci+2*q3*r02_gci+2*q2*r01_gci+4*q1*r00_gci)*(-2*(q1*q3+q0*q2)*r21_gci+2*(q2*q3-q0*q1)*r20_gci-2*(q1*q2-q0*q3)*r11_gci+(2*(q2_2+q0_2)-1)*r10_gci-(2*(q1_2+q0_2)-1)*r01_gci+2*(q0*q3+q1*q2)*r00_gci))/(4*pow(temp10, 3/2));
    H(10,4) = 2*q3*t3_gci+2*q2*t2_gci+4*q1*t1_gci;
    H(11,4) = 2*q2*t1_gci-2*q0*t3_gci;
    H(12,4) = 2*q0*t2_gci+2*q3*t1_gci;

    H(3,5) =(2*q3*r21_gci+2*q0*r20_gci+2*q3*r12_gci+4*q2*r11_gci+2*q1*r10_gci-2*q0*r02_gci+2*q1*r01_gci)/(4*sqrt(temp10));
    H(4,5) =(-2*q3*r22_gci-4*q2*r12_gci+2*q3*r11_gci-2*q1*r02_gci-2*q0*r01_gci)/(2*sqrt(temp10))-((2*q3*r21_gci+2*q0*r20_gci+2*q3*r12_gci+4*q2*r11_gci+2*q1*r10_gci-2*q0*r02_gci+2*q1*r01_gci)*(-2*(q2*q3-q0*q1)*r22_gci+(2*(q3_2+q0_2)-1)*r21_gci-(2*(q2_2+q0_2)-1)*r12_gci+2*(q2*q3+q0*q1)*r11_gci-2*(q0*q3+q1*q2)*r02_gci+2*(q1*q3-q0*q2)*r01_gci))/(4*pow(temp10, 3/2));
    H(5,5) =(2*q0*r22_gci+2*q1*r12_gci-2*q3*r10_gci+2*q0*r00_gci)/(2*sqrt(temp10))-((2*q3*r21_gci+2*q0*r20_gci+2*q3*r12_gci+4*q2*r11_gci+2*q1*r10_gci-2*q0*r02_gci+2*q1*r01_gci)*(2*(q1*q3+q0*q2)*r22_gci-(2*(q3_2+q0_2)-1)*r20_gci+2*(q1*q2-q0*q3)*r12_gci-2*(q2*q3+q0*q1)*r10_gci+(2*(q1_2+q0_2)-1)*r02_gci-2*(q1*q3-q0*q2)*r00_gci))/(4*pow(temp10, 3/2));
    H(6,5) =(-2*q0*r21_gci+2*q3*r20_gci-2*q1*r11_gci+4*q2*r10_gci+2*q1*r00_gci)/(2*sqrt(temp10))-((2*q3*r21_gci+2*q0*r20_gci+2*q3*r12_gci+4*q2*r11_gci+2*q1*r10_gci-2*q0*r02_gci+2*q1*r01_gci)*(-2*(q1*q3+q0*q2)*r21_gci+2*(q2*q3-q0*q1)*r20_gci-2*(q1*q2-q0*q3)*r11_gci+(2*(q2_2+q0_2)-1)*r10_gci-(2*(q1_2+q0_2)-1)*r01_gci+2*(q0*q3+q1*q2)*r00_gci))/(4*pow(temp10, 3/2));
    H(10,5) =2*q0*t3_gci+2*q1*t2_gci;
    H(11,5) =2*q3*t3_gci+4*q2*t2_gci+2*q1*t1_gci;
    H(12,5) =2*q3*t2_gci-2*q0*t1_gci;

    H(3,6) = (4*q3*r22_gci+2*q2*r21_gci+2*q1*r20_gci+2*q2*r12_gci-2*q0*r10_gci+2*q1*r02_gci+2*q0*r01_gci)/(4*sqrt(temp10));
    H(4,6) = (-2*q2*r22_gci+4*q3*r21_gci+2*q2*r11_gci-2*q0*r02_gci+2*q1*r01_gci)/(2*sqrt(temp10))-((4*q3*r22_gci+2*q2*r21_gci+2*q1*r20_gci+2*q2*r12_gci-2*q0*r10_gci+2*q1*r02_gci+2*q0*r01_gci)*(-2*(q2*q3-q0*q1)*r22_gci+(2*(q3_2+q0_2)-1)*r21_gci-(2*(q2_2+q0_2)-1)*r12_gci+2*(q2*q3+q0*q1)*r11_gci-2*(q0*q3+q1*q2)*r02_gci+2*(q1*q3-q0*q2)*r01_gci))/(4*pow(temp10, 3/2));
    H(5,6) = (2*q1*r22_gci-4*q3*r20_gci-2*q0*r12_gci-2*q2*r10_gci-2*q1*r00_gci)/(2*sqrt(temp10))-((4*q3*r22_gci+2*q2*r21_gci+2*q1*r20_gci+2*q2*r12_gci-2*q0*r10_gci+2*q1*r02_gci+2*q0*r01_gci)*(2*(q1*q3+q0*q2)*r22_gci-(2*(q3_2+q0_2)-1)*r20_gci+2*(q1*q2-q0*q3)*r12_gci-2*(q2*q3+q0*q1)*r10_gci+(2*(q1_2+q0_2)-1)*r02_gci-2*(q1*q3-q0*q2)*r00_gci))/(4*pow(temp10, 3/2));
    H(6,6) = (-2*q1*r21_gci+2*q2*r20_gci+2*q0*r11_gci+2*q0*r00_gci)/(2*sqrt(temp10))-((-2*(q1*q3+q0*q2)*r21_gci+2*(q2*q3-q0*q1)*r20_gci-2*(q1*q2-q0*q3)*r11_gci+(2*(q2_2+q0_2)-1)*r10_gci-(2*(q1_2+q0_2)-1)*r01_gci+2*(q0*q3+q1*q2)*r00_gci)*(4*q3*r22_gci+2*q2*r21_gci+2*q1*r20_gci+2*q2*r12_gci-2*q0*r10_gci+2*q1*r02_gci+2*q0*r01_gci))/(4*pow(temp10, 3/2));
    H(10,6) = 2*q1*t3_gci-2*q0*t2_gci;
    H(11,6) = 2*q2*t3_gci+2*q0*t1_gci;
    H(12,6) = 4*q3*t3_gci+2*q2*t2_gci+2*q1*t1_gci;

    H(7,7) = r00_gci_inv;
    H(8,7) = r10_gci_inv;
    H(9,7) = r20_gci_inv;

    H(7,8) = r01_gci_inv;
    H(8,8) = r11_gci_inv;
    H(9,8) = r21_gci_inv;

    H(7,9) = r02_gci_inv;
    H(8,9) = r12_gci_inv;
    H(9,9) = r22_gci_inv;
}

Eigen::Matrix4d invertG(Eigen::Matrix4d G)
{
    Eigen::Matrix4d invG;
    invG.setIdentity();
    invG.block<3,3>(0,0) = G.block<3,3>(0,0).transpose();
    invG.block<3,1>(0,3) = -G.block<3,3>(0,0).transpose() * G.block<3,1>(0,3);

    return invG;
}

void fixStateQuaternion(cv::KalmanFilter &KF, std::string stateName)
{
    if(stateName == "pre")
    {
        Eigen::Quaterniond q(KF.statePre.at<float>(3), KF.statePre.at<float>(4), KF.statePre.at<float>(5), KF.statePre.at<float>(6));
        fixQuatEigen(q);
        
        KF.statePre.at<float>(3) = q.w();
        KF.statePre.at<float>(4) = q.x();
        KF.statePre.at<float>(5) = q.y();
        KF.statePre.at<float>(6) = q.z();
    }
    else if (stateName == "post")
    {
        Eigen::Quaterniond q(KF.statePost.at<float>(3), KF.statePost.at<float>(4), KF.statePost.at<float>(5), KF.statePost.at<float>(6));
        fixQuatEigen(q);

        KF.statePost.at<float>(3) = q.w();
        KF.statePost.at<float>(4) = q.x();
        KF.statePost.at<float>(5) = q.y();
        KF.statePost.at<float>(6) = q.z();
    }
}

void fixQuatEigen(Eigen::Quaterniond &q)
{
    q.normalize();

    if (q.w() < 0)
        q.coeffs() *= -1;
}

int getBaseMarkerIndex(std::vector<int> markerIds, int baseMarkerId)
{
    int baseMarkerIndex = -1;
    for(size_t i = 0; i < markerIds.size(); i++)
    {
        if(markerIds[i] == baseMarkerId)
        {
            baseMarkerIndex = i;
            break;
        }
    }
    return baseMarkerIndex;
}

void getAllTransformsBetweenMarkers(
    FrameMarkersData firstFrameMarkersData,
    Eigen::Matrix4d Gcm,
    int indexBaseMarker,
    std::map<int, Eigen::Matrix4d> &oldCamMeasurementsMap,
    bool clearFile)
{
    for(size_t i = 0; i < firstFrameMarkersData.markerIds.size(); i++)
    {
        Eigen::Matrix4d gCamToMarker = getGFromFrameMarkersData(firstFrameMarkersData, i); 
        std::cout << "Gcm: " << std::endl << Gcm << std::endl << std::endl;
        std::cout << "gCamToMarker: " << std::endl << gCamToMarker << std::endl << std::endl;
        
        oldCamMeasurementsMap[firstFrameMarkersData.markerIds[i]] =  invertG(Gcm) * gCamToMarker;
        std::cout << "Transform: " << std::endl << oldCamMeasurementsMap[firstFrameMarkersData.markerIds[i]] << std::endl << std::endl;

        transformWrite(oldCamMeasurementsMap[firstFrameMarkersData.markerIds[i]], firstFrameMarkersData.markerIds[i], clearFile);
        
    }
}

///////////////////////////////////////////////// Functions that ar NOT used /////////////////////////////////////////////////////////
///////////////////////////////////////////////// Functions that ar NOT used /////////////////////////////////////////////////////////
///////////////////////////////////////////////// Functions that ar NOT used /////////////////////////////////////////////////////////
///////////////////////////////////////////////// Functions that ar NOT used /////////////////////////////////////////////////////////

// Convert quaternion to rotation vector.
Eigen::Vector3d QuatToRotVectEigen(Eigen::Quaterniond quaternion)
{
    Eigen::Vector3d rotVect;

    float q0 = quaternion.w();
    float q1 = quaternion.x();
    float q2 = quaternion.y();
    float q3 = quaternion.z();

    rotVect[0] = atan2(2 * (q1*q2 + q0*q3), q0*q0 + q1*q1 - q2*q2 - q3*q3);
    rotVect[1] = asin(-2 * (q1*q3 - q0*q2));
    rotVect[2] = atan2(2 * (q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3);

    return rotVect;
}

// Convert rotation vector to quaternion.
Eigen::Quaterniond convertOpencvRotVectToQuat(cv::Vec3d rotVect)
{
    float vecNorm = cv::norm(rotVect);
    float w = cos(vecNorm / 2);

    cv::Vec3d xyz = sin(vecNorm / 2) * rotVect / vecNorm;

    Eigen::Quaterniond quaternion{w, xyz[0], xyz[1], xyz[2]};

    return quaternion;
}

// Convert quaternion to rotation vector.
cv::Vec3d QuatToRotVect(Eigen::Quaterniond quaternion)
{
    cv::Vec3d rotVect;

    float w = quaternion.w();
    float x = quaternion.x();
    float y = quaternion.y();
    float z = quaternion.z();

    float vecNorm = 2 * acos(w);

    rotVect[0] = x * vecNorm / sin(vecNorm / 2);
    rotVect[1] = y * vecNorm / sin(vecNorm / 2);
    rotVect[2] = z * vecNorm / sin(vecNorm / 2);

    return rotVect;
}

Eigen::Matrix3d normalizeRotationMatrix(Eigen::Matrix3d matrix)
{
    Eigen::Quaterniond quat(matrix);

    return quat.normalized().toRotationMatrix();
}

Eigen::Matrix3d GramSchmidt(Eigen::Matrix3d rotationMatrix)
{
    Eigen::Vector3d v1 = rotationMatrix.block<3,1>(0,0);
    Eigen::Vector3d v2 = rotationMatrix.block<3,1>(0,1);
    Eigen::Vector3d v3 = rotationMatrix.block<3,1>(0,2);

    Eigen::Vector3d u1 = v1;
    Eigen::Vector3d u2 = v2 - proj(u1, v2);
    Eigen::Vector3d u3 = v3 - proj(u1, v3) - proj(u2, v3);

    Eigen::Matrix3d M;
    M << u1/u1.norm(), u2/u2.norm(), u3/u3.norm();

    return M;
}

Eigen::Vector3d proj(Eigen::Vector3d u, Eigen::Vector3d v)
{
    return (u * u.dot(v)) / u.dot(u);
}

Eigen::Matrix3d matrixExp(Eigen::Vector3d gyroTimesDeltaT)
{
    float norm = gyroTimesDeltaT.norm();
    float normInv = 1 / norm;
    Eigen::Matrix3d wHat = getWHat(gyroTimesDeltaT);

    Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();

    if (norm < 1e-3)
    {
        return I3x3 + wHat;
    }        
    else
    {
        Eigen::Matrix3d wHat2 = wHat * wHat;
        float normInv2 = normInv * normInv;

        return I3x3 + sin(norm) * normInv * wHat + (1 - cos(norm)) * normInv2 * wHat2;
    }
}

void normalizeDataSet(
    std::vector<Eigen::Vector3d> points,
    std::vector<float> &result,
    int variable)
{
    float max = -1000000;
    float min = 1000000;

    for (size_t i = 0; i < points.size(); i++)
    {
        if (points.at(i)[variable] > max)
        {
            max = points.at(i)[variable];
        }
        else if (points.at(i)[variable] < min)
        {
            min = points.at(i)[variable];
        }
    }

    for (size_t i = 0; i < points.size(); i++)
    {
        result.push_back((points.at(i)[variable] - min) / (max - min));
    }
}

Eigen::Matrix<double, 3, 3> getCamRotMatFromRotVec(cv::Vec3d camRvec)
{
    cv::Mat camRotMat;
    cv::Rodrigues(camRvec, camRotMat);
    Eigen::Matrix<double, 3, 3> camRot;
    camRot <<
    camRotMat.at<float>(0, 0), camRotMat.at<float>(0, 1), camRotMat.at<float>(0, 2),
    camRotMat.at<float>(1, 0), camRotMat.at<float>(1, 1), camRotMat.at<float>(1, 2),
    camRotMat.at<float>(2, 0), camRotMat.at<float>(2, 1), camRotMat.at<float>(2, 2);

    return camRot;
}

// convert euler angles to quaternion
Eigen::Quaterniond rotVecToQuat(Eigen::Vector3d euler)
{
    float angle = euler.norm();
    float sinA = std::sin(angle / 2);
    float cosA = std::cos(angle / 2);

    Eigen::Quaterniond q;
    q.x() = euler.x() * sinA;
    q.y() = euler.y() * sinA;
    q.z() = euler.z() * sinA;
    q.w() = cosA;

    return q.normalized();
}

// Create a hard copy of camera vector.
std::vector<CameraInput> hardCopyCameraVector(
    std::vector<CameraInput> cameraReadVector)
{
    std::vector<CameraInput> cameraReadVectorCopy;

    std::vector<CameraInput>::iterator it = cameraReadVector.begin();
    CameraInput tempCamera;

    for (; it != cameraReadVector.end(); it++)
    {
        tempCamera = *it;
        cameraReadVectorCopy.push_back(tempCamera);
    }

    return cameraReadVectorCopy;
}

std::vector<FrameMarkersData> getRotationTraslationFromAllFrames(
    std::vector<CameraInput> cameraReadVector,
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs,
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams)
{
    std::vector<FrameMarkersData> frameMarkersDataVector;

    for (size_t i = 0; i < cameraReadVector.size(); i++)
    {
        FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(
            cameraReadVector[i],
            dictionary,
            cameraMatrix,
            distCoeffs,
            detectorParams);

        frameMarkersDataVector.push_back(frameMarkersData);
    }

    return frameMarkersDataVector;
}

void gnuPrintImuPreintegration(
    FILE *output,
    std::vector<Eigen::Vector3d> vectorOfPointsOne,
    std::vector<Eigen::Vector3d> vectorOfPointsTwo,
    std::vector<Eigen::Vector3d> vectorOfMarkers)
{
    fprintf(output, "set title \"IMU Preintegration\"\n");
    fprintf(output, "set xlabel \"x\"\n");
    fprintf(output, "set ylabel \"y\"\n");
    fprintf(output, "set zlabel \"z\"\n");
    fprintf(output, "set ticslevel 3.\n");
    //fprintf(output, "set xrange [-1.0:1.0]\n");
    //fprintf(output, "set yrange [-1.0:1.0]\n");
    //fprintf(output, "set zrange [0.0:0.05]\n");

    fprintf(output, "splot '-' with points pointtype 7 ps 1 lc rgb 'blue' title 'Z', '-' with points pointtype 7 ps 1 lc rgb 'red' title 'X', '-' with points pointtype 7 ps 1 lc rgb 'black' title 'Marker'\n");
    
    Eigen::Vector3d tempPoint;

    for (size_t i = 0; i < vectorOfPointsOne.size(); i++)
    {
        tempPoint = vectorOfPointsOne.at(i);
        fprintf(output, "%g %g %g\n", tempPoint[0], tempPoint[1], tempPoint[2]);
    }
    fflush(output);
    fprintf(output, "e\n");
    
    for (size_t i = 0; i < vectorOfPointsTwo.size(); i++)
    {
        tempPoint = vectorOfPointsTwo.at(i);
        fprintf(output, "%g %g %g\n", tempPoint[0], tempPoint[1], tempPoint[2]);
    }
    fflush(output);
    fprintf(output, "e\n");

    for (size_t i = 0; i < vectorOfMarkers.size(); i++)
    {
        tempPoint = vectorOfMarkers.at(i);
        fprintf(output, "%g %g %g\n", tempPoint[0], tempPoint[1], tempPoint[2]);
    }
    fflush(output);
    fprintf(output, "e\n");
    
    usleep(1000000/5);
}

void applyIIRFilterToAccAndGyro(
    Eigen::Vector3d accReading,
    Eigen::Vector3d gyroReading,
    Eigen::Vector3d &accFiltered,
    Eigen::Vector3d &gyroFiltered)
{
    accFiltered = ALPHA_ACC * accFiltered + (1 - ALPHA_ACC) * accReading;
    gyroFiltered = ALPHA_GYRO * gyroFiltered + (1 - ALPHA_GYRO) * gyroReading;
}

void calculateBiasAccAndGyro(Eigen::Vector3d &accBiasVect, Eigen::Vector3d &gyroBiasVect)
{
    std::vector<ImuInputJetson> imuReadVector = readDataIMUJetson();

    Eigen::Vector3d gyro;
    Eigen::Vector3d acc;

    gyro.setZero();
    acc.setZero();

    for(size_t i = 0; i < imuReadVector.size(); i++)
    {
        ImuInputJetson tempImuData = imuReadVector.at(i);

        gyro = Eigen::Vector3d{tempImuData.gyroVect.x(), tempImuData.gyroVect.y(), tempImuData.gyroVect.z()};
        acc = Eigen::Vector3d{tempImuData.accVect.x(), tempImuData.accVect.y(), tempImuData.accVect.z()};
        
        accBiasVect += acc;
        gyroBiasVect += gyro;
    }

    accBiasVect /= imuReadVector.size();
    gyroBiasVect /= imuReadVector.size();
}

Eigen::Vector3d multiplyVectorByG(Eigen::Matrix4d G, Eigen::Vector3d v)
{
    Eigen::Vector4d v4;
    v4 << v, 1;
    v4 = G * v4;

    return v4.block<3,1>(0,0);
}
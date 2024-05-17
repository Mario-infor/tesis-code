#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <utils.h>
#include <iostream>
#include <BNO055-BBB_driver.h>
#include <curses.h>
#include <iterator>

// Convert rotation vector to quaternion.
glm::quat convertOpencvRotVectToQuat(cv::Vec3d rotVect)
{
    float vecNorm = cv::norm(rotVect);
    float w = cos(vecNorm / 2);

    cv::Vec3d xyz = sin(vecNorm / 2) * rotVect / vecNorm;

    glm::quat quaternion = glm::quat(w, xyz[0], xyz[1], xyz[2]);

    return quaternion;
}

// Convert quaternion to rotation vector.
cv::Vec3d QuatToRotVect(glm::quat quaternion)
{
    cv::Vec3d rotVect;

    float w = quaternion.w;
    float x = quaternion.x;
    float y = quaternion.y;
    float z = quaternion.z;

    float vecNorm = 2 * acos(w);

    rotVect[0] = x * vecNorm / sin(vecNorm / 2);
    rotVect[1] = y * vecNorm / sin(vecNorm / 2);
    rotVect[2] = z * vecNorm / sin(vecNorm / 2);

    return rotVect;
}

// Convert quaternion to rotation vector.
Eigen::Vector3d QuatToRotVectEigen(Eigen::Quaterniond quaternion)
{
    Eigen::Vector3d rotVect;

    float q0 = quaternion.w();
    float q1 = quaternion.x();
    float q2 = quaternion.y();
    float q3 = quaternion.z();

    /*float vecNorm = 2 * acos(w);

    rotVect[0] = x * vecNorm / sin(vecNorm / 2);
    rotVect[1] = y * vecNorm / sin(vecNorm / 2);
    rotVect[2] = z * vecNorm / sin(vecNorm / 2);*/

    rotVect[0] = atan2(2 * (q1*q2 + q0*q3), q0*q0 + q1*q1 - q2*q2 - q3*q3);
    rotVect[1] = asin(-2 * (q1*q3 - q0*q2));
    rotVect[2] = atan2(2 * (q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3);

    return rotVect;
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

void printIMUData()
{
    char filename[] = IMU_ADDRESS;
    BNO055 sensors;
    sensors.openDevice(filename);

    WINDOW *win;
    char buff[512];

    win = initscr();
    clearok(win, TRUE);

    float maxX = 0.0;
    float maxY = 0.0;
    float maxZ = 0.0;

    while (true)
    {
        sensors.readAll();
        wmove(win, 5, 2);

        float tempMaxX = sensors.accelVect.vi[0] * sensors.Scale;
        float tempMaxY = sensors.accelVect.vi[1] * sensors.Scale;
        float tempMaxZ = sensors.accelVect.vi[2] * sensors.Scale;

        if (tempMaxX > maxX)
            maxX = tempMaxX;

        if (tempMaxY > maxY)
            maxY = tempMaxY;

        if (tempMaxZ > maxZ)
            maxZ = tempMaxZ;

        // snprintf(buff, 511, "Acc = {X=%f, Y=%f, Z=%f}", maxX, maxY, maxZ);

        snprintf(buff, 79, "EULER=[%07.5lf, %07.5lf, %07.3lf]",
                 sensors.eOrientation.vi[0] * sensors.Scale * MATH_RAD_TO_DEGREE,
                 sensors.eOrientation.vi[1] * sensors.Scale * MATH_RAD_TO_DEGREE,
                 sensors.eOrientation.vi[2] * sensors.Scale * MATH_RAD_TO_DEGREE);

        waddstr(win, buff);

        wrefresh(win);
        wclear(win);
    }
    endwin();
}

std::vector<FrameMarkersData> getRotationTraslationFromAllFrames(
    std::vector<CameraInput> cameraReadVector,
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs)
{
    std::vector<FrameMarkersData> frameMarkersDataVector;

    for (size_t i = 0; i < cameraReadVector.size(); i++)
    {
        FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(cameraReadVector[i],
                                                                           dictionary, cameraMatrix, distCoeffs);
        frameMarkersDataVector.push_back(frameMarkersData);
    }

    return frameMarkersDataVector;
}

FrameMarkersData getRotationTraslationFromFrame(
    CameraInput frame,
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs)
{
    FrameMarkersData frameMarkersData;

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(frame.frame, dictionary, markerCorners, markerIds);

    if (markerIds.size() > 0)
    {
        cv::aruco::drawDetectedMarkers(frame.frame, markerCorners, markerIds);

        std::vector<cv::Vec3d> rvecs, tvecs;

        cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

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
        cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
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

int getImuStartingIdexBaseOnCamera(std::vector<CameraInput> cameraReadVector,
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

void gnuPrintImuPreintegration(
    FILE *output,
    std::vector<Eigen::Vector3d> vectorOfPointsOne,
    std::vector<Eigen::Vector3d> vectorOfPointsTwo)
{
    fprintf(output, "set title \"IMU Preintegration\"\n");
    fprintf(output, "set xlabel \"x\"\n");
    fprintf(output, "set ylabel \"y\"\n");
    fprintf(output, "set zlabel \"z\"\n");
    fprintf(output, "set ticslevel 3.\n");
    //fprintf(output, "set xrange [-0.5:0.5]\n");
    //fprintf(output, "set yrange [-0.5:0.5]\n");

    fprintf(output, "splot '-' with points pointtype 7 ps 1 lc rgb 'blue' title 'Original', '-' with points pointtype 7 ps 1 lc rgb 'red' title 'Prediction'\n");
    
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
    
    usleep(100000);
}

void gnuPrintImuCompareValues(
    FILE *output,
    std::vector<float> vectorOfPointsOne,
    std::vector<float> vectorOfPointsTwo)
{
    fprintf(output, "set title \"IMU Data Comparisson\"\n");
    fprintf(output, "set xlabel \"x\"\n");
    fprintf(output, "set ylabel \"y\"\n");
    fprintf(output, "set ticslevel 3.\n");

    fprintf(output, "plot '-' with points pointtype 7 ps 1 lc rgb 'blue' title 'Original', '-' with points pointtype 7 ps 1 lc rgb 'red' title 'Prediction'\n");
    
    float tempPoint;

    for (size_t i = 0; i < vectorOfPointsOne.size(); i++)
    {
        tempPoint = vectorOfPointsOne[i];
        fprintf(output, "%g %g\n", (double)i, tempPoint);
    }
    fflush(output);
    fprintf(output, "e\n");
    
    for (size_t i = 0; i < vectorOfPointsTwo.size(); i++)
    {
        tempPoint = vectorOfPointsTwo[i];
        fprintf(output, "%g %g\n", (double)i, tempPoint);
    }
    fflush(output);
    fprintf(output, "e\n");
    
    //usleep(500000);
}

Eigen::Matrix3d normalizeRotationMatrix(Eigen::Matrix3d matrix)
{
    Eigen::Quaterniond quat(matrix);

    return quat.normalized().toRotationMatrix();
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

Eigen::Matrix<double, 4, 4> getGFromFrameMarkersData(FrameMarkersData frameMarkersData)
{
    cv::Mat camRotMat;
    cv::Rodrigues(frameMarkersData.rvecs[0], camRotMat);
    Eigen::Matrix<double, 3, 3> camRot;
    camRot <<
    camRotMat.at<float>(0, 0), camRotMat.at<float>(0, 1), camRotMat.at<float>(0, 2),
    camRotMat.at<float>(1, 0), camRotMat.at<float>(1, 1), camRotMat.at<float>(1, 2),
    camRotMat.at<float>(2, 0), camRotMat.at<float>(2, 1), camRotMat.at<float>(2, 2);

    Eigen::Vector3d camT{frameMarkersData.tvecs[0].val[0], frameMarkersData.tvecs[0].val[1], frameMarkersData.tvecs[0].val[2]};
    
    Eigen::Matrix<double, 4, 4> g;
    g.setIdentity();

    g.block<3,3>(0,0) = camRot;
    g.block<3,1>(0,3) = camT;
    
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
    Eigen::Matrix<double, 4, 4> Gti,
    Eigen::Matrix<double, 4, 4> Gci,
    Eigen::Matrix<double, 4, 4> Gni,
    Eigen::Matrix<double, 13, 1> &h,
    Eigen::Matrix<double, 13, 13> &H
    )
{
    Eigen::Matrix<double, 4, 4> Gti_inv = Gti.inverse();
    Eigen::Matrix<double, 4, 4> Gni_inv = Gni.inverse();

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

    float w0 = KF.statePost.at<float>(14);
    float w1 = KF.statePost.at<float>(15);
    float w2 = KF.statePost.at<float>(16);

    float r00_gni = Gni(0,0);
    float r01_gni = Gni(0,1);
    float r02_gni = Gni(0,2);
    float r10_gni = Gni(1,0);
    float r11_gni = Gni(1,1);
    float r12_gni = Gni(1,2);
    float r20_gni = Gni(2,0);
    float r21_gni = Gni(2,1);
    float r22_gni = Gni(2,2);

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

    float r00_gni_inv = Gni_inv(0,0);
    float r01_gni_inv = Gni_inv(0,1);
    float r02_gni_inv = Gni_inv(0,2);
    float r10_gni_inv = Gni_inv(1,0);
    float r11_gni_inv = Gni_inv(1,1);
    float r12_gni_inv = Gni_inv(1,2);
    float r20_gni_inv = Gni_inv(2,0);
    float r21_gni_inv = Gni_inv(2,1);
    float r22_gni_inv = Gni_inv(2,2);

    float t1_gni_inv = Gni_inv(0,3);
    float t2_gni_inv = Gni_inv(1,3);
    float t3_gni_inv = Gni_inv(2,3);

    float r00_gti_inv = Gti_inv(0,0);
    float r01_gti_inv = Gti_inv(0,1);
    float r02_gti_inv = Gti_inv(0,2);
    float r10_gti_inv = Gti_inv(1,0);
    float r11_gti_inv = Gti_inv(1,1);
    float r12_gti_inv = Gti_inv(1,2);
    float r20_gti_inv = Gti_inv(2,0);
    float r21_gti_inv = Gti_inv(2,1);
    float r22_gti_inv = Gti_inv(2,2);

    float t1_gti_inv = Gti_inv(0,3);
    float t2_gti_inv = Gti_inv(1,3);
    float t3_gti_inv = Gti_inv(2,3);

    float temp16 = 2*(q2_2+q0_2)-1;
    float temp15 = q0*q2;
    float temp14 = q1*q3;
    float temp13 = q0*q1;
    float temp12 = q2*q3;
    float temp23 = q1*q2;
    float temp24 = q0*q3;

    float temp11 = 2*(q3_2+q0_2)-1;
    float temp4 = r22_gti_inv+r02_gti_inv;
    float temp20 = temp12-temp13;
    float temp21 = temp14+temp15;
    float temp22 = temp12+temp13;
    float temp25 = temp23-temp24;
    float temp26 = temp14-temp15;
    float temp27 = temp24+temp23;
    float temp28 = 2*(q1_2+q0_2)-1;

    float h0 = r20_gni*(r21_gni_inv*w1-r11_gni_inv*w2)+r21_gni*(r01_gni_inv*w2-r21_gni_inv*w0)+r22_gni*(r11_gni_inv*w0-r01_gni_inv*w1);
    float h1 = r00_gni*(r22_gni_inv*w1-r12_gni_inv*w2)+r01_gni*(r02_gni_inv*w2-r22_gni_inv*w0)+r02_gni*(r12_gni_inv*w0-r02_gni_inv*w1);
    float h2 = r10_gni*(r20_gni_inv*w1-r10_gni_inv*w2)+r11_gni*(r00_gni_inv*w2-r20_gni_inv*w0)+r12_gni*(r10_gni_inv*w0-r00_gni_inv*w1);
    float h3 = sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1)/2;
    float h4 = ((2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)*r22_gti_inv-r12_gti_inv*((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)+(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)*r21_gti_inv+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r20_gti_inv-r11_gti_inv*((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)-((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r10_gti_inv)/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1));
    float h5 = (-(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)*temp4*((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)-(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)*r21_gti_inv-(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)*r20_gti_inv+r01_gti_inv*((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)+r00_gti_inv*((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci))/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1));
    float h6 = (-r02_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r12_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)-r01_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r11_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)*r10_gti_inv-r00_gti_inv*(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci))/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1));
    float h7 = r00_gni*(-t2_gni_inv*w2+t3_gni_inv*w1+v1)+r01_gni*(t1_gni_inv*w2-t3_gni_inv*w0+v2)+r02_gni*(-t1_gni_inv*w1+t2_gni_inv*w0+v3);
    float h8 = r10_gni*(-t2_gni_inv*w2+t3_gni_inv*w1+v1)+r11_gni*(t1_gni_inv*w2-t3_gni_inv*w0+v2)+r12_gni*(-t1_gni_inv*w1+t2_gni_inv*w0+v3);
    float h9 = r20_gni*(-t2_gni_inv*w2+t3_gni_inv*w1+v1)+r21_gni*(t1_gni_inv*w2-t3_gni_inv*w0+v2)+r22_gni*(-t1_gni_inv*w1+t2_gni_inv*w0+v3);
    float h10 = r02_gti_inv*(r22_gci*t3_gmc+t3_gci+r21_gci*t2_gmc+r20_gci*t1_gmc)+r01_gti_inv*(r12_gci*t3_gmc+r11_gci*t2_gmc+t2_gci+r10_gci*t1_gmc)+r00_gti_inv*(r02_gci*t3_gmc+r01_gci*t2_gmc+r00_gci*t1_gmc+t1_gci)+t1_gti_inv;
    float h11 = r12_gti_inv*(r22_gci*t3_gmc+t3_gci+r21_gci*t2_gmc+r20_gci*t1_gmc)+r11_gti_inv*(r12_gci*t3_gmc+r11_gci*t2_gmc+t2_gci+r10_gci*t1_gmc)+r10_gti_inv*(r02_gci*t3_gmc+r01_gci*t2_gmc+r00_gci*t1_gmc+t1_gci)+t2_gti_inv;
    float h12 = t3_gti_inv+r22_gti_inv*(r22_gci*t3_gmc+t3_gci+r21_gci*t2_gmc+r20_gci*t1_gmc)+r21_gti_inv*(r12_gci*t3_gmc+r11_gci*t2_gmc+t2_gci+r10_gci*t1_gmc)+r20_gti_inv*(r02_gci*t3_gmc+r01_gci*t2_gmc+r00_gci*t1_gmc+t1_gci);

    h << h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;

    H(10,0) = r02_gti_inv*r20_gci+r01_gti_inv*r10_gci+r00_gci*r00_gti_inv;
    H(11,0) = r12_gti_inv*r20_gci+r10_gci*r11_gti_inv+r00_gci*r10_gti_inv;   
    H(12,0) = r20_gci*r22_gti_inv+r10_gci*r21_gti_inv+r00_gci*r20_gti_inv;

    H(10,1) = r02_gti_inv*r21_gci+r01_gti_inv*r11_gci+r00_gti_inv*r01_gci;
    H(11,1) = r12_gti_inv*r21_gci+r11_gci*r11_gti_inv+r01_gci*r10_gti_inv;   
    H(12,1) = r21_gci*r22_gti_inv+r11_gci*r21_gti_inv+r01_gci*r20_gti_inv;

    H(10,2) = r02_gti_inv*r22_gci+r01_gti_inv*r12_gci+r00_gti_inv*r02_gci;
    H(11,2) = r12_gti_inv*r22_gci+r11_gti_inv*r12_gci+r02_gci*r10_gti_inv;   
    H(12,2) = r22_gci*r22_gti_inv+r12_gci*r21_gti_inv+r02_gci*r20_gti_inv;
    
    float temp1 = 4*std::pow(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1, 3/2);
    float temp2 = 4*q0*r22_gci-2*q1*r21_gci+2*q2*r20_gci;
    float temp17 = 2*q2*r22_gci;
    float temp18 = 2*q3*r21_gci;
    float temp19 = q0*r20_gci;
    float temp3 = -temp17+temp18+4*temp19;
    float temp5 = 2*q1*r22_gci;
    float temp6 = 2*q3*r20_gci;
    float temp7 = temp5+4*q0*r21_gci-temp6;
    float temp8 = 2*q1*r11_gci;
    float temp10 = q0*r10_gci;
    float temp9 = -2*q2*r12_gci+2*q3*r11_gci+4*temp10;

    H(3,3) = ((temp2)*temp4*(temp2)+r12_gti_inv*(temp7)+(4*q0*r12_gci-temp8+2*q2*r10_gci)*r21_gti_inv+(4*q0*r02_gci-2*q1*r01_gci+2*q2*r00_gci)*r20_gti_inv+r01_gti_inv*(temp8)+r11_gti_inv*(2*q1*r12_gci+4*q0*r11_gci-2*q3*r10_gci)+(2*q1*r02_gci+4*q0*r01_gci-2*q3*r00_gci)*r10_gti_inv+r00_gti_inv*(-2*q2*r02_gci+2*q3*r01_gci+4*q0*r00_gci))/(4*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1));
    H(4,3) = ((temp7)*r22_gti_inv-r12_gti_inv*(temp2)+(2*q1*r12_gci+4*q0*r11_gci-2*q3*r10_gci)*r21_gti_inv+(2*q1*r02_gci+4*q0*r01_gci-2*q3*r00_gci)*r20_gti_inv-r11_gti_inv*(4*q0*r12_gci-temp8+2*q2*r10_gci)-(4*q0*r02_gci-2*q1*r01_gci+2*q2*r00_gci)*r10_gti_inv)/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-(((temp2)*temp4*(temp2)+r12_gti_inv*(temp7)+(4*q0*r12_gci-temp8+2*q2*r10_gci)*r21_gti_inv+(4*q0*r02_gci-2*q1*r01_gci+2*q2*r00_gci)*r20_gti_inv+r01_gti_inv*(temp8)+r11_gti_inv*(2*q1*r12_gci+4*q0*r11_gci-2*q3*r10_gci)+(2*q1*r02_gci+4*q0*r01_gci-2*q3*r00_gci)*r10_gti_inv+r00_gti_inv*(-2*q2*r02_gci+2*q3*r01_gci+4*q0*r00_gci))*((2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)*r22_gti_inv-r12_gti_inv*((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)+(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)*r21_gti_inv+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r20_gti_inv-r11_gti_inv*((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)-((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r10_gti_inv))/temp1;
    H(5,3) = (-(temp2)*temp4*(temp2)-(temp8)*r21_gti_inv-(-2*q2*r02_gci+2*q3*r01_gci+4*q0*r00_gci)*r20_gti_inv+r01_gti_inv*(4*q0*r12_gci-temp8+2*q2*r10_gci)+r00_gti_inv*(4*q0*r02_gci-2*q1*r01_gci+2*q2*r00_gci))/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-(((temp2)*temp4*(temp2)+r12_gti_inv*(temp7)+(4*q0*r12_gci-temp8+2*q2*r10_gci)*r21_gti_inv+(4*q0*r02_gci-2*q1*r01_gci+2*q2*r00_gci)*r20_gti_inv+r01_gti_inv*(temp8)+r11_gti_inv*(2*q1*r12_gci+4*q0*r11_gci-2*q3*r10_gci)+(2*q1*r02_gci+4*q0*r01_gci-2*q3*r00_gci)*r10_gti_inv+r00_gti_inv*(-2*q2*r02_gci+2*q3*r01_gci+4*q0*r00_gci))*(-(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)*temp4*((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)-(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)*r21_gti_inv-(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)*r20_gti_inv+r01_gti_inv*((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)+r00_gti_inv*((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)))/temp1;
    H(6,3) = (r12_gti_inv*(temp2)-r02_gti_inv*(temp7)+r11_gti_inv*(temp8)-r01_gti_inv*(2*q1*r12_gci+4*q0*r11_gci-2*q3*r10_gci)+(-2*q2*r02_gci+2*q3*r01_gci+4*q0*r00_gci)*r10_gti_inv-r00_gti_inv*(2*q1*r02_gci+4*q0*r01_gci-2*q3*r00_gci))/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-((-r02_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r12_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)-r01_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r11_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)*r10_gti_inv-r00_gti_inv*(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci))*((temp2)*temp4*(temp2)+r12_gti_inv*(temp7)+(4*q0*r12_gci-temp8+2*q2*r10_gci)*r21_gti_inv+(4*q0*r02_gci-2*q1*r01_gci+2*q2*r00_gci)*r20_gti_inv+r01_gti_inv*(temp8)+r11_gti_inv*(2*q1*r12_gci+4*q0*r11_gci-2*q3*r10_gci)+(2*q1*r02_gci+4*q0*r01_gci-2*q3*r00_gci)*r10_gti_inv+r00_gti_inv*(-2*q2*r02_gci+2*q3*r01_gci+4*q0*r00_gci)))/temp1;

    H(3,4) = ((temp6-2*q0*r21_gci)*temp4*(2*q3*r22_gci+2*q2*r21_gci+4*q1*r20_gci)+r12_gti_inv*(2*q0*r22_gci+2*q2*r20_gci)+(2*q3*r10_gci-2*q0*r11_gci)*r21_gti_inv+(2*q3*r00_gci-2*q0*r01_gci)*r20_gti_inv+r01_gti_inv*(2*q3*r12_gci+2*q2*r11_gci+4*q1*r10_gci)+r11_gti_inv*(2*q0*r12_gci+2*q2*r10_gci)+(2*q0*r02_gci+2*q2*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q3*r02_gci+2*q2*r01_gci+4*q1*r00_gci))/(4*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1));
    H(4,4) = ((2*q0*r22_gci+2*q2*r20_gci)*r22_gti_inv+(2*q0*r12_gci+2*q2*r10_gci)*r21_gti_inv-r12_gti_inv*(temp6-2*q0*r21_gci)+(2*q0*r02_gci+2*q2*r00_gci)*r20_gti_inv-(2*q3*r10_gci-2*q0*r11_gci)*r11_gti_inv-(2*q3*r00_gci-2*q0*r01_gci)*r10_gti_inv)/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-(((temp6-2*q0*r21_gci)*temp4*(2*q3*r22_gci+2*q2*r21_gci+4*q1*r20_gci)+r12_gti_inv*(2*q0*r22_gci+2*q2*r20_gci)+(2*q3*r10_gci-2*q0*r11_gci)*r21_gti_inv+(2*q3*r00_gci-2*q0*r01_gci)*r20_gti_inv+r01_gti_inv*(2*q3*r12_gci+2*q2*r11_gci+4*q1*r10_gci)+r11_gti_inv*(2*q0*r12_gci+2*q2*r10_gci)+(2*q0*r02_gci+2*q2*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q3*r02_gci+2*q2*r01_gci+4*q1*r00_gci))*((2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)*r22_gti_inv-r12_gti_inv*((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)+(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)*r21_gti_inv+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r20_gti_inv-r11_gti_inv*((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)-((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r10_gti_inv))/temp1;
    H(5,4) = (-(2*q3*r22_gci+2*q2*r21_gci+4*q1*r20_gci)*r22_gti_inv-(2*q3*r12_gci+2*q2*r11_gci+4*q1*r10_gci)*r21_gti_inv+r02_gti_inv*(temp6-2*q0*r21_gci)-(2*q3*r02_gci+2*q2*r01_gci+4*q1*r00_gci)*r20_gti_inv+r01_gti_inv*(2*q3*r10_gci-2*q0*r11_gci)+r00_gti_inv*(2*q3*r00_gci-2*q0*r01_gci))/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-(((temp6-2*q0*r21_gci)*temp4*(2*q3*r22_gci+2*q2*r21_gci+4*q1*r20_gci)+r12_gti_inv*(2*q0*r22_gci+2*q2*r20_gci)+(2*q3*r10_gci-2*q0*r11_gci)*r21_gti_inv+(2*q3*r00_gci-2*q0*r01_gci)*r20_gti_inv+r01_gti_inv*(2*q3*r12_gci+2*q2*r11_gci+4*q1*r10_gci)+r11_gti_inv*(2*q0*r12_gci+2*q2*r10_gci)+(2*q0*r02_gci+2*q2*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q3*r02_gci+2*q2*r01_gci+4*q1*r00_gci))*(-(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)*temp4*((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)-(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)*r21_gti_inv-(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)*r20_gti_inv+r01_gti_inv*((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)+r00_gti_inv*((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)))/temp1;
    H(6,4) = (r12_gti_inv*(2*q3*r22_gci+2*q2*r21_gci+4*q1*r20_gci)-r02_gti_inv*(2*q0*r22_gci+2*q2*r20_gci)+r11_gti_inv*(2*q3*r12_gci+2*q2*r11_gci+4*q1*r10_gci)-r01_gti_inv*(2*q0*r12_gci+2*q2*r10_gci)+(2*q3*r02_gci+2*q2*r01_gci+4*q1*r00_gci)*r10_gti_inv-r00_gti_inv*(2*q0*r02_gci+2*q2*r00_gci))/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-((-r02_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r12_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)-r01_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r11_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)*r10_gti_inv-r00_gti_inv*(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci))*((temp6-2*q0*r21_gci)*temp4*(2*q3*r22_gci+2*q2*r21_gci+4*q1*r20_gci)+r12_gti_inv*(2*q0*r22_gci+2*q2*r20_gci)+(2*q3*r10_gci-2*q0*r11_gci)*r21_gti_inv+(2*q3*r00_gci-2*q0*r01_gci)*r20_gti_inv+r01_gti_inv*(2*q3*r12_gci+2*q2*r11_gci+4*q1*r10_gci)+r11_gti_inv*(2*q0*r12_gci+2*q2*r10_gci)+(2*q0*r02_gci+2*q2*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q3*r02_gci+2*q2*r01_gci+4*q1*r00_gci)))/temp1;

    H(3,5) = ((temp18+2*temp19)*r22_gti_inv+r12_gti_inv*(2*q3*r22_gci+4*q2*r21_gci+2*q1*r20_gci)+r02_gti_inv*(2*q1*r21_gci-2*q0*r22_gci)+(2*q3*r11_gci+2*temp10)*r21_gti_inv+(2*q3*r01_gci+2*q0*r00_gci)*r20_gti_inv+r11_gti_inv*(2*q3*r12_gci+4*q2*r11_gci+2*q1*r10_gci)+r01_gti_inv*(temp8-2*q0*r12_gci)+(2*q3*r02_gci+4*q2*r01_gci+2*q1*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q1*r01_gci-2*q0*r02_gci))/(4*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1));
    H(4,5) = ((2*q3*r22_gci+4*q2*r21_gci+2*q1*r20_gci)*r22_gti_inv+(2*q3*r12_gci+4*q2*r11_gci+2*q1*r10_gci)*r21_gti_inv-r12_gti_inv*(temp18+2*temp19)+(2*q3*r02_gci+4*q2*r01_gci+2*q1*r00_gci)*r20_gti_inv-(2*q3*r11_gci+2*temp10)*r11_gti_inv-(2*q3*r01_gci+2*q0*r00_gci)*r10_gti_inv)/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-(((temp18+2*temp19)*r22_gti_inv+r12_gti_inv*(2*q3*r22_gci+4*q2*r21_gci+2*q1*r20_gci)+r02_gti_inv*(2*q1*r21_gci-2*q0*r22_gci)+(2*q3*r11_gci+2*temp10)*r21_gti_inv+(2*q3*r01_gci+2*q0*r00_gci)*r20_gti_inv+r11_gti_inv*(2*q3*r12_gci+4*q2*r11_gci+2*q1*r10_gci)+r01_gti_inv*(temp8-2*q0*r12_gci)+(2*q3*r02_gci+4*q2*r01_gci+2*q1*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q1*r01_gci-2*q0*r02_gci))*((2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)*r22_gti_inv-r12_gti_inv*((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)+(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)*r21_gti_inv+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r20_gti_inv-r11_gti_inv*((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)-((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r10_gti_inv))/temp1;
    H(5,5) = (-(2*q1*r21_gci-2*q0*r22_gci)*r22_gti_inv-(temp8-2*q0*r12_gci)*r21_gti_inv+r02_gti_inv*(temp18+2*temp19)-(2*q1*r01_gci-2*q0*r02_gci)*r20_gti_inv+r01_gti_inv*(2*q3*r11_gci+2*temp10)+r00_gti_inv*(2*q3*r01_gci+2*q0*r00_gci))/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-(((temp18+2*temp19)*r22_gti_inv+r12_gti_inv*(2*q3*r22_gci+4*q2*r21_gci+2*q1*r20_gci)+r02_gti_inv*(2*q1*r21_gci-2*q0*r22_gci)+(2*q3*r11_gci+2*temp10)*r21_gti_inv+(2*q3*r01_gci+2*q0*r00_gci)*r20_gti_inv+r11_gti_inv*(2*q3*r12_gci+4*q2*r11_gci+2*q1*r10_gci)+r01_gti_inv*(temp8-2*q0*r12_gci)+(2*q3*r02_gci+4*q2*r01_gci+2*q1*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q1*r01_gci-2*q0*r02_gci))*(-(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)*temp4*((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)-(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)*r21_gti_inv-(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)*r20_gti_inv+r01_gti_inv*((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)+r00_gti_inv*((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)))/temp1;
    H(6,5) = (-r02_gti_inv*(2*q3*r22_gci+4*q2*r21_gci+2*q1*r20_gci)+r12_gti_inv*(2*q1*r21_gci-2*q0*r22_gci)-r01_gti_inv*(2*q3*r12_gci+4*q2*r11_gci+2*q1*r10_gci)+r11_gti_inv*(temp8-2*q0*r12_gci)+(2*q1*r01_gci-2*q0*r02_gci)*r10_gti_inv-r00_gti_inv*(2*q3*r02_gci+4*q2*r01_gci+2*q1*r00_gci))/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-((-r02_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r12_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)-r01_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r11_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)*r10_gti_inv-r00_gti_inv*(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci))*((temp18+2*temp19)*r22_gti_inv+r12_gti_inv*(2*q3*r22_gci+4*q2*r21_gci+2*q1*r20_gci)+r02_gti_inv*(2*q1*r21_gci-2*q0*r22_gci)+(2*q3*r11_gci+2*temp10)*r21_gti_inv+(2*q3*r01_gci+2*q0*r00_gci)*r20_gti_inv+r11_gti_inv*(2*q3*r12_gci+4*q2*r11_gci+2*q1*r10_gci)+r01_gti_inv*(temp8-2*q0*r12_gci)+(2*q3*r02_gci+4*q2*r01_gci+2*q1*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q1*r01_gci-2*q0*r02_gci)))/temp1;

    H(3,6) = ((4*q3*r22_gci+2*q2*r21_gci+2*q1*r20_gci)*r22_gti_inv+r12_gti_inv*(temp17-2*temp19)+r02_gti_inv*(temp5+2*q0*r21_gci)+(4*q3*r12_gci+2*q2*r11_gci+2*q1*r10_gci)*r21_gti_inv+(4*q3*r02_gci+2*q2*r01_gci+2*q1*r00_gci)*r20_gti_inv+r11_gti_inv*(2*q2*r12_gci-2*temp10)+r01_gti_inv*(2*q1*r12_gci+2*q0*r11_gci)+(2*q2*r02_gci-2*q0*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q1*r02_gci+2*q0*r01_gci))/(4*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1));
    H(4,6) = ((temp17-2*temp19)*r22_gti_inv-r12_gti_inv*(4*q3*r22_gci+2*q2*r21_gci+2*q1*r20_gci)+(2*q2*r12_gci-2*temp10)*r21_gti_inv+(2*q2*r02_gci-2*q0*r00_gci)*r20_gti_inv-r11_gti_inv*(4*q3*r12_gci+2*q2*r11_gci+2*q1*r10_gci)-(4*q3*r02_gci+2*q2*r01_gci+2*q1*r00_gci)*r10_gti_inv)/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-(((4*q3*r22_gci+2*q2*r21_gci+2*q1*r20_gci)*r22_gti_inv+r12_gti_inv*(temp17-2*temp19)+r02_gti_inv*(temp5+2*q0*r21_gci)+(4*q3*r12_gci+2*q2*r11_gci+2*q1*r10_gci)*r21_gti_inv+(4*q3*r02_gci+2*q2*r01_gci+2*q1*r00_gci)*r20_gti_inv+r11_gti_inv*(2*q2*r12_gci-2*temp10)+r01_gti_inv*(2*q1*r12_gci+2*q0*r11_gci)+(2*q2*r02_gci-2*q0*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q1*r02_gci+2*q0*r01_gci))*((2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)*r22_gti_inv-r12_gti_inv*((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)+(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)*r21_gti_inv+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r20_gti_inv-r11_gti_inv*((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)-((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r10_gti_inv))/temp1;
    H(5,6) = (-(temp5+2*q0*r21_gci)*temp4*(4*q3*r22_gci+2*q2*r21_gci+2*q1*r20_gci)-(2*q1*r12_gci+2*q0*r11_gci)*r21_gti_inv-(2*q1*r02_gci+2*q0*r01_gci)*r20_gti_inv+r01_gti_inv*(4*q3*r12_gci+2*q2*r11_gci+2*q1*r10_gci)+r00_gti_inv*(4*q3*r02_gci+2*q2*r01_gci+2*q1*r00_gci))/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-(((4*q3*r22_gci+2*q2*r21_gci+2*q1*r20_gci)*r22_gti_inv+r12_gti_inv*(temp17-2*temp19)+r02_gti_inv*(temp5+2*q0*r21_gci)+(4*q3*r12_gci+2*q2*r11_gci+2*q1*r10_gci)*r21_gti_inv+(4*q3*r02_gci+2*q2*r01_gci+2*q1*r00_gci)*r20_gti_inv+r11_gti_inv*(2*q2*r12_gci-2*temp10)+r01_gti_inv*(2*q1*r12_gci+2*q0*r11_gci)+(2*q2*r02_gci-2*q0*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q1*r02_gci+2*q0*r01_gci))*(-(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)*temp4*((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)-(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)*r21_gti_inv-(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)*r20_gti_inv+r01_gti_inv*((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)+r00_gti_inv*((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)))/temp1;
    H(6,6) = (-r02_gti_inv*(temp17-2*temp19)+r12_gti_inv*(temp5+2*q0*r21_gci)-r01_gti_inv*(2*q2*r12_gci-2*temp10)+r11_gti_inv*(2*q1*r12_gci+2*q0*r11_gci)+(2*q1*r02_gci+2*q0*r01_gci)*r10_gti_inv-r00_gti_inv*(2*q2*r02_gci-2*q0*r00_gci))/(2*sqrt(((temp11)*r22_gci+2*(temp20)*r21_gci+2*(temp21)*r20_gci)*r22_gti_inv+r12_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r02_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)+((temp11)*r12_gci+2*(temp20)*r11_gci+2*(temp21)*r10_gci)*r21_gti_inv+((temp11)*r02_gci+2*(temp20)*r01_gci+2*(temp21)*r00_gci)*r20_gti_inv+r11_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r01_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci)*r10_gti_inv+r00_gti_inv*(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)+1))-((-r02_gti_inv*(2*(temp22)*r22_gci+(temp16)*r21_gci+2*(temp25)*r20_gci)+r12_gti_inv*(2*(temp26)*r22_gci+2*(temp27)*r21_gci+(temp28)*r20_gci)-r01_gti_inv*(2*(temp22)*r12_gci+(temp16)*r11_gci+2*(temp25)*r10_gci)+r11_gti_inv*(2*(temp26)*r12_gci+2*(temp27)*r11_gci+(temp28)*r10_gci)+(2*(temp26)*r02_gci+2*(temp27)*r01_gci+(temp28)*r00_gci)*r10_gti_inv-r00_gti_inv*(2*(temp22)*r02_gci+(temp16)*r01_gci+2*(temp25)*r00_gci))*((4*q3*r22_gci+2*q2*r21_gci+2*q1*r20_gci)*r22_gti_inv+r12_gti_inv*(temp17-2*temp19)+r02_gti_inv*(temp5+2*q0*r21_gci)+(4*q3*r12_gci+2*q2*r11_gci+2*q1*r10_gci)*r21_gti_inv+(4*q3*r02_gci+2*q2*r01_gci+2*q1*r00_gci)*r20_gti_inv+r11_gti_inv*(2*q2*r12_gci-2*temp10)+r01_gti_inv*(2*q1*r12_gci+2*q0*r11_gci)+(2*q2*r02_gci-2*q0*r00_gci)*r10_gti_inv+r00_gti_inv*(2*q1*r02_gci+2*q0*r01_gci)))/temp1;

    H(0,7) = r11_gni_inv*r22_gni-r21_gni*r21_gni_inv;
    H(1,7) = r02_gni*r12_gni_inv-r01_gni*r22_gni_inv;
    H(2,7) = r10_gni_inv*r12_gni-r11_gni*r20_gni_inv;
    H(7,7) = r02_gni*t2_gni_inv-r01_gni*t3_gni_inv;
    H(8,7) = r12_gni*t2_gni_inv-r11_gni*t3_gni_inv;
    H(9,7) = r22_gni*t2_gni_inv-r21_gni*t3_gni_inv;

    H(0,8) = r20_gni*r21_gni_inv-r01_gni_inv*r22_gni;
    H(1,8) = r00_gni*r22_gni_inv-r02_gni*r02_gni_inv;
    H(2,8) = r10_gni*r20_gni_inv-r00_gni_inv*r12_gni;
    H(7,8) = r00_gni*t3_gni_inv-r02_gni*t1_gni_inv;
    H(8,8) = r10_gni*t3_gni_inv-r12_gni*t1_gni_inv;
    H(9,8) = r20_gni*t3_gni_inv-r22_gni*t1_gni_inv;

    H(0,9) = r01_gni_inv*r21_gni-r11_gni_inv*r20_gni;
    H(1,9) = r01_gni*r02_gni_inv-r00_gni*r12_gni_inv;
    H(2,9) = r00_gni_inv*r11_gni-r10_gni*r10_gni_inv;
    H(7,9) = r01_gni*t1_gni_inv-r00_gni*t2_gni_inv;
    H(8,9) = r11_gni*t1_gni_inv-r10_gni*t2_gni_inv;
    H(9,9) = r21_gni*t1_gni_inv-r20_gni*t2_gni_inv;

    H(7,10) = r00_gni;
    H(8,10) = r10_gni;
    H(9,10) = r20_gni;

    H(7,11) = r01_gni;
    H(8,11) = r11_gni;
    H(9,11) = r21_gni;

    H(7,11) = r02_gni;
    H(8,11) = r12_gni;
    H(9,11) = r22_gni;
}
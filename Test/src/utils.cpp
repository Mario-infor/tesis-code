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

Eigen::Matrix3d getWHat(const Eigen::Vector3d v)
{
    Eigen::Matrix3d wHat;
    wHat << 
    0,      -v.z(),  v.y(),
    v.z(),   0,      -v.x(),
    -v.y(),  v.x(),   0;

    return wHat;
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
    
    usleep(500000);
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


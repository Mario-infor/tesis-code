#include <utils.h>
#include <BNO055-BBB_driver.h>
#include <curses.h>

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
cv::Vec3d convertQuatToOpencvRotVect(glm::quat quaternion)
{
    cv::Vec3d rotVect;

    float w = quaternion.w;
    float x = quaternion.x;
    float y = quaternion.y;
    float z = quaternion.z;

    float vecNorm = acos(w * 2);

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

        // sleep(1);

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
    cv::Mat distCoeffs)
{
    for (int i = 0; i < (int)rvecs.size(); i++)
    {
        cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
    }

    cv::imshow("draw axis", frame);
}
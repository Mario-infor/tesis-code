#include <interpolationUtils.h>
#include <utils.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/spline.hpp>

std::string dirRotationsFolder = "/home/nvidia/Mario/tesis-code/Test/data/rotations/";

std::vector<glm::vec3> createSplinePoint(std::vector<ImuInputJetson> imuReadVector)
{
    std::vector<glm::vec3> points;

    for (size_t i = 0; i < imuReadVector.size() - 4; i++)
    {
        // Create a for loop for t values from 0 to 1 with a step of 0.1.
        for (float t = 0; t < 1; t += 0.1)
        {
            glm::vec3 tempPoint = glm::catmullRom(imuReadVector[i].accVect, imuReadVector[i + 1].accVect,
                                                  imuReadVector[i + 2].accVect, imuReadVector[i + 3].accVect, t);

            points.push_back(tempPoint);
        }
    }

    return points;
}

std::vector<glm::quat> createSlerpPoint(std::vector<ImuInputJetson> imuReadVector)
{
    std::vector<glm::quat> points;

    for (size_t i = 0; i < imuReadVector.size() - 1; i++)
    {
        // Create a for loop for t values from 0 to 1 with a step of 0.1.
        for (float t = 0; t < 1; t += 0.1)
        {
            glm::quat tempPoint = glm::slerp(imuReadVector.at(i).rotQuat, imuReadVector.at(i + 1).rotQuat, t);
            points.push_back(tempPoint);
        }
    }

    return points;
}

void testSlerpAndSpline(std::vector<ImuInputJetson> imuReadVector, std::vector<CameraInput> cameraReadVector)
{
    std::vector<glm::vec3> splinePoints = createSplinePoint(imuReadVector);
    std::vector<glm::quat> slerpPoints = createSlerpPoint(imuReadVector);

    std::cout << "Spline points: " << std::endl;
    for (size_t i = 0; i < splinePoints.size(); i++)
    {
        std::cout << "X: " << splinePoints[i].x << " Y: " << splinePoints[i].y << " Z: " << splinePoints[i].z << std::endl;
    }

    std::cout << "Slerp points: " << std::endl;
    for (size_t i = 0; i < slerpPoints.size(); i++)
    {
        std::cout << "W: " << slerpPoints[i].w << " X: " << slerpPoints[i].x << " Y: " << slerpPoints[i].y << " Z: " << slerpPoints[i].z << std::endl;
    }
}

std::vector<CameraInterpolatedData> interpolateCameraRotation(const std::vector<ImuInputJetson> imuReadVectorCopy,
                                                              const std::vector<CameraInput> cameraReadVectorCopy,
                                                              std::vector<FrameMarkersData> frameMarkersDataVector)
{
    std::vector<CameraInterpolatedData> interpolatedPoints;

    int indexCamera = 0;
    int indexIMU = 0;

    while (indexCamera != (int)(cameraReadVectorCopy.size() - 1))
    {
        CameraInput tempCameraInput = cameraReadVectorCopy[indexCamera];
        FrameMarkersData tempFrameMarkersData = frameMarkersDataVector[indexCamera];
        CameraInterpolatedData tempCameraInterpolatedData;

        tempCameraInterpolatedData.originalOrNot = 0;
        tempCameraInterpolatedData.frame = tempCameraInput;
        tempCameraInterpolatedData.frameMarkersData = tempFrameMarkersData;

        interpolatedPoints.push_back(tempCameraInterpolatedData);

        for (size_t i = indexIMU; i < imuReadVectorCopy.size(); i++)
        {
            if (imuReadVectorCopy[i].time > cameraReadVectorCopy[indexCamera].time && imuReadVectorCopy[i].time < cameraReadVectorCopy[indexCamera + 1].time)
            {
                tempCameraInput.index = -1;
                tempCameraInput.time = imuReadVectorCopy[i].time;
                tempCameraInput.frame = cameraReadVectorCopy[indexCamera].frame;

                for (size_t j = 0; j < frameMarkersDataVector[indexCamera].rvecs.size(); j++)
                {
                    cv::Vec3d rotVect0 = frameMarkersDataVector[indexCamera].rvecs[j];
                    cv::Vec3d rotVect1 = frameMarkersDataVector[indexCamera + 1].rvecs[j];

                    glm::quat quaternion0 = convertOpencvRotVectToQuat(rotVect0);
                    glm::quat quaternion1 = convertOpencvRotVectToQuat(rotVect1);

                    float relativePos = (float)(imuReadVectorCopy[i].time - cameraReadVectorCopy[indexCamera].time) /
                                        (float)(cameraReadVectorCopy[indexCamera + 1].time - cameraReadVectorCopy[indexCamera].time);

                    glm::quat interpolatedPoint = glm::slerp(quaternion0, quaternion1, relativePos);
                    cv::Vec3d rotacionVec3 = QuatToRotVect(interpolatedPoint);

                    tempFrameMarkersData.rvecs.push_back(rotacionVec3);
                    tempFrameMarkersData.qvecs.push_back(cv::Vec4d(interpolatedPoint.w, interpolatedPoint.x, interpolatedPoint.y, interpolatedPoint.z));

                    tempFrameMarkersData.tvecs.push_back(frameMarkersDataVector[indexCamera].tvecs[j]);
                    tempFrameMarkersData.markerIds.push_back(frameMarkersDataVector[indexCamera].markerIds[j]);
                }

                tempCameraInterpolatedData.originalOrNot = 1;
                tempCameraInterpolatedData.frame = tempCameraInput;
                tempCameraInterpolatedData.frameMarkersData = tempFrameMarkersData;

                interpolatedPoints.push_back(tempCameraInterpolatedData);

                indexIMU = i + 1;
            }
            else if (imuReadVectorCopy[i].time > cameraReadVectorCopy[indexCamera + 1].time)
                break;
        }
        indexCamera++;
    }

    return interpolatedPoints;
}

void testInterpolateCamera(std::vector<CameraInterpolatedData> interpolatedPoints,
                           cv::Mat cameraMatrix, cv::Mat distCoeffs)
{
    for (int i = 0; i < (int)interpolatedPoints.size(); i++)
    {
        for (size_t j = 0; j < interpolatedPoints[i].frameMarkersData.rvecs.size(); j++)
        {
            if (!std::isnan(interpolatedPoints[i].frameMarkersData.rvecs[j][0]))
            {
                std::cout << "Rvec: " << interpolatedPoints[i].frameMarkersData.rvecs[j] << std::endl;

                cv::aruco::drawAxis(interpolatedPoints[i].frame.frame, cameraMatrix, distCoeffs, interpolatedPoints[i].frameMarkersData.rvecs[j],
                                    interpolatedPoints[i].frameMarkersData.tvecs[j], 0.1);
            }
        }

        cv::imshow("Test", interpolatedPoints[i].frame.frame);

        cv::waitKey(1);
    }
    cv::destroyAllWindows();
}


void cameraRotationSlerpDataWrite(std::vector<CameraInterpolatedData> cameraSlerpRotationsVector)
{
    std::string tempName1 = dirRotationsFolder + "slerpRotations23.csv";
    std::string tempName2 = dirRotationsFolder + "slerpRotations30.csv";
    std::string tempName3 = dirRotationsFolder + "slerpRotations45.csv";
    std::string tempName4 = dirRotationsFolder + "slerpRotations80.csv";

    std::ofstream cameraRotationsFile1(tempName1, std::ios::out);
    std::ofstream cameraRotationsFile2(tempName2, std::ios::out);
    std::ofstream cameraRotationsFile3(tempName3, std::ios::out);
    std::ofstream cameraRotationsFile4(tempName4, std::ios::out);

    if (cameraRotationsFile1.is_open() && cameraRotationsFile2.is_open() && cameraRotationsFile3.is_open() && cameraRotationsFile4.is_open())
    {
        for (size_t i = 0; i < cameraSlerpRotationsVector.size(); i++)
        {
            for (size_t j = 0; j < cameraSlerpRotationsVector[i].frameMarkersData.markerIds.size(); j++)
            {
                int originalOrNot = cameraSlerpRotationsVector[i].originalOrNot;
                cv::Vec3d tempRvec = cameraSlerpRotationsVector[i].frameMarkersData.rvecs[j];
                int tempMarkerId = cameraSlerpRotationsVector[i].frameMarkersData.markerIds[j];

                if (tempMarkerId == 23)
                    cameraRotationsFile1 << originalOrNot << "," << tempRvec[0] << "," << tempRvec[1] << "," << tempRvec[2] << std::endl;

                else if (tempMarkerId == 30)
                    cameraRotationsFile2 << originalOrNot << "," << tempRvec[0] << "," << tempRvec[1] << "," << tempRvec[2] << std::endl;

                else if (tempMarkerId == 45)
                    cameraRotationsFile3 << originalOrNot << "," << tempRvec[0] << "," << tempRvec[1] << "," << tempRvec[2] << std::endl;

                else if (tempMarkerId == 80)
                    cameraRotationsFile4 << originalOrNot << "," << tempRvec[0] << "," << tempRvec[1] << "," << tempRvec[2] << std::endl;
            }
        }
    }
}

void cameraRotationSlerpDataWrite(std::vector<FrameMarkersData> cameraRotationsVector)
{
    std::string tempName1 = dirRotationsFolder + "rotations23.csv";
    std::string tempName2 = dirRotationsFolder + "rotations30.csv";
    std::string tempName3 = dirRotationsFolder + "rotations45.csv";
    std::string tempName4 = dirRotationsFolder + "rotations80.csv";

    std::ofstream cameraRotationsFile1(tempName1, std::ios::out);
    std::ofstream cameraRotationsFile2(tempName2, std::ios::out);
    std::ofstream cameraRotationsFile3(tempName3, std::ios::out);
    std::ofstream cameraRotationsFile4(tempName4, std::ios::out);

    if (cameraRotationsFile1.is_open() && cameraRotationsFile2.is_open() && cameraRotationsFile3.is_open() && cameraRotationsFile4.is_open())
    {
        for (size_t i = 0; i < cameraRotationsVector.size(); i++)
        {
            for (size_t j = 0; j < cameraRotationsVector[i].markerIds.size(); j++)
            {
                cv::Vec3d tempRvec = cameraRotationsVector[i].rvecs[j];
                int tempMarkerId = cameraRotationsVector[i].markerIds[j];

                if (tempMarkerId == 23)
                    cameraRotationsFile1 << tempRvec[0] << "," << tempRvec[1] << "," << tempRvec[2] << std::endl;

                else if (tempMarkerId == 30)
                    cameraRotationsFile2 << tempRvec[0] << "," << tempRvec[1] << "," << tempRvec[2] << std::endl;

                else if (tempMarkerId == 45)
                    cameraRotationsFile3 << tempRvec[0] << "," << tempRvec[1] << "," << tempRvec[2] << std::endl;

                else if (tempMarkerId == 80)
                    cameraRotationsFile4 << tempRvec[0] << "," << tempRvec[1] << "," << tempRvec[2] << std::endl;
            }
        }
    }
}

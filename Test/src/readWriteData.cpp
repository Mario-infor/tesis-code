#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include "readWriteData.h"
#include "structsFile.h"
#include "RingBuffer.h"

std::string dirCameraFolder = "./data/camera/";
std::string dirIMUFolder = "./data/imu/";
std::string dirRotationsFolder = "./data/rotations/";

void IMUDataJetsonWrite(RingBuffer<ImuInputJetson> &imuDataJetsonBuffer)
{
    std::ofstream IMUTimeFile(dirIMUFolder + "IMUTime", std::ios::out);
    std::ofstream IMUDataFile(dirIMUFolder + "IMUData", std::ios::out);

    if (IMUTimeFile.is_open() && IMUDataFile.is_open())
    {
        while (!imuDataJetsonBuffer.QueueIsEmpty())
        {
            ImuInputJetson tempIMU;
            imuDataJetsonBuffer.Dequeue(tempIMU);

            IMUTimeFile << tempIMU.time << std::endl;

            IMUDataFile << tempIMU.index << std::endl;
            IMUDataFile << tempIMU.gyroVect.x << std::endl;
            IMUDataFile << tempIMU.gyroVect.y << std::endl;
            IMUDataFile << tempIMU.gyroVect.z << std::endl;
            IMUDataFile << tempIMU.eulerVect.x << std::endl;
            IMUDataFile << tempIMU.eulerVect.y << std::endl;
            IMUDataFile << tempIMU.eulerVect.z << std::endl;
            IMUDataFile << tempIMU.rotQuat.w << std::endl;
            IMUDataFile << tempIMU.rotQuat.x << std::endl;
            IMUDataFile << tempIMU.rotQuat.y << std::endl;
            IMUDataFile << tempIMU.rotQuat.z << std::endl;
            IMUDataFile << tempIMU.accVect.x << std::endl;
            IMUDataFile << tempIMU.accVect.y << std::endl;
            IMUDataFile << tempIMU.accVect.z << std::endl;
            IMUDataFile << tempIMU.gravVect.x << std::endl;
            IMUDataFile << tempIMU.gravVect.y << std::endl;
            IMUDataFile << tempIMU.gravVect.z << std::endl;
        }
    }
}

void IMUDataWriteTestAxis(RingBuffer<ImuInputJetson> &imuDataJetsonBuffer)
{
    std::ofstream IMUDataFile(dirIMUFolder + "IMUData", std::ios::out);

    if (IMUDataFile.is_open())
    {
        while (!imuDataJetsonBuffer.QueueIsEmpty())
        {
            ImuInputJetson tempIMU;
            imuDataJetsonBuffer.Dequeue(tempIMU);

            IMUDataFile << tempIMU.accVect.x << "  " << tempIMU.accVect.y << "  " << tempIMU.accVect.z << std::endl;
        }
    }
}


std::vector<ImuInputJetson> readDataIMUJetson()
{
    std::vector<ImuInputJetson> IMUData;
    std::ifstream fileTime(dirIMUFolder + "IMUTime");
    std::ifstream fileData(dirIMUFolder + "IMUData");

    if (!fileTime || !fileData)
        std::cerr << "Files not found." << std::endl;
    else
    {
        int value;
        while (fileTime >> value)
        {
            ImuInputJetson tempIMUInput;

            tempIMUInput.time = value;

            fileData >> tempIMUInput.index;
            fileData >> tempIMUInput.gyroVect.x;
            fileData >> tempIMUInput.gyroVect.y;
            fileData >> tempIMUInput.gyroVect.z;
            fileData >> tempIMUInput.eulerVect.x;
            fileData >> tempIMUInput.eulerVect.y;
            fileData >> tempIMUInput.eulerVect.z;
            fileData >> tempIMUInput.rotQuat.w;
            fileData >> tempIMUInput.rotQuat.x;
            fileData >> tempIMUInput.rotQuat.y;
            fileData >> tempIMUInput.rotQuat.z;
            fileData >> tempIMUInput.accVect.x;
            fileData >> tempIMUInput.accVect.y;
            fileData >> tempIMUInput.accVect.z;
            fileData >> tempIMUInput.gravVect.x;
            fileData >> tempIMUInput.gravVect.y;
            fileData >> tempIMUInput.gravVect.z;

            IMUData.push_back(tempIMUInput);
        }
    }

    return IMUData;
}

void cameraDataWrite(RingBuffer<CameraInput> &cameraFramesBuffer)
{
    std::ofstream cameraTimeFile(dirCameraFolder + "cameraTime", std::ios::out);

    if (cameraTimeFile.is_open())
    {
        while (!cameraFramesBuffer.QueueIsEmpty())
        {
            char buff[256];

            CameraInput tempFrame;
            cameraFramesBuffer.Dequeue(tempFrame);
            snprintf(buff, 255, "frame_%06d.png", tempFrame.index);
            std::string imageName(buff);
            cv::imwrite(dirCameraFolder + imageName, tempFrame.frame);

            cameraTimeFile << tempFrame.time << std::endl;
        }
    }
}

std::vector<CameraInput> readDataCamera()
{
    std::vector<CameraInput> cameraData;
    std::ifstream fileTime(dirCameraFolder + "cameraTime");

    int index = 0;
    std::string imageName = "";
    cv::Mat image;

    char buff[256];

    if (!fileTime)
        std::cerr << "File not found." << std::endl;
    else
    {
        int value;
        while (fileTime >> value)
        {
            CameraInput tempCameraInput;
            tempCameraInput.time = value;
            tempCameraInput.index = index;

            snprintf(buff, 255, "frame_%06d.png", tempCameraInput.index);
            // imageName = "frame_" + std::to_string(index) + ".png";
            std::string imageName(buff);
            image = cv::imread(dirCameraFolder + imageName, cv::IMREAD_GRAYSCALE);
            image.copyTo(tempCameraInput.frame);

            cameraData.push_back(tempCameraInput);
            index++;
        }
    }

    return cameraData;
}

void cameraRotationSlerpDataWrite(std::vector<CameraInterpolatedData> cameraSlerpRotationsVector)
{
    std::ofstream cameraRotationsFile1(dirRotationsFolder + "slerpRotations23.csv", std::ios::out);
    std::ofstream cameraRotationsFile2(dirRotationsFolder + "slerpRotations30.csv", std::ios::out);
    std::ofstream cameraRotationsFile3(dirRotationsFolder + "slerpRotations45.csv", std::ios::out);
    std::ofstream cameraRotationsFile4(dirRotationsFolder + "slerpRotations80.csv", std::ios::out);

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
    std::ofstream cameraRotationsFile1(dirRotationsFolder + "rotations23.csv", std::ios::out);
    std::ofstream cameraRotationsFile2(dirRotationsFolder + "rotations30.csv", std::ios::out);
    std::ofstream cameraRotationsFile3(dirRotationsFolder + "rotations45.csv", std::ios::out);
    std::ofstream cameraRotationsFile4(dirRotationsFolder + "rotations80.csv", std::ios::out);

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
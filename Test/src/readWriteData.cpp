#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <readWriteData.h>
#include <RingBuffer.h>

std::string dirPointsFolder = "/home/nvidia/Mario/tesis-code/Test/data/points_data/";

std::string dirCameraFolder = "/home/nvidia/Mario/tesis-code/Test/data/dynamic_data/camera/";
std::string dirIMUFolder = "/home/nvidia/Mario/tesis-code/Test/data/dynamic_data/imu/";

//std::string dirCameraFolder = "/home/nvidia/Mario/tesis-code/Test/data/static_data/camera/";
//std::string dirIMUFolder = "/home/nvidia/Mario/tesis-code/Test/data/static_data/imu/";

void IMUDataJetsonWrite(RingBuffer<ImuInputJetson> &imuDataJetsonBuffer)
{
    std::string tempNameIMUTime = dirIMUFolder + "IMUTime";
    std::string tempNameIMUData = dirIMUFolder + "IMUData";

    std::ofstream IMUTimeFile(tempNameIMUTime, std::ios::out);
    std::ofstream IMUDataFile(tempNameIMUData, std::ios::out);

    if (IMUTimeFile.is_open() && IMUDataFile.is_open())
    {
        while (!imuDataJetsonBuffer.QueueIsEmpty())
        {
            ImuInputJetson tempIMU;
            imuDataJetsonBuffer.Dequeue(tempIMU);

            IMUTimeFile << tempIMU.time << std::endl;

            IMUDataFile << tempIMU.index << std::endl;
            IMUDataFile << tempIMU.gyroVect.x() << std::endl;
            IMUDataFile << tempIMU.gyroVect.y() << std::endl;
            IMUDataFile << tempIMU.gyroVect.z() << std::endl;
            IMUDataFile << tempIMU.eulerVect.x() << std::endl;
            IMUDataFile << tempIMU.eulerVect.y() << std::endl;
            IMUDataFile << tempIMU.eulerVect.z() << std::endl;
            IMUDataFile << tempIMU.rotQuat.w() << std::endl;
            IMUDataFile << tempIMU.rotQuat.x() << std::endl;
            IMUDataFile << tempIMU.rotQuat.y() << std::endl;
            IMUDataFile << tempIMU.rotQuat.z() << std::endl;
            IMUDataFile << tempIMU.accVect.x()<< std::endl;
            IMUDataFile << tempIMU.accVect.y() << std::endl;
            IMUDataFile << tempIMU.accVect.z() << std::endl;
            IMUDataFile << tempIMU.gravVect.x() << std::endl;
            IMUDataFile << tempIMU.gravVect.y() << std::endl;
            IMUDataFile << tempIMU.gravVect.z() << std::endl;
        }
    }
}

std::vector<ImuInputJetson> readDataIMUJetson()
{
    std::string tempNameIMUTime = dirIMUFolder + "IMUTime";
    std::string tempNameIMUData = dirIMUFolder + "IMUData";

    std::cout << "Searching IMU times at: " << tempNameIMUTime << std::endl;
    std::cout << "Searching IMU data at: " << tempNameIMUTime << std::endl;

    std::vector<ImuInputJetson> IMUData;
    std::ifstream fileTime(tempNameIMUTime);
    std::ifstream fileData(tempNameIMUData);

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
            fileData >> tempIMUInput.gyroVect.x();
            fileData >> tempIMUInput.gyroVect.y();
            fileData >> tempIMUInput.gyroVect.z();
            fileData >> tempIMUInput.eulerVect.x();
            fileData >> tempIMUInput.eulerVect.y();
            fileData >> tempIMUInput.eulerVect.z();
            fileData >> tempIMUInput.rotQuat.w();
            fileData >> tempIMUInput.rotQuat.x();
            fileData >> tempIMUInput.rotQuat.y();
            fileData >> tempIMUInput.rotQuat.z();
            fileData >> tempIMUInput.accVect.x();
            fileData >> tempIMUInput.accVect.y();
            fileData >> tempIMUInput.accVect.z();
            fileData >> tempIMUInput.gravVect.x();
            fileData >> tempIMUInput.gravVect.y();
            fileData >> tempIMUInput.gravVect.z();

            IMUData.push_back(tempIMUInput);
        }
    }

    return IMUData;
}

void cameraDataWrite(RingBuffer<CameraInput> &cameraFramesBuffer)
{
    std::string tempName = dirCameraFolder + "cameraTime";
    std::ofstream cameraTimeFile(tempName, std::ios::out);

    std::cout << tempName << std::endl;

    if (cameraTimeFile.is_open())
    {
        while (!cameraFramesBuffer.QueueIsEmpty())
        {
            char buff[256];

            CameraInput tempFrame;
            cameraFramesBuffer.Dequeue(tempFrame);
            snprintf(buff, 255, "frame_%06d.png", tempFrame.index);
            std::string imageName(buff);
            std::cout << dirCameraFolder + imageName << std::endl;
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
            std::string imageName(buff);
            image = cv::imread(dirCameraFolder + imageName, cv::IMREAD_GRAYSCALE);
            
            image.copyTo(tempCameraInput.frame);

            cameraData.push_back(tempCameraInput);
            index++;
        }
    }

    std::cout << "Exit readDataCamera method."<< std::endl;

    return cameraData;
}

void pointsDataWrite(
    std::vector<Eigen::VectorXd> vectorOfPointsOne,
    std::vector<Eigen::VectorXd> vectorOfPointsTwo,
    std::vector<float> timeStamps,
    std::string fileName)
{
    std::string tempNamePointsData = dirPointsFolder + fileName;

    std::ofstream PointsFile(tempNamePointsData, std::ios::out);

    if (PointsFile.is_open())
    {
        for (size_t i = 0; i < vectorOfPointsOne.size(); i++)
        {
            PointsFile << timeStamps.at(i) << ",";

            for (int j = 0; j < vectorOfPointsOne.at(i).rows(); j++)
            {
                PointsFile << vectorOfPointsOne.at(i)(j) << ",";
            }

            for (int j = 0; j < vectorOfPointsTwo.at(i).rows(); j++)
            {
                if (j == vectorOfPointsTwo.at(i).rows() - 1)
                {
                    PointsFile << vectorOfPointsTwo.at(i)(j) << std::endl;
                    
                }
                else
                {
                    PointsFile << vectorOfPointsTwo.at(i)(j) << ",";
                }
            }
        }
    }
}

void quatDataWrite(
    std::vector<Eigen::Quaterniond> vectorOfQuats,
    std::vector<float> timeStamps,
    std::string fileName)
{
    std::string tempNamePointsData = dirPointsFolder + fileName;

    std::ofstream PointsFile(tempNamePointsData, std::ios::out);

    if (PointsFile.is_open())
    {
        for (size_t i = 0; i < vectorOfQuats.size(); i++)
        {
            PointsFile << timeStamps.at(i) << ",";

            PointsFile << vectorOfQuats.at(i).w() << ",";
            PointsFile << vectorOfQuats.at(i).x() << ",";
            PointsFile << vectorOfQuats.at(i).y() << ",";
            PointsFile << vectorOfQuats.at(i).z() << std::endl;
        }
    }
}
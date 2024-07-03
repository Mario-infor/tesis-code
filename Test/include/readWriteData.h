#ifndef READWRITEDATA_H
#define READWRITEDATA_H    

#include <vector>
#include <structsFile.h>
#include <RingBuffer.h>
#include <Eigen/Dense>

/**
 * @brief This method is in charge of writing the IMU data to a binary file. It writes the time, index, gyro, euler, quaternion,
 * acceleration and gravity data of the IMU to a file.
 * @param imuDataJetsonBuffer Buffer that contains the IMU data.
*/
void IMUDataJetsonWrite(RingBuffer<ImuInputJetson> &imuDataJetsonBuffer);

// Read IMU data from files.
std::vector<ImuInputJetson> readDataIMUJetson();

// Write camera time data to file and store all frams as .png files.
void cameraDataWrite(RingBuffer<CameraInput> &cameraFramesBuffer);

// Read camera data and frames from files.
std::vector<CameraInput> readDataCamera();

void pointsDataWrite(
    std::vector<Eigen::VectorXd> vectorOfPointsOne,
    std::vector<Eigen::VectorXd> vectorOfPointsTwo,
    std::vector<float> timeStamps,
    std::string fileName);

void quatDataWrite(
    std::vector<Eigen::Quaterniond> vectorOfQuats,
    std::vector<float> timeStamps,
    std::string fileName);

#endif // READWRITEDATA_H
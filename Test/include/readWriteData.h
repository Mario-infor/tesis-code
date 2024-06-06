#ifndef READWRITEDATA_H
#define READWRITEDATA_H    

#include <vector>
#include <structsFile.h>
#include <RingBuffer.h>
#include <Eigen/Dense>

// Write IMU data to files.
void IMUDataJetsonWrite(RingBuffer<ImuInputJetson> &imuDataJetsonBuffer);

// Write camera time data to file and store all frams as .png files.
void cameraDataWrite(RingBuffer<CameraInput> &cameraFramesBuffer);

// Read camera data and frames from files.
std::vector<CameraInput> readDataCamera();

// Read IMU data from files.
std::vector<ImuInputJetson> readDataIMUJetson();

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
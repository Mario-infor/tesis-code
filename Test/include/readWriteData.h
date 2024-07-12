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
 * @return None
*/
void IMUDataJetsonWrite(RingBuffer<ImuInputJetson> &imuDataJetsonBuffer);

/**
 * @brief This method is in charge of reading the IMU data from a binary file. It reads the time, index, gyro, euler, quaternion,
 * acceleration and gravity data of the IMU from a file.
 * @param None
 * @return std::vector<ImuInputJetson> Vector filled with IMU data structurs.
*/
std::vector<ImuInputJetson> readDataIMUJetson();

/**
 * @brief This method is in charge of writing the camera data to a binary file. It writes the index, time and frame of the camera to a file.
 * @param cameraFramesBuffer Buffer that contains the camera data.
 * @return None
*/
void cameraDataWrite(RingBuffer<CameraInput> &cameraFramesBuffer);

/**
 * @brief This method is in charge of reading the camera data from a binary file. It reads the index, time and frame of the camera from a file.
 * @param None
 * @return std::vector<CameraInput> Vector filled with camera data structurs.
*/
std::vector<CameraInput> readDataCamera();

/**
 * @brief This method is in charge of writing the points data to a binary file. It takes two vectors of points, a vector of timestamps and a file name.
 * This method is used to write the camera measurements and the state to a csv file at the end of the Kalman Filter algorthm. But it can be used to 
 * write any two vectors of points with any dimensions it does not have to be 3D. It writes the one element of the timestamp vector, one element of the
 * first vector of points and one element of the second vector of points in each row of the file. All three vectors need to be the same size.
 * @param vectorOfPointsOne First vector of points that will be written to the file.
 * @param vectorOfPointsTwo Second vector of points that will be written next to the data from the first vector.
 * @param timeStamps Vector of timestamps that will be written to the file.
 * @param fileName Name of the file where the data will be written.
 * @return None
*/
void pointsDataWrite(
    std::vector<Eigen::VectorXd> vectorOfPointsOne,
    std::vector<Eigen::VectorXd> vectorOfPointsTwo,
    std::vector<float> timeStamps,
    std::string fileName);

/**
 * @brief This method is in charge of writing a list of quaternions to a binary file. It takes a vector of quaternions, a vector of timestamps and a file name.
 * This method is used to write quaternions and there timestamps to a csv file at the end of the Kalman Filter algorthm.
 * @param vectorOfQuats Vector of quaternions that will be written to the file.
 * @param timeStamps Vector of timestamps that will be written to the file.
 * @param fileName Name of the file where the data will be written.
 * @return None
*/
void quatDataWrite(
    std::vector<Eigen::Quaterniond> vectorOfQuats,
    std::vector<float> timeStamps,
    std::string fileName);

void transformWrite(const Eigen::Matrix4d transform, int fileName, const bool clearFile);

#endif // READWRITEDATA_H
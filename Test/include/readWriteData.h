#ifndef READWRITEDATA_H
#define READWRITEDATA_H    

#include <vector>
#include <structsFile.h>
#include <RingBuffer.h>

// Write IMU data to files.
void IMUDataJetsonWrite(RingBuffer<ImuInputJetson> &imuDataJetsonBuffer);

// Write IMU data to files for testing the axis distribution.
void IMUDataWriteTestAxis(RingBuffer<ImuInputJetson> &imuDataJetsonBuffer);

// Write camera time data to file and store all frams as .png files.
void cameraDataWrite(RingBuffer<CameraInput> &cameraFramesBuffer);

// Write camera rotations after slerp and store on .csv file.
void cameraRotationSlerpDataWrite(std::vector<CameraInterpolatedData> cameraSlerpRotationsVector);
// Write camera rotations without slerp and store on .csv file.
void cameraRotationSlerpDataWrite(std::vector<FrameMarkersData> cameraRotationsVector);

// Read camera data and frames from files.
std::vector<CameraInput> readDataCamera();

// Read IMU data from files.
std::vector<ImuInputJetson> readDataIMUJetson();

#endif // READWRITEDATA_H
#include <vector>

// Write IMU data to files.
void IMUDataJetsonWrite();

// Write IMU data to files for testing the axis distribution.
void IMUDataWriteTestAxis();

// Read IMU data from files.
std::vector<struct ImuInputJetson> readDataIMUJetson();

// Write camera time data to file and store all frams as .png files.
void cameraDataWrite();

// Read camera data and frames from files.
std::vector<struct CameraInput> readDataCamera();

// Write camera rotations after slerp and store on .csv file.
void cameraRotationSlerpDataWrite(std::vector<struct CameraInterpolatedData> cameraSlerpRotationsVector);
// Write camera rotations without slerp and store on .csv file.
void cameraRotationSlerpDataWrite(std::vector<struct FrameMarkersData> cameraRotationsVector);

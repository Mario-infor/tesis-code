#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <glm/gtc/quaternion.hpp>
#include <structsFile.h>
#include <Eigen/Dense>

#define IMU_ADDRESS "/dev/i2c-1" // Address of the IMU sensor.
#define	MATH_PI					3.1415926535 // Definition of variable pi.
#define	MATH_DEGREE_TO_RAD		(MATH_PI / 180.0) // Conversion from degrees to radians.
#define	MATH_RAD_TO_DEGREE		(180.0 / MATH_PI) // Conversion from radians to degrees.
#define ALPHA_ACC 0.7 // Alpha value for the accelerometer IIR.
#define ALPHA_GYRO 0.6 // Alpha value for the gyroscope IIR.

// Convert rotation vector to quaternion.
glm::quat convertOpencvRotVectToQuat(cv::Vec3d rotVect);

// Convert quaternion to rotation vector using glm and opencv.
cv::Vec3d QuatToRotVect(glm::quat quaternion);

// Convert quaternion to rotation vector using Eigen.
Eigen::Vector3d QuatToRotVectEigen(Eigen::Quaterniond quaternion);

// Create a hard copy of camera vector.
std::vector<CameraInput> hardCopyCameraVector(
    std::vector<CameraInput> cameraReadVector);

// Get rotation and translation from all frames.
std::vector<FrameMarkersData> getRotationTraslationFromAllFrames(
    std::vector<CameraInput> cameraReadVector,
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs);

// Get rotation and translation from one single frame.
FrameMarkersData getRotationTraslationFromFrame(
    CameraInput frame, 
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs);

// Prints Euler or Accel data from the IMU to the console.
void printIMUData();

// Draw axis on a frame using information from rvecs and tvecs.
void drawAxisOnFrame(
    std::vector<cv::Vec3d> rvecs,
    std::vector<cv::Vec3d> tvecs,
    cv::Mat frame,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs,
    std::string windowTitle);

// Get the antisymetric matrix from a vector.
Eigen::Matrix3d getWHat(const Eigen::Vector3d v);

Eigen::Matrix4d getGhi(const Eigen::Vector3d w, const Eigen::Vector3d v);

int getImuStartingIdexBaseOnCamera(std::vector<CameraInput> cameraReadVector,
 std::vector<ImuInputJetson> imuReadVector);

// convert euler angles to quaternion
Eigen::Quaternion<double> rotVecToQuat(Eigen::Vector3d euler);

void gnuPrintImuPreintegration(
    FILE *output,
    std::vector<Eigen::Vector3d> vectorOfPointsOne,
    std::vector<Eigen::Vector3d> vectorOfPointsTwo,
    std::vector<Eigen::Vector3d> vectorOfMarkers);

void gnuPrintImuCompareValues(
    FILE *output,
    std::vector<float> vectorOfPointsOne,
    std::vector<float> vectorOfPointsTwo);

// Normalize a rotation matrix converting it to Quaternion 
// and using Eigen's normalized() method.
Eigen::Matrix3d normalizeRotationMatrix(Eigen::Matrix3d matrix);

// Normalize a quaternion using Eigen's normalized() method 
// and transforming it to positive angle if necessary.
Eigen::Quaterniond normalizeQuaternion(Eigen::Quaterniond quat);

// Guarantee that the rotation matrix is orthonormal.
Eigen::Matrix3d GramSchmidt(Eigen::Matrix3d rotationMatrix);

// Project a vector u onto a vector v.
Eigen::Vector3d proj(Eigen::Vector3d u, Eigen::Vector3d v);

// Get the exponential matrix of a vector.
Eigen::Matrix3d matrixExp(Eigen::Vector3d gyroTimesDeltaT);

void normalizeDataSet(std::vector<Eigen::Vector3d> points, std::vector<float> &result, int variable);

cv::Mat convertEigenMatToOpencvMat(Eigen::MatrixXd eigenMat);
Eigen::MatrixXd convertOpencvMatToEigenMat(cv::Mat cvMat);

Eigen::Matrix<double, 3, 3> getCamRotMatFromRotVec(cv::Vec3d camRvec);

Eigen::Matrix<double, 4, 4> getGFromFrameMarkersData(FrameMarkersData frameMarkersData, int index);

Eigen::Vector3d getAngularVelocityFromTwoQuats(Eigen::Quaterniond q1, Eigen::Quaterniond q2, float deltaT);

void calculateHAndJacobian(
    cv::KalmanFilter KF,
    Eigen::Matrix<double, 4, 4> Gci,
    Eigen::Matrix<double, 4, 4> Gci_inv,
    Eigen::MatrixXd &h,
    Eigen::MatrixXd &H
);

Eigen::Matrix4d invertG(Eigen::Matrix4d G);

void fixStateQuaternion(cv::KalmanFilter &KF, std::string stateName);

void fixQuatEigen(Eigen::Quaterniond &q);

int getBaseMarkerIndex(std::vector<int> markerIds, int baseMarkerId);

std::vector<TransformBetweenMarkers> getAllTransformsBetweenMarkers(FrameMarkersData firstFrameMarkersData, Eigen::Matrix4d Gcm, int indexBaseMarker);

void applyIIRFilterToAccAndGyro(
    Eigen::Vector3d accReading,
    Eigen::Vector3d gyroReading,
    Eigen::Vector3d &accFiltered,
    Eigen::Vector3d &gyroFiltered);

void calculateBiasAccAndGyro(Eigen::Vector3d &accBiasVect, Eigen::Vector3d &gyroBiasVect);

Eigen::Vector3d multiplyVectorByG(Eigen::Matrix4d G, Eigen::Vector3d v);

#endif // UTILS_H
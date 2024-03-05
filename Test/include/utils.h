#include <opencv2/opencv.hpp>
#include <glm/gtc/quaternion.hpp>

// Convert rotation vector to quaternion.
glm::quat convertOpencvRotVectToQuat(cv::Vec3d rotVect);

// Convert quaternion to rotation vector.
cv::Vec3d convertQuatToOpencvRotVect(glm::quat quaternion);

// Create a hard copy of camera vector.
std::vector<CameraInput> hardCopyCameraVector(std::vector<CameraInput> cameraReadVector);
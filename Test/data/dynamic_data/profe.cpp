cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    791.330070013496,   0,                  409.12509669936657,
    0,                  595.8977808485818,  295.52098484167175,
    0,                  0,                  1);

cv::Mat distCoeffs = (cv::Mat_<double>(1, 4) << 0.12912260003585463, \
                                                -0.25388754628599064, \
                                                0.000943128326254423, \
                                                0.0018654081762782874);

cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);


struct FrameMarkersData
{
    std::vector<int> markerIds;
    std::vector<cv::Vec3d> rvecs;
    std::vector<cv::Vec3d> tvecs;
    std::vector<cv::Vec4d> qvecs;
};

struct CameraInput
{
    int index;
    int time;
    cv::Mat frame;

    CameraInput &operator=(const CameraInput &other)
    {
        if (this != &other)
        {
            index = other.index;
            time = other.time;
            frame = other.frame.clone();
        }
        return *this;
    }
};

void testAruco()
{
    std::map<int, Eigen::Matrix4d> transformsMap;
    std::vector<CameraInput> cameraData = readDataCamera();

    CameraInput CamMeasurement = cameraData.at(0);

    FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(CamMeasurement,
         dictionary, cameraMatrix, distCoeffs);

    drawAxisOnFrame(frameMarkersData.rvecs, frameMarkersData.tvecs,
                    CamMeasurement.frame, cameraMatrix, distCoeffs, "Camera Measurement");

    int indexBaseMarker = getBaseMarkerIndex(frameMarkersData.markerIds, BASE_MARKER_ID);
    Eigen::Matrix4d Gcm = getGFromFrameMarkersData(frameMarkersData, indexBaseMarker);

    getAllTransformsBetweenMarkers(frameMarkersData, Gcm, indexBaseMarker, transformsMap, false);
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

std::vector<FrameMarkersData> getRotationTraslationFromAllFrames(
    std::vector<CameraInput> cameraReadVector,
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs)
{
    std::vector<FrameMarkersData> frameMarkersDataVector;

    for (size_t i = 0; i < cameraReadVector.size(); i++)
    {
        FrameMarkersData frameMarkersData = getRotationTraslationFromFrame(cameraReadVector[i], dictionary, cameraMatrix, distCoeffs);
        frameMarkersDataVector.push_back(frameMarkersData);
    }

    return frameMarkersDataVector;
}

FrameMarkersData getRotationTraslationFromFrame(
    CameraInput frame,
    cv::Ptr<cv::aruco::Dictionary> dictionary,
    cv::Mat cameraMatrix,
    cv::Mat distCoeffs)
{
    FrameMarkersData frameMarkersData;

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(frame.frame, dictionary, markerCorners, markerIds);

    if (markerIds.size() > 0)
    {
        cv::aruco::drawDetectedMarkers(frame.frame, markerCorners, markerIds);

        std::vector<cv::Vec3d> rvecs, tvecs;

        cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.17, cameraMatrix, distCoeffs, rvecs, tvecs);

        frameMarkersData.markerIds = markerIds;
        frameMarkersData.rvecs = rvecs;
        frameMarkersData.tvecs = tvecs;
    }

    return frameMarkersData;
}

int getBaseMarkerIndex(std::vector<int> markerIds, int baseMarkerId)
{
    int baseMarkerIndex = -1;
    for(size_t i = 0; i < markerIds.size(); i++)
    {
        if(markerIds[i] == baseMarkerId)
        {
            baseMarkerIndex = i;
            break;
        }
    }
    return baseMarkerIndex;
}

Eigen::Matrix<double, 4, 4> getGFromFrameMarkersData(FrameMarkersData frameMarkersData, int index)
{
    cv::Mat camRotMat;
    cv::Rodrigues(frameMarkersData.rvecs[index], camRotMat);

    std::cout << "camRotMat: " << camRotMat << std::endl << std::endl;

    Eigen::Matrix<double, 3, 3> camRot;
    camRot <<
    camRotMat.at<double>(0, 0), camRotMat.at<double>(0, 1), camRotMat.at<double>(0, 2),
    camRotMat.at<double>(1, 0), camRotMat.at<double>(1, 1), camRotMat.at<double>(1, 2),
    camRotMat.at<double>(2, 0), camRotMat.at<double>(2, 1), camRotMat.at<double>(2, 2);

    std::cout << "camRot: " << std::endl << camRot << std::endl << std::endl;

    Eigen::Vector3d camT{frameMarkersData.tvecs[index].val[0], frameMarkersData.tvecs[index].val[1], frameMarkersData.tvecs[index].val[2]};
    
    std::cout << "camT: " << std::endl << camT << std::endl << std::endl;

    Eigen::Matrix<double, 4, 4> g;
    g.setIdentity();

    g.block<3,3>(0,0) = camRot;
    g.block<3,1>(0,3) = camT;

    std::cout << "g: " << std::endl << g << std::endl << std::endl;
    
    return g;
}

void getAllTransformsBetweenMarkers(
    FrameMarkersData firstFrameMarkersData,
    Eigen::Matrix4d Gcm,
    int indexBaseMarker,
    std::map<int, Eigen::Matrix4d> &oldCamMeasurementsMap,
    bool clearFile)
{
    int baseMarkerId = firstFrameMarkersData.markerIds[indexBaseMarker];

    for(size_t i = 0; i < firstFrameMarkersData.markerIds.size(); i++)
    {
        if(firstFrameMarkersData.markerIds[i] != baseMarkerId)
        {
            
            Eigen::Matrix4d gCamToMarker = getGFromFrameMarkersData(firstFrameMarkersData, i); 

            std::cout << "Gcm: " << std::endl << Gcm << std::endl << std::endl;
            std::cout << "gCamToMarker: " << std::endl << gCamToMarker << std::endl << std::endl;
            
            oldCamMeasurementsMap[firstFrameMarkersData.markerIds[i]] =  invertG(Gcm) * gCamToMarker;

            std::cout << "Transform: " << std::endl << oldCamMeasurementsMap[firstFrameMarkersData.markerIds[i]] << std::endl << std::endl;

            if(firstFrameMarkersData.markerIds[i] == 38)
            {
                
                std::cout << oldCamMeasurementsMap[firstFrameMarkersData.markerIds[i]].determinant() << std::endl << std::endl;
            }

            transformWrite(oldCamMeasurementsMap[firstFrameMarkersData.markerIds[i]], firstFrameMarkersData.markerIds[i], clearFile);
        }
    }
}

void transformWrite(const Eigen::Matrix4d transform, int fileName, const bool clearFile)
{   
    std::string buildedFileName = dirPointsFolder + std::to_string(fileName);
    std::ofstream file(buildedFileName, std::ios::app);

    if (clearFile)
    {
        std::ofstream file(buildedFileName, std::ios::out);
    }

    if (file.is_open())
    {
        for (int i = 0; i < transform.rows(); i++)
        {
            for (int j = 0; j < transform.cols(); j++)
            {
                file << transform(i, j);

                if (j < transform.cols())
                {
                    file << ",";
                }
                
            }
        }

        file << std::endl;

        file.close();
    }
    else 
    {
        std::cout << "Error openning file." << std::endl;
    }
}


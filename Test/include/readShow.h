// Struct to store information about each frame saved.
struct CameraInput
{
    cv::Mat frame;
};

// Pipeline for camera on JEtson Board.
std::string gstreamerPipelineReadShow (
    int capture_width,
    int capture_height,
    int display_width,
    int display_height,
    int framerate,
    int flip_method);
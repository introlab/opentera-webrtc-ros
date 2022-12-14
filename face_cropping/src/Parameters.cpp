#include "face_cropping/Parameters.h"

#include <type_traits>

using namespace face_cropping;

template<typename T>
static T getParam(const std::string& name, const T& defaultValue)
{
    ros::NodeHandle nh{"~"};
    T value;
    nh.param<T>(name, value, defaultValue);
    return value;
}


Parameters::Parameters(ros::NodeHandle& nodeHandle)
{
    m_refreshRate = getParam<double>("refresh_rate", 30.0);
    m_width = getParam<int>("width", 400);
    m_height = getParam<int>("height", 400);

    m_secondsWithoutDetection = getParam<int>("seconds_without_detection", 5);

    m_framesUsedForStabilizer = getParam<int>("frames_used_for_stabilizer", 15);

    m_minWidthChange = getParam<float>("min_width_change", 0.2);
    m_minHeightChange = getParam<float>("min_height_change", 0.4);
    m_minXChange = getParam<float>("min_x_change", 0.1);
    m_minYChange = getParam<float>("min_y_change", 0.1);

    m_rightMargin = getParam<float>("right_margin", 0.2);
    m_leftMargin = getParam<float>("left_margin", 0.2);
    m_topMargin = getParam<float>("top_margin", 0.2);
    m_bottomMargin = getParam<float>("bottom_margin", 0.25);

    m_isPeerImage = getParam<bool>("is_peer_image", false);

    m_haarCascadePath = getParam<std::string>(
        "haar_cascade_path",
        ros::package::getPath("face_cropping") + "/models/haarcascade_frontalface_default.xml");
    m_lbpCascadePath = getParam<std::string>(
        "lbp_cascade_path",
        ros::package::getPath("face_cropping") + "/models/lbpcascade_frontalface.xml");

    m_useLbp = getParam<bool>("use_lbp", false);

    m_detectionFrames = getParam<int>("detection_frames", 1);
    m_detectionScale = getParam<double>("detection_scale", 1);
    m_minFaceWidth = getParam<int>("min_face_width", 50);
    m_minFaceHeight = getParam<int>("min_face_height", 50);

    m_maxSizeStep = getParam<double>("max_size_step", 0.05);
    m_maxPositionStep = getParam<double>("max_position_step", 0.05);

    m_faceStoringFrames = getParam<int>("face_storing_frames", 10);
    m_validFaceMinTime = getParam<double>("valid_face_min_time", 0.75);

    m_highlightDetections = getParam<bool>("highlight_detections", false);
}

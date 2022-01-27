#include "map_image_generator/Parameters.h"

#include "map_image_generator/utils.h"

#include <sstream>
#include <type_traits>

using namespace map_image_generator;
using namespace std;

namespace internal
{
    // Not defined for T: compilation fails if T is not cv::Scalar or cv::Vec3b
    template <typename T>
    T parseColor(const std::string& color);

    template <>
    cv::Scalar parseColor<cv::Scalar>(const std::string& color)
    {
        int r, g, b, a;
        stringstream ss(color);
        ss >> r >> g >> b >> a;

        return {static_cast<double>(b), static_cast<double>(g), static_cast<double>(r),
                static_cast<double>(a)};
    }

    template <>
    cv::Vec3b parseColor<cv::Vec3b>(const std::string& color)
    {
        int r, g, b;
        stringstream ss(color);
        ss >> r >> g >> b;

        return {static_cast<uchar>(b), static_cast<uchar>(g), static_cast<uchar>(r)};
    }

    // Compilation fails if T is not cv::Scalar or cv::Vec3b because of parseColor<T>
    template <typename T>
    T makeColor(const std::string& name, const std::string& defaultColor)
    {
        ros::NodeHandle nh{"~"};
        std::string colorString;
        nh.param<std::string>(name, colorString, defaultColor);
        return parseColor<T>(colorString);
    }

    // Not defined if T is cv::Scalar or cv::Vec3b
    template <typename T>
    std::enable_if_t<
        !std::is_same<T, cv::Scalar>::value && !std::is_same<T, cv::Vec3b>::value, T>
    getParam(const std::string& name, const T& defaultValue)
    {
        ros::NodeHandle nh{"~"};
        T value;
        nh.param<T>(name, value, defaultValue);
        return value;
    }

    // Defined only if T is cv::Scalar or cv::Vec3b
    template <typename T>
    std::enable_if_t<
        std::is_same<T, cv::Scalar>::value || std::is_same<T, cv::Vec3b>::value, T>
    getParam(const std::string& name, const std::string& defaultValue)
    {
        return makeColor<T>(name, defaultValue);
    }
}

Parameters::Parameters(ros::NodeHandle& nodeHandle) : m_scaleFactor{1.0}
{
    using namespace internal;

    m_refreshRate = getParam<double>("refresh_rate", 1.0);
    m_resolution = getParam<int>("resolution", 50);
    m_width = getParam<int>("width", 30);
    m_height = getParam<int>("height", 15);
    m_xOrigin = getParam<int>("x_origin", (m_width * m_resolution) / 2);
    m_yOrigin = getParam<int>("y_origin", (m_height * m_resolution) / 2);
    m_robotVerticalOffset = getParam<int>("robot_vertical_offset", 0);

    m_robotFrameId = getParam<std::string>("robot_frame_id", "base_footprint");
    m_mapFrameId = getParam<std::string>("map_frame_id", "map");

    m_drawOccupancyGrid = getParam<bool>("draw_occupancy_grid", true);
    m_drawGlobalPath = getParam<bool>("draw_global_path", true);
    m_drawRobot = getParam<bool>("draw_robot", true);
    m_drawGoals = getParam<bool>("draw_goals", true);
    m_drawLaserScan = getParam<bool>("draw_laser_scan", true);
    m_drawLabels = getParam<bool>("draw_labels", true);
    m_drawSoundSources = getParam<bool>("draw_sound_sources", true);

    m_wallColor = getParam<cv::Vec3b>("wall_color", "0 0 0");
    m_freeSpaceColor = getParam<cv::Vec3b>("free_space_color", "255 255 255");
    m_unknownSpaceColor = getParam<cv::Vec3b>("unknown_space_color", "175 175 175");
    m_globalPathColor = getParam<cv::Scalar>("global_path_color", "0 255 0 255");
    m_robotColor = getParam<cv::Scalar>("robot_color", "0 0 255 255");
    m_goalColor = getParam<cv::Scalar>("goal_color", "0 175 0 255");
    m_laserScanColor = getParam<cv::Scalar>("laser_scan_color", "255 0 0 255");
    m_textColor = getParam<cv::Scalar>("text_color", "255 255 255 255");
    m_labelColor = getParam<cv::Scalar>("label_color", "255 0 255 255");
    m_soundSourceColor = getParam<cv::Scalar>("sound_source_color", "255 255 0 255");

    m_globalPathThickness = getParam<int>("global_path_thickness", 3);
    m_robotSize = getParam<int>("robot_size", 30);
    m_goalSize = getParam<int>("goal_size", 20);
    m_laserScanSize = getParam<int>("laser_scan_size", 6);
    m_labelSize = getParam<int>("label_size", 35);
    m_soundSourceSize = getParam<int>("sound_source_size", 50);

    m_centeredRobot = getParam<bool>("centered_robot", true);

    validateParameters();
}

Parameters::~Parameters() = default;

void Parameters::validateParameters()
{
    int maxOffset = static_cast<int>(std::floor((m_height * m_resolution - 1) / 2.0));
    int minOffset = static_cast<int>(std::floor(-(m_height * m_resolution - 1) / 2.0));
    if (m_centeredRobot
        && (m_robotVerticalOffset > maxOffset || m_robotVerticalOffset < minOffset))
    {
        int oldOffset = m_robotVerticalOffset;
        m_robotVerticalOffset = static_cast<int>(std::floor(
            sign(m_robotVerticalOffset) * (m_height * m_resolution - 1) / 2.0));
        ROS_WARN(
            "Robot vertical offset is [%d], which is out of inclusive range [%d, %d]. "
            "It will be set to [%d], which is the maximum value based on the sign.",
            oldOffset, minOffset, maxOffset, m_robotVerticalOffset);
    }
}

const std::string& Parameters::refFrameId() const
{
    if (m_centeredRobot)
    {
        return m_robotFrameId;
    }
    else
    {
        return m_mapFrameId;
    }
}

void Parameters::setCenteredRobot(bool centeredRobot)
{
    m_centeredRobot = centeredRobot;
    m_scaleFactor = 1.0;
}

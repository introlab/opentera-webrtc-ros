#include "map_image_generator/Parameters.h"

#include "map_image_generator/utils.h"

#include <sstream>
#include <type_traits>

using namespace map_image_generator;
using namespace std;

namespace internal
{
    // Not defined for T: compilation fails if T is not cv::Scalar or cv::Vec3b
    template<typename T>
    T parseColor(const std::string& color);

    template<>
    cv::Scalar parseColor<cv::Scalar>(const std::string& color)
    {
        int r, g, b, a;
        stringstream ss(color);
        ss >> r >> g >> b >> a;

        return {static_cast<double>(b), static_cast<double>(g), static_cast<double>(r), static_cast<double>(a)};
    }

    template<>
    cv::Vec3b parseColor<cv::Vec3b>(const std::string& color)
    {
        int r, g, b;
        stringstream ss(color);
        ss >> r >> g >> b;

        return {static_cast<uchar>(b), static_cast<uchar>(g), static_cast<uchar>(r)};
    }

    // Compilation fails if T is not cv::Scalar or cv::Vec3b because of parseColor<T>
    template<typename T>
    T makeColor(rclcpp::Node& node, const std::string& name, const std::string& defaultColor)
    {
        return parseColor<T>(node.declare_parameter(name, defaultColor));
    }

    // Not defined if T is cv::Scalar or cv::Vec3b
    template<typename T>
    std::enable_if_t<!std::is_same<T, cv::Scalar>::value && !std::is_same<T, cv::Vec3b>::value, T>
        getParam(rclcpp::Node& node, const std::string& name, const T& defaultValue)
    {
        return node.declare_parameter(name, defaultValue);
        ;
    }

    // Defined only if T is cv::Scalar or cv::Vec3b
    template<typename T>
    std::enable_if_t<std::is_same<T, cv::Scalar>::value || std::is_same<T, cv::Vec3b>::value, T>
        getParam(rclcpp::Node& node, const std::string& name, const std::string& defaultValue)
    {
        return makeColor<T>(node, name, defaultValue);
    }
}

Parameters::Parameters(rclcpp::Node& node) : m_node{node}, m_scaleFactor{1.0}
{
    using namespace internal;

    m_refreshRate = getParam<double>(m_node, "refresh_rate", 1.0);
    m_resolution = getParam<int>(m_node, "resolution", 50);
    m_width = getParam<int>(m_node, "width", 30);
    m_height = getParam<int>(m_node, "height", 15);
    m_xOrigin = getParam<int>(m_node, "x_origin", (m_width * m_resolution) / 2);
    m_yOrigin = getParam<int>(m_node, "y_origin", (m_height * m_resolution) / 2);
    m_robotVerticalOffset = getParam<int>(m_node, "robot_vertical_offset", 0);
    m_soundSourceRange = getParam<float>(m_node, "sound_source_range", 1);
    m_soundSourceMaxRange = getParam<float>(m_node, "sound_source_max_range", 2);

    m_robotFrameId = getParam<std::string>(m_node, "robot_frame_id", "base_footprint");
    m_mapFrameId = getParam<std::string>(m_node, "map_frame_id", "map");

    m_drawOccupancyGrid = getParam<bool>(m_node, "draw_occupancy_grid", true);
    m_drawGlobalPath = getParam<bool>(m_node, "draw_global_path", true);
    m_drawRobot = getParam<bool>(m_node, "draw_robot", true);
    m_drawGoals = getParam<bool>(m_node, "draw_goals", true);
    m_drawLaserScan = getParam<bool>(m_node, "draw_laser_scan", true);
    m_drawLabels = getParam<bool>(m_node, "draw_labels", true);
    m_drawSoundSources = getParam<bool>(m_node, "draw_sound_sources", true);

    m_wallColor = getParam<cv::Vec3b>(m_node, "wall_color", "0 0 0");
    m_freeSpaceColor = getParam<cv::Vec3b>(m_node, "free_space_color", "255 255 255");
    m_unknownSpaceColor = getParam<cv::Vec3b>(m_node, "unknown_space_color", "175 175 175");
    m_globalPathColor = getParam<cv::Scalar>(m_node, "global_path_color", "0 255 0 255");
    m_robotColor = getParam<cv::Scalar>(m_node, "robot_color", "0 0 255 255");
    m_goalColor = getParam<cv::Scalar>(m_node, "goal_color", "0 175 0 255");
    m_laserScanColor = getParam<cv::Scalar>(m_node, "laser_scan_color", "255 0 0 255");
    m_textColor = getParam<cv::Scalar>(m_node, "text_color", "255 255 255 255");
    m_labelColor = getParam<cv::Scalar>(m_node, "label_color", "255 0 255 255");
    m_soundSourceColorFull = getParam<cv::Scalar>(m_node, "sound_source_color_full", "255 0 0 255");
    m_soundSourceColorDim = getParam<cv::Scalar>(m_node, "sound_source_color_dim", "0 255 0 255");


    m_globalPathThickness = getParam<int>(m_node, "global_path_thickness", 3);
    m_robotSize = getParam<int>(m_node, "robot_size", 30);
    m_goalSize = getParam<int>(m_node, "goal_size", 20);
    m_laserScanSize = getParam<int>(m_node, "laser_scan_size", 6);
    m_labelSize = getParam<int>(m_node, "label_size", 35);
    m_soundSourceSize = getParam<int>(m_node, "sound_source_size", 50);

    m_centeredRobot = getParam<bool>(m_node, "centered_robot", true);

    validateParameters();
}

Parameters::~Parameters() = default;

void Parameters::validateParameters()
{
    int maxOffset = static_cast<int>(std::floor((m_height * m_resolution - 1) / 2.0));
    int minOffset = static_cast<int>(std::floor(-(m_height * m_resolution - 1) / 2.0));
    if (m_centeredRobot && (m_robotVerticalOffset > maxOffset || m_robotVerticalOffset < minOffset))
    {
        int oldOffset = m_robotVerticalOffset;
        m_robotVerticalOffset =
            static_cast<int>(std::floor(sign(m_robotVerticalOffset) * (m_height * m_resolution - 1) / 2.0));
        RCLCPP_WARN(
            m_node.get_logger(),
            "Robot vertical offset is [%d], which is out of inclusive range [%d, %d]. "
            "It will be set to [%d], which is the maximum value based on the sign.",
            oldOffset,
            minOffset,
            maxOffset,
            m_robotVerticalOffset);
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

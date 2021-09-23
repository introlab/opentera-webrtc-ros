#include "map_image_generator/Parameters.h"

#include <sstream>

using namespace map_image_generator;
using namespace std;

Parameters::Parameters(ros::NodeHandle& nodeHandle)
{
    nodeHandle.param("refresh_rate", m_refreshRate, 1.0);
    nodeHandle.param("resolution", m_resolution, 50);
    nodeHandle.param("width", m_width, 30);
    nodeHandle.param("height", m_height, 15);
    nodeHandle.param("x_origin", m_xOrigin, (m_width * m_resolution) / 2);
    nodeHandle.param("y_origin", m_yOrigin, (m_height * m_resolution) / 2);

    nodeHandle.param("robot_frame_id", m_robotFrameId, string("base_footprint"));
    nodeHandle.param("map_frame_id", m_mapFrameId, string("map"));

    nodeHandle.param("draw_occupancy_grid", m_drawOccupancyGrid, true);
    nodeHandle.param("draw_global_path", m_drawGlobalPath, true);
    nodeHandle.param("draw_robot", m_drawRobot, true);
    nodeHandle.param("draw_goal", m_drawGoal, true);
    nodeHandle.param("draw_laser_scan", m_drawLaserScan, true);

    string wallColorString;
    string freeSpaceColorString;
    string unknownSpaceColorString;
    string globalPathColorString;
    string robotColorString;
    string goalColorString;
    string laserScanColorString;

    nodeHandle.param("wall_color", wallColorString, string("0 0 0"));
    nodeHandle.param("free_space_color", freeSpaceColorString, string("255 255 255"));
    nodeHandle.param("unknown_space_color", unknownSpaceColorString, string("175 175 175"));
    nodeHandle.param("global_path_color", globalPathColorString, string("0 255 0 255"));
    nodeHandle.param("robot_color", robotColorString, string("0 0 255 255"));
    nodeHandle.param("goal_color", goalColorString, string("0 175 0 255"));
    nodeHandle.param("laser_scan_color", laserScanColorString, string("255 0 0 255"));

    m_wallColor = parseColorVec3b(wallColorString);
    m_freeSpaceColor = parseColorVec3b(freeSpaceColorString);
    m_unknownSpaceColor = parseColorVec3b(unknownSpaceColorString);
    m_globalPathColor = parseColorScalar(globalPathColorString);
    m_robotColor = parseColorScalar(robotColorString);
    m_goalColor = parseColorScalar(goalColorString);
    m_laserScanColor = parseColorScalar(laserScanColorString);

    nodeHandle.param("global_path_thickness", m_globalPathThickness, 3);
    nodeHandle.param("robot_size", m_robotSize, 30);
    nodeHandle.param("goal_size", m_goalSize, 20);
    nodeHandle.param("laser_scan_size", m_laserScanSize, 6);
}

Parameters::~Parameters()
{
}

cv::Vec3b Parameters::parseColorVec3b(const std::string& color)
{
    int r, g, b;
    stringstream ss(color);
    ss >> r >> g >> b;

    return cv::Vec3b(b, g, r);
}

cv::Scalar Parameters::parseColorScalar(const std::string& color)
{
    int r, g, b, a;
    stringstream ss(color);
    ss >> r >> g >> b >> a;

    return cv::Scalar(b, g, r, a);
}

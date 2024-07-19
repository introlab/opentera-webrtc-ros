#include "map_image_generator/MapImageGenerator.h"

#include "map_image_generator/drawers/GlobalPathImageDrawer.h"
#include "map_image_generator/drawers/GoalImageDrawer.h"
#include "map_image_generator/drawers/LabelImageDrawer.h"
#include "map_image_generator/drawers/LaserScanImageDrawer.h"
#include "map_image_generator/drawers/OccupancyGridImageDrawer.h"
#include "map_image_generator/drawers/RobotImageDrawer.h"
#include "map_image_generator/drawers/SoundSourceImageDrawer.h"

using namespace map_image_generator;
using namespace std;

MapImageGenerator::MapImageGenerator(Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer)
    : m_parameters(parameters),
      m_node(node),
      m_tfBuffer(tfBuffer)
{
    // The order of the drawers is important as it determines the layering of the
    // elements.
    if (m_parameters.drawOccupancyGrid())
    {
        m_drawers.push_back(std::make_unique<OccupancyGridImageDrawer>(parameters, node, m_tfBuffer));
    }

    if (m_parameters.drawGlobalPath())
    {
        m_drawers.push_back(std::make_unique<GlobalPathImageDrawer>(m_parameters, node, m_tfBuffer));
    }

    if (m_parameters.drawRobot())
    {
        m_drawers.push_back(std::make_unique<RobotImageDrawer>(m_parameters, node, m_tfBuffer));
    }

    if (m_parameters.drawLabels())
    {
        m_drawers.push_back(std::make_unique<LabelImageDrawer>(m_parameters, node, m_tfBuffer));
    }

    if (m_parameters.drawGoals())
    {
        m_drawers.push_back(std::make_unique<GoalImageDrawer>(m_parameters, node, m_tfBuffer));
    }

    if (m_parameters.drawLaserScan())
    {
        m_drawers.push_back(std::make_unique<LaserScanImageDrawer>(m_parameters, node, m_tfBuffer));
    }

    if (m_parameters.drawSoundSources())
    {
        m_drawers.push_back(std::make_unique<SoundSourceImageDrawer>(m_parameters, node, m_tfBuffer));
    }

    int imageWidth = parameters.resolution() * parameters.width();
    int imageHeight = parameters.resolution() * parameters.height();


    m_cvImage.header.stamp = m_node.now();
    m_cvImage.header.frame_id = "map_image";

    m_cvImage.encoding = sensor_msgs::image_encodings::BGR8;
    m_cvImage.image = cv::Mat(imageHeight, imageWidth, CV_8UC3);
}

MapImageGenerator::~MapImageGenerator() = default;

void MapImageGenerator::generate(sensor_msgs::msg::Image& sensorImage)
{
    m_cvImage.header.stamp = m_node.now();
    m_cvImage.image = m_parameters.unknownSpaceColor();

    for (auto& drawer : m_drawers)
    {
        drawer->draw(m_cvImage.image);
    }

    m_cvImage.toImageMsg(sensorImage);
}

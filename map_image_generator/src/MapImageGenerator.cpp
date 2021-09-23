#include "map_image_generator/MapImageGenerator.h"

#include "map_image_generator/drawers/OccupancyGridImageDrawer.h"
#include "map_image_generator/drawers/GlobalPathImageDrawer.h"
#include "map_image_generator/drawers/RobotImageDrawer.h"
#include "map_image_generator/drawers/GoalImageDrawer.h"
#include "map_image_generator/drawers/LaserScanImageDrawer.h"

using namespace map_image_generator;
using namespace std;

MapImageGenerator::MapImageGenerator(const Parameters& parameters,
    ros::NodeHandle& nodeHandle,
    tf::TransformListener& tfListener) :
    m_parameters(parameters), m_nodeHandle(nodeHandle), m_tfListener(tfListener)
{
    if (m_parameters.drawOccupancyGrid())
    {
        m_drawers.push_back(new OccupancyGridImageDrawer(m_parameters, nodeHandle, m_tfListener));
    }

    if (m_parameters.drawGlobalPath())
    {
        m_drawers.push_back(new GlobalPathImageDrawer(m_parameters, nodeHandle, m_tfListener));
    }

    if (m_parameters.drawRobot())
    {
        m_drawers.push_back(new RobotImageDrawer(m_parameters, nodeHandle, m_tfListener));
    }

    if (m_parameters.drawGoal())
    {
        m_drawers.push_back(new GoalImageDrawer(m_parameters, nodeHandle, m_tfListener));
    }

    if (m_parameters.drawLaserScan())
    {
        m_drawers.push_back(new LaserScanImageDrawer(m_parameters, nodeHandle, m_tfListener));
    }

    int imageWidth = parameters.resolution() * parameters.width();
    int imageHeight = parameters.resolution() * parameters.height();

    m_cvImage.header.seq = 0;
    m_cvImage.header.stamp = ros::Time::now();
    m_cvImage.header.frame_id = "map_image";

    m_cvImage.encoding = sensor_msgs::image_encodings::BGR8;        
    m_cvImage.image = cv::Mat(imageHeight, imageWidth, CV_8UC3);
}

MapImageGenerator::~MapImageGenerator()
{
    for (vector<ImageDrawer*>::iterator it = m_drawers.begin(); it != m_drawers.end(); ++it)
    {
        delete *it;
    }
}

void MapImageGenerator::generate(sensor_msgs::Image& sensorImage)
{
    m_cvImage.header.seq++;
    m_cvImage.header.stamp = ros::Time::now();
    m_cvImage.image = m_parameters.unknownSpaceColor();

    for (vector<ImageDrawer*>::iterator it = m_drawers.begin(); it != m_drawers.end(); ++it)
    {
        (*it)->draw(m_cvImage.image);
    }

    cv::flip(m_cvImage.image, m_cvImage.image, 1);
    m_cvImage.toImageMsg(sensorImage);
}
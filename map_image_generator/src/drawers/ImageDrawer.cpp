#include "map_image_generator/drawers/ImageDrawer.h"

using namespace map_image_generator;

ImageDrawer::ImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                         tf::TransformListener& tfListener)
    : m_parameters(parameters), m_nodeHandle(nodeHandle), m_tfListener(tfListener)
{
}

ImageDrawer::~ImageDrawer() {}

void ImageDrawer::convertTransformToMapCoordinates(const tf::Transform& transform, int& x,
                                                   int& y)
{
    x = static_cast<int>(transform.getOrigin().getX() * m_parameters.resolution()
                         + m_parameters.xOrigin());
    y = static_cast<int>(transform.getOrigin().getY() * m_parameters.resolution()
                         + m_parameters.yOrigin());
}

void ImageDrawer::convertTransformToInputMapCoordinates(
    const tf::Transform& transform, const nav_msgs::MapMetaData& mapInfo, int& x, int& y)
{
    x = static_cast<int>((transform.getOrigin().getX() - mapInfo.origin.position.x)
                         / mapInfo.resolution);
    y = static_cast<int>((transform.getOrigin().getY() - mapInfo.origin.position.y)
                         / mapInfo.resolution);
}

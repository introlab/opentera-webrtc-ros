#include "map_image_generator/drawers/ImageDrawer.h"

using namespace map_image_generator;

ImageDrawer::ImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle, tf::TransformListener& tfListener) :
    m_parameters(parameters), m_nodeHandle(nodeHandle), m_tfListener(tfListener)
{
}

ImageDrawer::~ImageDrawer()
{
}

void ImageDrawer::convertTransformToMapCoordinates(const tf::Transform& transform, int& x, int& y)
{
    x = static_cast<int>(transform.getOrigin().getX() * m_parameters.resolution() + m_parameters.xOrigin());
    y = static_cast<int>(transform.getOrigin().getY() * m_parameters.resolution() + m_parameters.yOrigin());
}
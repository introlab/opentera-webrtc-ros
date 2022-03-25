#include "map_image_generator/drawers/ImageDrawer.h"

using namespace map_image_generator;

ImageDrawer::ImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle, tf::TransformListener& tfListener)
    : m_parameters(parameters),
      m_nodeHandle(nodeHandle),
      m_tfListener(tfListener)
{
}

ImageDrawer::~ImageDrawer() = default;

void ImageDrawer::convertTransformToMapCoordinates(const tf::Transform& transform, int& x, int& y) const
{
    x = static_cast<int>(
        transform.getOrigin().getX() * m_parameters.resolution() * m_parameters.scaleFactor() + m_parameters.xOrigin());
    y = static_cast<int>(
        transform.getOrigin().getY() * m_parameters.resolution() * m_parameters.scaleFactor() + m_parameters.yOrigin() +
        m_parameters.robotVerticalOffset());
}

void ImageDrawer::convertTransformToInputMapCoordinates(
    const tf::Transform& transform,
    const nav_msgs::MapMetaData& mapInfo,
    int& x,
    int& y) const
{
    x = static_cast<int>((transform.getOrigin().getX() - mapInfo.origin.position.x) * m_parameters.resolution());
    y = static_cast<int>((transform.getOrigin().getY() - mapInfo.origin.position.y) * m_parameters.resolution());
}

void ImageDrawer::convertInputMapCoordinatesToTransform(
    int x,
    int y,
    const nav_msgs::MapMetaData& mapInfo,
    tf::Transform& transform) const
{
    transform.setOrigin(tf::Vector3(
        static_cast<double>(x) / m_parameters.resolution() + mapInfo.origin.position.x,
        static_cast<double>(y) / m_parameters.resolution() + mapInfo.origin.position.y,
        0.0));
}

std::optional<tf::Transform> ImageDrawer::getTransformInRef(const std::string& frameId) const
{
    tf::StampedTransform transform;

    try
    {
        m_tfListener.lookupTransform(m_parameters.refFrameId(), frameId, ros::Time(0), transform);
    }
    catch (tf::TransformException& ex)
    {
        ROS_ERROR("%s", ex.what());
        return {};
    }

    return transform;
}

void ImageDrawer::adjustTransformForRobotRef(tf::Transform& transform) const
{
    transform.getOrigin().setX(transform.getOrigin().getX() * -1);
    flipYawOnY(transform);

    adjustTransformAngleForRobotRef(transform);
}


void ImageDrawer::adjustTransformAngleForRobotRef(tf::Transform& transform) const
{
    if (m_parameters.centeredRobot())
    {
        double yaw = tf::getYaw(transform.getRotation());
        yaw += M_PI_2;
        transform.setRotation(tf::createQuaternionFromYaw(yaw));

        auto x = transform.getOrigin().getX();
        auto y = transform.getOrigin().getY();
        transform.getOrigin().setX(-y);
        transform.getOrigin().setY(x);
    }
}

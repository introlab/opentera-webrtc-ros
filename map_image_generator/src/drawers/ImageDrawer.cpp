#include "map_image_generator/drawers/ImageDrawer.h"

#include <tf2/utils.h>
#include <tf2/transform_datatypes.h>

#include "map_image_generator/utils.h"

using namespace map_image_generator;

ImageDrawer::ImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer)
    : m_parameters(parameters),
      m_node(node),
      m_tfBuffer(tfBuffer)
{
}

ImageDrawer::~ImageDrawer() = default;

void ImageDrawer::convertTransformToMapCoordinates(const tf2::Transform& transform, int& x, int& y) const
{
    x = static_cast<int>(
        transform.getOrigin().getX() * m_parameters.resolution() * m_parameters.scaleFactor() + m_parameters.xOrigin());
    y = static_cast<int>(
        transform.getOrigin().getY() * m_parameters.resolution() * m_parameters.scaleFactor() + m_parameters.yOrigin() +
        m_parameters.robotVerticalOffset());
}

void ImageDrawer::convertTransformToInputMapCoordinates(
    const tf2::Transform& transform,
    const nav_msgs::msg::MapMetaData& mapInfo,
    int& x,
    int& y) const
{
    x = static_cast<int>((transform.getOrigin().getX() - mapInfo.origin.position.x) * m_parameters.resolution());
    y = static_cast<int>((transform.getOrigin().getY() - mapInfo.origin.position.y) * m_parameters.resolution());
}

void ImageDrawer::convertInputMapCoordinatesToTransform(
    int x,
    int y,
    const nav_msgs::msg::MapMetaData& mapInfo,
    tf2::Transform& transform) const
{
    transform.setOrigin(tf2::Vector3(
        static_cast<double>(x) / m_parameters.resolution() + mapInfo.origin.position.x,
        static_cast<double>(y) / m_parameters.resolution() + mapInfo.origin.position.y,
        0.0));
}

std::optional<tf2::Transform> ImageDrawer::getTransformInRef(const std::string& frameId) const
{
    tf2::Stamped<tf2::Transform> transform;

    try
    {
        auto transformMsg = m_tfBuffer.lookupTransform(m_parameters.refFrameId(), frameId, tf2::TimePointZero);
        tf2::fromMsg(transformMsg, transform);
    }
    catch (const tf2::TransformException& ex)
    {
        RCLCPP_ERROR(m_node.get_logger(), "%s", ex.what());
        return {};
    }

    return transform;
}

void ImageDrawer::adjustTransformForRobotRef(tf2::Transform& transform) const
{
    transform.getOrigin().setX(transform.getOrigin().getX() * -1);
    flipYawOnY(transform);

    adjustTransformAngleForRobotRef(transform);
}


void ImageDrawer::adjustTransformAngleForRobotRef(tf2::Transform& transform) const
{
    if (m_parameters.centeredRobot())
    {
        double yaw = tf2::getYaw(transform.getRotation());
        yaw += M_PI_2;
        transform.setRotation(createQuaternionFromYaw(yaw));

        auto x = transform.getOrigin().getX();
        auto y = transform.getOrigin().getY();
        transform.getOrigin().setX(-y);
        transform.getOrigin().setY(x);
    }
}

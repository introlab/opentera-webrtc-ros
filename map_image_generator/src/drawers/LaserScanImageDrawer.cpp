#include "map_image_generator/drawers/LaserScanImageDrawer.h"

#include <cmath>
#include <tf2_ros/buffer.h>

using namespace map_image_generator;
using namespace std;

LaserScanImageDrawer::LaserScanImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer)
    : ImageDrawer(parameters, node, tfBuffer),
      m_laserScanSubscriber{m_node.create_subscription<sensor_msgs::msg::LaserScan>(
          "laser_scan",
          1,
          bind_this<sensor_msgs::msg::LaserScan>(this, &LaserScanImageDrawer::laserScanCallback))}
{
}

LaserScanImageDrawer::~LaserScanImageDrawer() = default;

void LaserScanImageDrawer::draw(cv::Mat& image)
{
    if (!m_lastLaserScan)
    {
        return;
    }

    auto tf = getTransformInRef(m_lastLaserScan->header.frame_id);
    if (tf)
    {
        drawLaserScan(image, *tf);
    }
}

void LaserScanImageDrawer::laserScanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& laserScan)
{
    m_lastLaserScan = laserScan;
}


void LaserScanImageDrawer::drawLaserScan(cv::Mat& image, tf2::Transform& transform)
{
    float angle = m_lastLaserScan->angle_min;
    for (const auto& range : m_lastLaserScan->ranges)
    {
        if (m_lastLaserScan->range_min <= range && range <= m_lastLaserScan->range_max)
        {
            drawRange(image, transform, range, angle);
        }

        angle += m_lastLaserScan->angle_increment;
    }
}

void LaserScanImageDrawer::drawRange(cv::Mat& image, tf2::Transform& transform, float range, float angle)
{
    tf2::Transform rangePose{tf2::Quaternion(0, 0, 0, 0), tf2::Vector3(range * cos(angle), range * sin(angle), 0)};
    rangePose = transform * rangePose;
    adjustTransformForRobotRef(rangePose);

    const cv::Scalar& color = m_parameters.laserScanColor();
    int size = m_parameters.laserScanSize();
    int halfSize = size / 2;

    int centerX, centerY;
    convertTransformToMapCoordinates(rangePose, centerX, centerY);

    cv::Point p1(centerX - halfSize, centerY - halfSize);
    cv::Point p2(centerX + halfSize, centerY + halfSize);

    cv::rectangle(image, p1, p2, color, cv::FILLED);
}

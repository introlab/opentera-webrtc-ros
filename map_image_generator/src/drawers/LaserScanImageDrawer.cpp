#include "map_image_generator/drawers/LaserScanImageDrawer.h"

#include <cmath>
#include <tf/tf.h>

using namespace map_image_generator;
using namespace std;

LaserScanImageDrawer::LaserScanImageDrawer(const Parameters& parameters,
                                           ros::NodeHandle& nodeHandle,
                                           tf::TransformListener& tfListener)
    : ImageDrawer(parameters, nodeHandle, tfListener)
{
    m_laserScanSubscriber = m_nodeHandle.subscribe(
        "laser_scan", 1, &LaserScanImageDrawer::laserScanCallback, this);
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

void LaserScanImageDrawer::laserScanCallback(
    const sensor_msgs::LaserScan::ConstPtr& laserScan)
{
    m_lastLaserScan = laserScan;
}


void LaserScanImageDrawer::drawLaserScan(cv::Mat& image, tf::Transform& transform)
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

void LaserScanImageDrawer::drawRange(cv::Mat& image, tf::Transform& transform,
                                     float range, float angle)
{
    tf::Pose rangePose(tf::Quaternion(0, 0, 0, 0),
                       tf::Vector3(range * cos(angle), range * sin(angle), 0));
    rangePose = transform * rangePose;

    const cv::Scalar& color = m_parameters.laserScanColor();
    int size = m_parameters.laserScanSize();
    int halfSize = size / 2;

    int centerX, centerY;
    convertTransformToMapCoordinates(rangePose, centerX, centerY);

    cv::Point p1(centerX - halfSize, centerY - halfSize);
    cv::Point p2(centerX + halfSize, centerY + halfSize);

    cv::rectangle(image, p1, p2, color, cv::FILLED);
}

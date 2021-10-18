#include "map_image_generator/drawers/LaserScanImageDrawer.h"

#include <tf/tf.h>

#include <cmath>

using namespace map_image_generator;
using namespace std;

LaserScanImageDrawer::LaserScanImageDrawer(const Parameters& parameters,
    ros::NodeHandle& nodeHandle,
    tf::TransformListener& tfListener) :
    ImageDrawer(parameters, nodeHandle, tfListener)
{
    m_laserScanSubscriber = m_nodeHandle.subscribe("laser_scan", 1,
        &LaserScanImageDrawer::laserScanCallback, this);
}

LaserScanImageDrawer::~LaserScanImageDrawer()
{
}

void LaserScanImageDrawer::draw(cv::Mat& image)
{
    if (!m_lastLaserScan) { return; }

    tf::StampedTransform transform;
    try
    {
        m_tfListener.lookupTransform(m_parameters.mapFrameId(), m_lastLaserScan->header.frame_id,
            ros::Time(0), transform);
        drawLaserScan(image, transform);
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s",ex.what());
    }
}

void LaserScanImageDrawer::laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& laserScan)
{
    m_lastLaserScan = laserScan;
}


void LaserScanImageDrawer::drawLaserScan(cv::Mat& image, tf::StampedTransform& transform)
{  
    typedef sensor_msgs::LaserScan::_ranges_type::const_iterator RangesConstIterator;

    float angle = m_lastLaserScan->angle_min;
    for (RangesConstIterator it = m_lastLaserScan->ranges.begin(); it != m_lastLaserScan->ranges.end(); ++it)
    {
        if (m_lastLaserScan->range_min <= *it &&
            *it <= m_lastLaserScan->range_max)
        {
            drawRange(image, transform, *it, angle);
        }

        angle += m_lastLaserScan->angle_increment;
    }
}

void LaserScanImageDrawer::drawRange(cv::Mat& image, tf::StampedTransform& transform, float range, float angle)
{
    tf::Pose rangePose(tf::Quaternion(0, 0, 0, 0), tf::Vector3(range * cos(angle), range * sin(angle), 0));
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
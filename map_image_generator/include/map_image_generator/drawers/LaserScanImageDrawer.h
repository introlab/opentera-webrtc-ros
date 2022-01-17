#ifndef LASER_SCAN_IMAGE_DRAWER_H
#define LASER_SCAN_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

namespace map_image_generator
{
    class LaserScanImageDrawer : public ImageDrawer
    {
        ros::Subscriber m_laserScanSubscriber;
        sensor_msgs::LaserScan::ConstPtr m_lastLaserScan;

    public:
        LaserScanImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                             tf::TransformListener& tfListener);
        ~LaserScanImageDrawer() override;

        void draw(cv::Mat& image, double& scaleFactor) override;

    private:
        void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& laserScan);

        void drawLaserScan(cv::Mat& image, tf::Transform& transform, double& scaleFactor);
        void drawRange(cv::Mat& image, tf::Transform& transform, double& scaleFactor,
                       float range, float angle);
    };
}
#endif

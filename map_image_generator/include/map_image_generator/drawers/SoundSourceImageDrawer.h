#ifndef SOUND_SOURCE_IMAGE_DRAWER_H
#define SOUND_SOURCE_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <geometry_msgs/msg/pose.hpp>
#include <odas_ros_msgs/msg/odas_sst_array_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/buffer.h>

namespace map_image_generator
{
    class SoundSourceImageDrawer : public ImageDrawer
    {
        rclcpp::Subscription<odas_ros_msgs::msg::OdasSstArrayStamped>::SharedPtr m_soundSourcesArraySubscriber;
        odas_ros_msgs::msg::OdasSstArrayStamped::ConstSharedPtr m_lastSoundSourcesArray;
        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_laserScanSubscriber;
        sensor_msgs::msg::LaserScan::ConstSharedPtr m_lastLaserScan;

    public:
        SoundSourceImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer);
        ~SoundSourceImageDrawer() override;

        void draw(cv::Mat& image) override;

    protected:
        void soundSourcesCallback(const odas_ros_msgs::msg::OdasSstArrayStamped::ConstSharedPtr& soundSources);
        void laserScanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& laserScan);

    private:
        void drawWithLidar(cv::Mat& image);
        void drawSoundSourcesWithLidar(
            cv::Mat& image,
            const tf2::Transform& sourceToLidarTf,
            const tf2::Transform& lidarToRefTf);

        void drawWithoutLidar(cv::Mat& image);
        void drawSoundSourcesWithoutLidar(cv::Mat& image, const tf2::Transform& sourceToRefTf);

        void drawSoundSource(cv::Mat& image, const odas_ros_msgs::msg::OdasSst& source, tf2::Transform poseInRef);
        void drawConcentricCircles(cv::Mat& image, int x, int y, int radius, double colorRatio);

        static tf2::Transform getPoseFromSst(const odas_ros_msgs::msg::OdasSst& sst);

        float getRangeForAngle(double angle);
        tf2::Transform getRangePose(const tf2::Transform& lidarPose);
        tf2::Transform getRefEndPose(const tf2::Transform& refPose);
    };
}
#endif

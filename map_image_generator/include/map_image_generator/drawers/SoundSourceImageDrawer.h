#ifndef SOUND_SOURCE_IMAGE_DRAWER_H
#define SOUND_SOURCE_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <geometry_msgs/Pose.h>
#include <odas_ros/OdasSstArrayStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

namespace map_image_generator
{
    class SoundSourceImageDrawer : public ImageDrawer
    {
        ros::Subscriber m_soundSourcesArraySubscriber;
        odas_ros::OdasSstArrayStamped::ConstPtr m_lastSoundSourcesArray;
        ros::Subscriber m_laserScanSubscriber;
        sensor_msgs::LaserScan::ConstPtr m_lastLaserScan;

    public:
        SoundSourceImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                               tf::TransformListener& tfListener);
        ~SoundSourceImageDrawer() override;

        void draw(cv::Mat& image) override;

    protected:
        void
        soundSourcesCallback(const odas_ros::OdasSstArrayStamped::ConstPtr& soundSources);
        void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& laserScan);

    private:
        void drawSoundSources(cv::Mat& image, const tf::Transform& sourceToLidarTf,
                              const tf::Transform& lidarToRefTf);
        void drawSoundSource(cv::Mat& image, const odas_ros::OdasSst& source,
                             const tf::Transform& sourceToLidarTf,
                             const tf::Transform& lidarToRefTf);
        void drawConcentricCircles(cv::Mat& image, int x, int y, int radius,
                                   double colorRatio);

        static tf::Pose getPoseFromSst(const odas_ros::OdasSst& sst);

        float getRangeForAngle(double angle);
        tf::Pose getRangePose(const tf::Pose& lidarPose);
    };
}
#endif

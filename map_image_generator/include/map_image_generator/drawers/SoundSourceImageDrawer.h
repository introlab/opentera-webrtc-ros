#ifndef SOUND_SOURCE_IMAGE_DRAWER_H
#define SOUND_SOURCE_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <geometry_msgs/PoseArray.h>
#include <ros/ros.h>

namespace map_image_generator
{
    class SoundSourceImageDrawer : public ImageDrawer
    {
        ros::Subscriber m_soundSourcesArraySubscriber;
        geometry_msgs::PoseArray::ConstPtr m_lastSoundSourcesArray;

    public:
        SoundSourceImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                               tf::TransformListener& tfListener);
        ~SoundSourceImageDrawer() override;

        void draw(cv::Mat& image) override;

    protected:
        void soundSourcesCallback(const geometry_msgs::PoseArray::ConstPtr& soundSources);

    private:
        void drawSoundSources(cv::Mat& image, tf::Transform& transform);
        void drawConcentricCircles(cv::Mat& image, int x, int y, int radius);
    };
}
#endif

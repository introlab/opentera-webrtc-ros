#include "map_image_generator/drawers/SoundSourceImageDrawer.h"

#include <cmath>
#include <tf/tf.h>

using namespace map_image_generator;
using namespace std;

SoundSourceImageDrawer::SoundSourceImageDrawer(const Parameters& parameters,
                                               ros::NodeHandle& nodeHandle,
                                               tf::TransformListener& tfListener)
    : ImageDrawer(parameters, nodeHandle, tfListener),
      m_soundSourcesArraySubscriber{nodeHandle.subscribe(
          "sound_sources", 1, &SoundSourceImageDrawer::soundSourcesCallback, this)}
{
}

SoundSourceImageDrawer::~SoundSourceImageDrawer() = default;

void SoundSourceImageDrawer::soundSourcesCallback(
    const geometry_msgs::PoseArray::ConstPtr& soundSources)
{
    m_lastSoundSourcesArray = soundSources;
}

void SoundSourceImageDrawer::draw(cv::Mat& image)
{
    auto tf = getTransformInRef(m_lastSoundSourcesArray->header.frame_id);
    if (tf)
    {
        drawSoundSources(image, *tf);
    }
}

void SoundSourceImageDrawer::drawSoundSources(cv::Mat& image, tf::Transform& transform)
{
    int size = m_parameters.soundSourceSize();

    for (const auto& sourcePose : m_lastSoundSourcesArray->poses)
    {
        tf::Pose pose;
        tf::poseMsgToTF(sourcePose, pose);
        pose = transform * pose;
        adjustTransformForRobotRef(pose);
        double yaw = tf::getYaw(pose.getRotation());

        int startX, startY;
        convertTransformToMapCoordinates(pose, startX, startY);

        int endX = static_cast<int>(startX + size * std::cos(yaw));
        int endY = static_cast<int>(startY + size * std::sin(yaw));

        drawConcentricCircles(image, endX, endY, size);
    }
}

void SoundSourceImageDrawer::drawConcentricCircles(cv::Mat& image, int x, int y,
                                                   int radius)
{
    const cv::Scalar& color = m_parameters.soundSourceColor();
    cv::circle(image, cv::Point(x, y), ceilDivision(radius, 25.0), color, cv::FILLED,
               cv::LINE_8);
    cv::circle(image, cv::Point(x, y), ceilDivision(radius, 8.0), color,
               ceilDivision(radius, 30.0), cv::LINE_8);
}

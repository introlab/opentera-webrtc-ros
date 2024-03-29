#include "map_image_generator/drawers/SoundSourceImageDrawer.h"

#include <cmath>
#include <sstream>
#include <tf/tf.h>

using namespace map_image_generator;
using namespace std;

SoundSourceImageDrawer::SoundSourceImageDrawer(
    const Parameters& parameters,
    ros::NodeHandle& nodeHandle,
    tf::TransformListener& tfListener)
    : ImageDrawer(parameters, nodeHandle, tfListener),
      m_soundSourcesArraySubscriber{
          nodeHandle.subscribe("sound_sources", 1, &SoundSourceImageDrawer::soundSourcesCallback, this)},
      m_laserScanSubscriber{nodeHandle.subscribe("laser_scan", 1, &SoundSourceImageDrawer::laserScanCallback, this)}
{
}

SoundSourceImageDrawer::~SoundSourceImageDrawer() = default;

void SoundSourceImageDrawer::soundSourcesCallback(const odas_ros::OdasSstArrayStamped::ConstPtr& soundSources)
{
    m_lastSoundSourcesArray = soundSources;
}

void SoundSourceImageDrawer::laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& laserScan)
{
    m_lastLaserScan = laserScan;
}

tf::Pose SoundSourceImageDrawer::getPoseFromSst(const odas_ros::OdasSst& sst)
{
    tf::Pose pose;
    pose.setOrigin({0, 0, 0});

    double yaw = std::atan2(sst.y, sst.x);
    double pitch = -std::atan2(sst.z, std::sqrt(sst.x * sst.x + sst.y * sst.y));
    double roll = 0;

    tf::Quaternion quaternion;
    quaternion.setRPY(roll, pitch, yaw);
    pose.setRotation(quaternion.normalized());

    return pose;
}


void SoundSourceImageDrawer::draw(cv::Mat& image)
{
    if (!m_lastSoundSourcesArray)
    {
        return;
    }

    if (m_lastLaserScan)
    {
        drawWithLidar(image);
    }
    else
    {
        drawWithoutLidar(image);
    }
}

void SoundSourceImageDrawer::drawWithLidar(cv::Mat& image)
{
    tf::StampedTransform sourceToLidarTf;

    try
    {
        m_tfListener.lookupTransform(
            m_lastLaserScan->header.frame_id,
            m_lastSoundSourcesArray->header.frame_id,
            ros::Time(0),
            sourceToLidarTf);
    }
    catch (tf::TransformException& ex)
    {
        ROS_ERROR("%s", ex.what());
        return;
    }

    auto lidarToRefTf = getTransformInRef(m_lastLaserScan->header.frame_id);
    if (lidarToRefTf)
    {
        drawSoundSourcesWithLidar(image, sourceToLidarTf, *lidarToRefTf);
    }
}

void SoundSourceImageDrawer::drawWithoutLidar(cv::Mat& image)
{
    auto sourceToRefTf = getTransformInRef(m_lastSoundSourcesArray->header.frame_id);
    if (sourceToRefTf)
    {
        drawSoundSourcesWithoutLidar(image, *sourceToRefTf);
    }
}

void SoundSourceImageDrawer::drawSoundSourcesWithLidar(
    cv::Mat& image,
    const tf::Transform& sourceToLidarTf,
    const tf::Transform& lidarToRefTf)
{
    for (const auto& source : m_lastSoundSourcesArray->sources)
    {
        tf::Pose poseSource = getPoseFromSst(source);
        tf::Pose poseLidar = sourceToLidarTf * poseSource;

        tf::Pose poseRef = getRangePose(poseLidar);
        poseRef = lidarToRefTf * poseRef;

        drawSoundSource(image, source, poseRef);
    }
}

void SoundSourceImageDrawer::drawSoundSourcesWithoutLidar(cv::Mat& image, const tf::Transform& sourceToRefTf)
{
    for (const auto& source : m_lastSoundSourcesArray->sources)
    {
        tf::Pose poseSource = getPoseFromSst(source);
        tf::Pose poseRef = sourceToRefTf * poseSource;

        drawSoundSource(image, source, getRefEndPose(poseRef));
    }
}

void SoundSourceImageDrawer::drawSoundSource(cv::Mat& image, const odas_ros::OdasSst& source, tf::Pose poseInRef)
{
    int size = m_parameters.soundSourceSize();

    adjustTransformForRobotRef(poseInRef);

    int centerX, centerY;
    convertTransformToMapCoordinates(poseInRef, centerX, centerY);

    drawConcentricCircles(image, centerX, centerY, size, source.activity);
}

void SoundSourceImageDrawer::drawConcentricCircles(cv::Mat& image, int x, int y, int radius, double colorRatio)
{
    const cv::Scalar& color =
        interpolateColors(m_parameters.soundSourceColorFull(), m_parameters.soundSourceColorDim(), colorRatio);

    cv::circle(image, cv::Point(x, y), ceilDivision(radius, 25.0), color, cv::FILLED, cv::LINE_8);
    cv::circle(image, cv::Point(x, y), ceilDivision(radius, 8.0), color, ceilDivision(radius, 30.0), cv::LINE_8);
}

float SoundSourceImageDrawer::getRangeForAngle(double angle)
{
    if (angle < m_lastLaserScan->angle_min || angle > m_lastLaserScan->angle_max)
    {
        return -1;
    }

    int index = static_cast<int>(std::floor((angle - m_lastLaserScan->angle_min) / m_lastLaserScan->angle_increment));
    if (index < 0 || index >= m_lastLaserScan->ranges.size())
    {
        std::ostringstream oss;
        oss << "Index out of range, this shouldn't happen. => angle: " << angle << " ; angle_range: ["
            << m_lastLaserScan->angle_min << ", " << m_lastLaserScan->angle_max
            << "] ; increment: " << m_lastLaserScan->angle_increment << " ; index: " << index
            << " ; ranges.size(): " << m_lastLaserScan->ranges.size();
        throw std::runtime_error(oss.str());
    }

    float range = m_lastLaserScan->ranges[index];
    if (range < m_lastLaserScan->range_min || range > m_lastLaserScan->range_max)
    {
        return -2;
    }

    return range;
}

tf::Pose SoundSourceImageDrawer::getRangePose(const tf::Pose& lidarPose)
{
    double angle = tf::getYaw(lidarPose.getRotation());
    angle = tfNormalizeAngle0To2Pi(angle);
    float range = getRangeForAngle(angle);

    if (range < 0)
    {
        range = m_parameters.soundSourceRange();
    }
    else if (range > m_parameters.soundSourceMaxRange())
    {
        range = m_parameters.soundSourceMaxRange();
    }

    return tf::Pose(tf::Quaternion(0, 0, 0, 0), tf::Vector3(range * cos(angle), range * sin(angle), 0));
}

tf::Pose SoundSourceImageDrawer::getRefEndPose(const tf::Pose& refPose)
{
    double angle = tf::getYaw(refPose.getRotation());
    float range = m_parameters.soundSourceRange();

    return tf::Pose(tf::Quaternion(0, 0, 0, 0), tf::Vector3(range * cos(angle), range * sin(angle), 0));
}

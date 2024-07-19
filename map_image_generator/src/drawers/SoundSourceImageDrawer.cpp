#include "map_image_generator/drawers/SoundSourceImageDrawer.h"

#include <cmath>
#include <sstream>

#include <tf2/utils.h>

using namespace map_image_generator;
using namespace std;

SoundSourceImageDrawer::SoundSourceImageDrawer(
    const Parameters& parameters,
    rclcpp::Node& node,
    tf2_ros::Buffer& tfBuffer)
    : ImageDrawer(parameters, node, tfBuffer),
      m_soundSourcesArraySubscriber{node.create_subscription<odas_ros_msgs::msg::OdasSstArrayStamped>(
          "sound_sources",
          1,
          bind_this<odas_ros_msgs::msg::OdasSstArrayStamped>(this, &SoundSourceImageDrawer::soundSourcesCallback))},
      m_laserScanSubscriber{m_node.create_subscription<sensor_msgs::msg::LaserScan>(
          "laser_scan",
          1,
          bind_this<sensor_msgs::msg::LaserScan>(this, &SoundSourceImageDrawer::laserScanCallback))}
{
}

SoundSourceImageDrawer::~SoundSourceImageDrawer() = default;

void SoundSourceImageDrawer::soundSourcesCallback(
    const odas_ros_msgs::msg::OdasSstArrayStamped::ConstSharedPtr& soundSources)
{
    m_lastSoundSourcesArray = soundSources;
}

void SoundSourceImageDrawer::laserScanCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& laserScan)
{
    m_lastLaserScan = laserScan;
}

tf2::Transform SoundSourceImageDrawer::getPoseFromSst(const odas_ros_msgs::msg::OdasSst& sst)
{
    tf2::Transform pose;
    pose.setOrigin({0, 0, 0});

    double yaw = std::atan2(sst.y, sst.x);
    double pitch = -std::atan2(sst.z, std::sqrt(sst.x * sst.x + sst.y * sst.y));
    double roll = 0;

    tf2::Quaternion quaternion;
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
    tf2::Stamped<tf2::Transform> sourceToLidarTf;

    try
    {
        auto transformMsg = m_tfBuffer.lookupTransform(
            m_lastLaserScan->header.frame_id,
            m_lastSoundSourcesArray->header.frame_id,
            tf2::TimePointZero);
        tf2::fromMsg(transformMsg, sourceToLidarTf);
    }
    catch (tf2::TransformException& ex)
    {
        RCLCPP_ERROR(m_node.get_logger(), "%s", ex.what());
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
    const tf2::Transform& sourceToLidarTf,
    const tf2::Transform& lidarToRefTf)
{
    for (const auto& source : m_lastSoundSourcesArray->sources)
    {
        tf2::Transform poseSource = getPoseFromSst(source);
        tf2::Transform poseLidar = sourceToLidarTf * poseSource;

        tf2::Transform poseRef = getRangePose(poseLidar);
        poseRef = lidarToRefTf * poseRef;

        drawSoundSource(image, source, poseRef);
    }
}

void SoundSourceImageDrawer::drawSoundSourcesWithoutLidar(cv::Mat& image, const tf2::Transform& sourceToRefTf)
{
    for (const auto& source : m_lastSoundSourcesArray->sources)
    {
        tf2::Transform poseSource = getPoseFromSst(source);
        tf2::Transform poseRef = sourceToRefTf * poseSource;

        drawSoundSource(image, source, getRefEndPose(poseRef));
    }
}

void SoundSourceImageDrawer::drawSoundSource(
    cv::Mat& image,
    const odas_ros_msgs::msg::OdasSst& source,
    tf2::Transform poseInRef)
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
    if (index < 0 || index >= static_cast<int64_t>(m_lastLaserScan->ranges.size()))
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

tf2::Transform SoundSourceImageDrawer::getRangePose(const tf2::Transform& lidarPose)
{
    double angle = tf2::getYaw(lidarPose.getRotation());
    angle = tf2NormalizeAngle0To2Pi(angle);
    float range = getRangeForAngle(angle);

    if (range < 0)
    {
        range = m_parameters.soundSourceRange();
    }
    else if (range > m_parameters.soundSourceMaxRange())
    {
        range = m_parameters.soundSourceMaxRange();
    }

    return tf2::Transform(tf2::Quaternion(0, 0, 0, 0), tf2::Vector3(range * cos(angle), range * sin(angle), 0));
}

tf2::Transform SoundSourceImageDrawer::getRefEndPose(const tf2::Transform& refPose)
{
    double angle = tf2::getYaw(refPose.getRotation());
    float range = m_parameters.soundSourceRange();

    return tf2::Transform(tf2::Quaternion(0, 0, 0, 0), tf2::Vector3(range * cos(angle), range * sin(angle), 0));
}

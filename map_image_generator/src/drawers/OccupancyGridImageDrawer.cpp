#include "map_image_generator/drawers/OccupancyGridImageDrawer.h"

#include "map_image_generator/utils.h"

#include <queue>
#include <tf/tf.h>

using namespace map_image_generator;

OccupancyGridImageDrawer::OccupancyGridImageDrawer(const Parameters& parameters,
                                                   ros::NodeHandle& nodeHandle,
                                                   tf::TransformListener& tfListener)
    : ImageDrawer(parameters, nodeHandle, tfListener)
{
    m_occupancyGridSubscriber = m_nodeHandle.subscribe(
        "occupancy_grid", 1, &OccupancyGridImageDrawer::occupancyGridCallback, this);
}

OccupancyGridImageDrawer::~OccupancyGridImageDrawer() = default;

void OccupancyGridImageDrawer::draw(cv::Mat& image)
{
    if (!m_lastOccupancyGrid)
    {
        return;
    }

    drawNotScaledOccupancyGridImage();
    scaleOccupancyGridImage();

    if (m_parameters.centeredRobot())
    {
        drawOccupancyGridImageCenteredAroundRobot(image);
    }
    else
    {
        drawOccupancyGridImage(image);
    }
}

void OccupancyGridImageDrawer::occupancyGridCallback(
    const nav_msgs::OccupancyGrid::ConstPtr& occupancyGrid)
{
    m_lastOccupancyGrid = occupancyGrid;
}

void OccupancyGridImageDrawer::drawNotScaledOccupancyGridImage()
{
    changeNotScaledOccupancyGridImageIfNeeded();

    const cv::Vec3b& wallColor = m_parameters.wallColor();
    const cv::Vec3b& freeSpaceColor = m_parameters.freeSpaceColor();
    const cv::Vec3b& unknownSpaceColor = m_parameters.unknownSpaceColor();

    for (int y = 0; y < m_notScaledOccupancyGridImage.rows; y++)
    {
        for (int x = 0; x < m_notScaledOccupancyGridImage.cols; x++)
        {
            int dataIndex = y * m_notScaledOccupancyGridImage.cols + x;
            int dataValue = m_lastOccupancyGrid->data[dataIndex];

            cv::Point pixelPosition(x, y);
            if (dataValue == -1)
            {
                m_notScaledOccupancyGridImage.at<cv::Vec3b>(pixelPosition) =
                    unknownSpaceColor;
            }
            else if (dataValue == 0)
            {
                m_notScaledOccupancyGridImage.at<cv::Vec3b>(pixelPosition) =
                    freeSpaceColor;
            }
            else
            {
                m_notScaledOccupancyGridImage.at<cv::Vec3b>(pixelPosition) = wallColor;
            }
        }
    }
}

void OccupancyGridImageDrawer::changeNotScaledOccupancyGridImageIfNeeded()
{
    int gridHeight = m_lastOccupancyGrid->info.height;
    int gridWidth = m_lastOccupancyGrid->info.width;

    if (m_notScaledOccupancyGridImage.rows != gridHeight
        || m_notScaledOccupancyGridImage.cols != gridWidth)
    {
        m_notScaledOccupancyGridImage = cv::Mat(gridHeight, gridWidth, CV_8UC3);
    }
}

void OccupancyGridImageDrawer::scaleOccupancyGridImage()
{
    changeScaledOccupancyGridImageIfNeeded();
    cv::resize(m_notScaledOccupancyGridImage, m_scaledOccupancyGridImage,
               m_scaledOccupancyGridImage.size());
}

void OccupancyGridImageDrawer::changeScaledOccupancyGridImageIfNeeded()
{
    int gridHeight = m_lastOccupancyGrid->info.height;
    int gridWidth = m_lastOccupancyGrid->info.width;
    float gridResolution = m_lastOccupancyGrid->info.resolution;

    int height =
        static_cast<int>(gridHeight * gridResolution * m_parameters.resolution());
    int width = static_cast<int>(gridWidth * gridResolution * m_parameters.resolution());

    if (m_scaledOccupancyGridImage.rows != height
        || m_scaledOccupancyGridImage.cols != width)
    {
        m_scaledOccupancyGridImage = cv::Mat(height, width, CV_8UC3);
    }
}

void OccupancyGridImageDrawer::rotateImageAboutPoint(cv::Mat& image, double angle,
                                                     const cv::Point& point)
{
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(point, angle, 1.0);
    cv::warpAffine(image, image, rotationMatrix, image.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, m_parameters.unknownSpaceColor());
}
void OccupancyGridImageDrawer::rotateImageAboutCenter(cv::Mat& image, double angle)
{
    cv::Point center{image.cols / 2, image.rows / 2};
    rotateImageAboutPoint(image, angle, center);
}

void OccupancyGridImageDrawer::drawOccupancyGridImage(cv::Mat& image)
{
    double occupancyXOrigin = m_lastOccupancyGrid->info.origin.position.x;
    double occupancyYOrigin = m_lastOccupancyGrid->info.origin.position.y;

    int rowOffset = static_cast<int>(occupancyYOrigin * m_parameters.resolution()
                                     + m_parameters.yOrigin());
    int colOffset = static_cast<int>(occupancyXOrigin * m_parameters.resolution()
                                     + m_parameters.xOrigin());

    if (rowOffset >= 0 && rowOffset < image.rows - m_scaledOccupancyGridImage.rows
        && colOffset >= 0 && colOffset < image.cols - m_scaledOccupancyGridImage.cols)
    {
        cv::Rect roi(
            cv::Point(colOffset, rowOffset),
            cv::Size(m_scaledOccupancyGridImage.cols, m_scaledOccupancyGridImage.rows));
        m_scaledOccupancyGridImage.copyTo(image(roi));
    }
    else
    {
        ROS_ERROR_STREAM(
            "Unable to draw the occupancy grid because the map is too small");
    }
}

void OccupancyGridImageDrawer::drawOccupancyGridImageCenteredAroundRobot(cv::Mat& image)
{
    tf::StampedTransform robotTransform;
    try
    {
        m_tfListener.lookupTransform(m_parameters.mapFrameId(),
                                     m_parameters.robotFrameId(), ros::Time(0),
                                     robotTransform);
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s", ex.what());
        return;
    }

    int outHeight = image.rows;
    int outWidth = image.cols;

    int robotX = 0;
    int robotY = 0;
    convertTransformToInputMapCoordinates(robotTransform, m_lastOccupancyGrid->info,
                                          robotX, robotY);
    int top = 0 + robotY;
    int bottom = (m_scaledOccupancyGridImage.rows - 1) - robotY;
    int left = 0 + robotX;
    int right = (m_scaledOccupancyGridImage.cols - 1) - robotX;

    int topPadding = std::max(0, (outHeight - 1) / 2 - top);
    int bottomPadding = std::max(0, outHeight / 2 - bottom);
    int leftPadding = std::max(0, (outWidth - 1) / 2 - left);
    int rightPadding = std::max(0, outWidth / 2 - right);

    cv::Mat paddedImage;
    cv::copyMakeBorder(m_scaledOccupancyGridImage, paddedImage, topPadding, bottomPadding,
                       leftPadding, rightPadding, cv::BORDER_CONSTANT,
                       m_parameters.unknownSpaceColor());

    if (m_parameters.centeredRobot())
    {
        using namespace map_image_generator;

        double rotationAngle = rad2deg(tf::getYaw(robotTransform.getRotation()));
        cv::Point rotationCentre{robotX + leftPadding, robotY + topPadding};
        rotateImageAboutPoint(paddedImage, rotationAngle, rotationCentre);
    }

    cv::Rect roi(cv::Point(std::max(0, left - (outWidth - 1) / 2),
                           std::max(0, top - (outHeight - 1) / 2)),
                 cv::Size(outWidth, outHeight));
    paddedImage(roi).copyTo(image);
}

void OccupancyGridImageDrawer::convertMapInfoToMapCoordinates(
    const nav_msgs::MapMetaData& mapInfo, int& x, int& y)
{
    tf::Transform mapInfoTransform;
    mapInfoTransform.setOrigin(tf::Vector3(
        mapInfo.origin.position.x, mapInfo.origin.position.y, mapInfo.origin.position.z));
    convertTransformToMapCoordinates(mapInfoTransform, x, y);
}

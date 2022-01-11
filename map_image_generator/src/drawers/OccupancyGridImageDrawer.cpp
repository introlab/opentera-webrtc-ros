#include "map_image_generator/drawers/OccupancyGridImageDrawer.h"

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

OccupancyGridImageDrawer::~OccupancyGridImageDrawer() {}

void OccupancyGridImageDrawer::draw(cv::Mat& image)
{
    if (!m_lastOccupancyGrid)
    {
        return;
    }

    drawNotScaledOccupancyGridImage();
    scaleOccupancyGridImage();

    drawOccupancyGridImage(image);
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

void OccupancyGridImageDrawer::drawOccupancyGridImage(cv::Mat& image)
{
    double occupancyXOrigin = m_lastOccupancyGrid->info.origin.position.x;
    double occupancyYOrigin = m_lastOccupancyGrid->info.origin.position.y;

    // TODO: Make the robot position the center of the map. The following commented code
    // works.
    tf::StampedTransform robotTransform;
    tf::StampedTransform transform;
    try
    {
        m_tfListener.lookupTransform(m_parameters.mapFrameId(),
                                     m_parameters.robotFrameId(), ros::Time(0),
                                     robotTransform);
        m_tfListener.lookupTransform(m_parameters.refFrameId(), m_parameters.mapFrameId(),
                                     ros::Time(0), transform);
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s", ex.what());
        return;
    }

    int rowOffset = static_cast<int>(
        (occupancyYOrigin - robotTransform.getOrigin().getY()) * m_parameters.resolution()
        + m_parameters.yOrigin());
    int colOffset = static_cast<int>(
        (occupancyXOrigin - robotTransform.getOrigin().getX()) * m_parameters.resolution()
        + m_parameters.xOrigin());

    // int rowOffset = static_cast<int>(occupancyYOrigin * m_parameters.resolution() +
    // m_parameters.yOrigin());
    // int colOffset = static_cast<int>(occupancyXOrigin * m_parameters.resolution() +
    // m_parameters.xOrigin());

    int maxRowOffset = image.rows - m_scaledOccupancyGridImage.rows - 1;
    int maxColOffset = image.cols - m_scaledOccupancyGridImage.cols - 1;

    int topPadding = rowOffset < 0 ? -rowOffset : 0;
    int bottomPadding = rowOffset > maxRowOffset ? rowOffset - maxRowOffset : 0;
    int leftPadding = colOffset < 0 ? -colOffset : 0;
    int rightPadding = colOffset > maxColOffset ? colOffset - maxColOffset : 0;
    // int maxPadding = std::max({topPadding, bottomPadding, leftPadding, rightPadding});

    // cv::Mat paddedOccupancyGridImage;
    cv::copyMakeBorder(image, image, maxPadding, maxPadding, maxPadding, maxPadding,
                       cv::BORDER_CONSTANT, m_parameters.unknownSpaceColor());

    // if (rowOffset >= 0 &&
    //     rowOffset < image.rows - m_scaledOccupancyGridImage.rows &&
    //     colOffset >= 0 &&
    //     colOffset < image.cols - m_scaledOccupancyGridImage.cols)
    // {
    cv::Rect roi(
        cv::Point(colOffset + maxPadding, rowOffset + maxPadding),
        cv::Size(m_scaledOccupancyGridImage.cols, m_scaledOccupancyGridImage.rows));
    auto truc = roi.tl();
    m_scaledOccupancyGridImage.copyTo(image(roi));
    // }
    // else
    // {
    //     ROS_ERROR_STREAM("Unable to draw the occupancy grid because the map is too
    //     small"); return;
    // }
}

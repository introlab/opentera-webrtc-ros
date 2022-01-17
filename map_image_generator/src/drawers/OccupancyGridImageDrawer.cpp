#include "map_image_generator/drawers/OccupancyGridImageDrawer.h"

#include "map_image_generator/utils.h"

#include <queue>
#include <tf/tf.h>

using namespace map_image_generator;

OccupancyGridImageDrawer::OccupancyGridImageDrawer(const Parameters& parameters,
                                                   ros::NodeHandle& nodeHandle,
                                                   tf::TransformListener& tfListener)
    : ImageDrawer(parameters, nodeHandle, tfListener),
      m_occupancyGridSubscriber{m_nodeHandle.subscribe(
          "occupancy_grid", 1, &OccupancyGridImageDrawer::occupancyGridCallback, this)}
{
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

    for (auto y = 0; y < m_notScaledOccupancyGridImage.rows; y++)
    {
        for (auto x = 0; x < m_notScaledOccupancyGridImage.cols; x++)
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

// Replace with std::optional in C++17
std::unique_ptr<tf::Transform> OccupancyGridImageDrawer::getRobotTransform() const
{
    tf::StampedTransform transform;

    try
    {
        m_tfListener.lookupTransform(m_parameters.mapFrameId(),
                                     m_parameters.robotFrameId(), ros::Time(0),
                                     transform);
    }
    catch (tf::TransformException& ex)
    {
        ROS_ERROR("%s", ex.what());
        return {};
    }

    // Replace with std::optional in C++17
    return std::make_unique<tf::Transform>(transform);
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

OccupancyGridImageDrawer::DirectionalValues
OccupancyGridImageDrawer::computePadding(const DirectionalValues& robotPosition,
                                         int height, int width)
{
    return {
        .top = restrictToPositive((height - 1) / 2 - robotPosition.top),
        .bottom = restrictToPositive(height / 2 - robotPosition.bottom),

        .left = restrictToPositive((width - 1) / 2 - robotPosition.left),
        .right = restrictToPositive(width / 2 - robotPosition.right),
    };
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
    auto robotTransform = getRobotTransform();
    if (!robotTransform)
    {
        return;
    }

    int outHeight = image.rows;
    int outWidth = image.cols;

    int robotX = 0;
    int robotY = 0;
    convertTransformToInputMapCoordinates(*robotTransform, m_lastOccupancyGrid->info,
                                          robotX, robotY);

    DirectionalValues robotPosition = {
        .top = 0 + robotY,
        .bottom = (m_scaledOccupancyGridImage.rows - 1) - robotY,
        .left = 0 + robotX,
        .right = (m_scaledOccupancyGridImage.cols - 1) - robotX,
    };

    DirectionalValues padding = computePadding(robotPosition, outHeight, outWidth);
    adjustPaddingForCenteredRobotOffset(padding, outWidth, robotPosition);

    cv::Mat paddedImage;
    cv::copyMakeBorder(m_scaledOccupancyGridImage, paddedImage, padding.top,
                       padding.bottom, padding.left, padding.right, cv::BORDER_CONSTANT,
                       m_parameters.unknownSpaceColor());

        using namespace map_image_generator;

    double rotationAngle = rad2deg(tf::getYaw(robotTransform->getRotation()));
    cv::Point rotationCentre{robotX + padding.left, robotY + padding.top};
        rotateImageAboutPoint(paddedImage, rotationAngle, rotationCentre);

    cv::Rect roi(cv::Point(restrictToPositive(robotPosition.left - (outWidth - 1) / 2
                                              + m_parameters.robotVerticalOffset()),
                           restrictToPositive(robotPosition.top - (outHeight - 1) / 2)),
                 cv::Size(outWidth, outHeight));
    paddedImage(roi).copyTo(image);
    cv::flip(image, image, 1);
    cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
}

void OccupancyGridImageDrawer::adjustPaddingForCenteredRobotOffset(
    DirectionalValues& padding, int width, const DirectionalValues& robotPosition)
{
    // The vertical axis on which we apply the robot vertical offset is horizontal
    // currently as we will rotate it later
    int& topPaddingRotated = padding.right;
    int& bottomPaddingRotated = padding.left;
    if (m_parameters.robotVerticalOffset() > 0)
    {
        // If the offset is positive, we need to add padding to the top and can
        // possibly reduce padding to the bottom
        bottomPaddingRotated -= std::min(
            bottomPaddingRotated, restrictToPositive(m_parameters.robotVerticalOffset()));
        topPaddingRotated += restrictToPositive(m_parameters.robotVerticalOffset());
    }
    if (m_parameters.robotVerticalOffset() < 0)
{
        // If the offset is negative, we need to recompute padding to the bottom and
        // can possibly reduce padding to the top
        bottomPaddingRotated = restrictToPositive(
            (width - 1) / 2 + restrictToPositive(-m_parameters.robotVerticalOffset())
            - robotPosition.left);
        topPaddingRotated -= std::min(
            topPaddingRotated, restrictToPositive(-m_parameters.robotVerticalOffset()));
    }
}

#include "map_image_generator/drawers/OccupancyGridImageDrawer.h"

#include "map_image_generator/utils.h"

#include <queue>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/utils.h>

using namespace map_image_generator;

OccupancyGridImageDrawer::OccupancyGridImageDrawer(
    Parameters& parameters,
    rclcpp::Node& node,
    tf2_ros::Buffer& tfBuffer)
    : ImageDrawer{parameters, node, tfBuffer},
      m_mutableParameters{parameters},
      m_occupancyGridSubscriber{m_node.create_subscription<nav_msgs::msg::OccupancyGrid>(
          "occupancy_grid",
          1,
          bind_this<nav_msgs::msg::OccupancyGrid>(this, &OccupancyGridImageDrawer::occupancyGridCallback))},
      m_mapViewChangerService{m_node.create_service<opentera_webrtc_ros_msgs::srv::ChangeMapView>(
          "change_map_view",
          bind_this<opentera_webrtc_ros_msgs::srv::ChangeMapView>(
              this,
              &OccupancyGridImageDrawer::changeMapViewCallback))}
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

void OccupancyGridImageDrawer::occupancyGridCallback(const nav_msgs::msg::OccupancyGrid::ConstSharedPtr& occupancyGrid)
{
    m_lastOccupancyGrid = occupancyGrid;
}

void OccupancyGridImageDrawer::changeMapViewCallback(
    const opentera_webrtc_ros_msgs::srv::ChangeMapView::Request::ConstSharedPtr& req,
    const opentera_webrtc_ros_msgs::srv::ChangeMapView::Response::SharedPtr& res)
{
    res->success = true;
    if (req->view_new == opentera_webrtc_ros_msgs::srv::ChangeMapView::Request::VIEW_CENTERED_ROBOT)
    {
        m_mutableParameters.setCenteredRobot(true);
    }
    else if (req->view_new == opentera_webrtc_ros_msgs::srv::ChangeMapView::Request::VIEW_STATIC_MAP)
    {
        m_mutableParameters.setCenteredRobot(false);
    }
    else
    {
        RCLCPP_ERROR(m_node.get_logger(), "Unknown view type: %s. View was not changed.", req->view_new.c_str());
        res->success = false;
        res->message = "unknown view type";
    }
    return;
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
            int8_t dataValue = m_lastOccupancyGrid->data[dataIndex];

            cv::Point pixelPosition(x, y);
            if (dataValue == -1)
            {
                m_notScaledOccupancyGridImage.at<cv::Vec3b>(pixelPosition) = unknownSpaceColor;
            }
            else if (dataValue == 0)
            {
                m_notScaledOccupancyGridImage.at<cv::Vec3b>(pixelPosition) = freeSpaceColor;
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
    int gridHeight = static_cast<int>(m_lastOccupancyGrid->info.height);
    int gridWidth = static_cast<int>(m_lastOccupancyGrid->info.width);

    if (m_notScaledOccupancyGridImage.rows != gridHeight || m_notScaledOccupancyGridImage.cols != gridWidth)
    {
        m_notScaledOccupancyGridImage = cv::Mat(gridHeight, gridWidth, CV_8UC3);
    }
}

void OccupancyGridImageDrawer::scaleOccupancyGridImage()
{
    changeScaledOccupancyGridImageIfNeeded();
    cv::resize(m_notScaledOccupancyGridImage, m_scaledOccupancyGridImage, m_scaledOccupancyGridImage.size());
}

const cv::Mat& OccupancyGridImageDrawer::getZoomedOccupancyImage()
{
    if (map_image_generator::areApproxEqual(m_parameters.scaleFactor(), 1.0))
    {
        return m_scaledOccupancyGridImage;
    }
    else
    {
        cv::resize(
            m_scaledOccupancyGridImage,
            m_zoomedOccupancyGridImage,
            cv::Size{},
            m_parameters.scaleFactor(),
            m_parameters.scaleFactor());
        return m_zoomedOccupancyGridImage;
    }
}

void OccupancyGridImageDrawer::changeScaledOccupancyGridImageIfNeeded()
{
    int gridHeight = static_cast<int>(m_lastOccupancyGrid->info.height);
    int gridWidth = static_cast<int>(m_lastOccupancyGrid->info.width);
    float gridResolution = m_lastOccupancyGrid->info.resolution;

    int height = static_cast<int>(static_cast<double>(gridHeight) * gridResolution * m_parameters.resolution());
    int width = static_cast<int>(static_cast<double>(gridWidth) * gridResolution * m_parameters.resolution());

    if (m_scaledOccupancyGridImage.rows != height || m_scaledOccupancyGridImage.cols != width)
    {
        m_scaledOccupancyGridImage = cv::Mat(height, width, CV_8UC3);
    }
}

std::optional<tf2::Transform> OccupancyGridImageDrawer::getRobotTransform() const
{
    tf2::Stamped<tf2::Transform> transform;

    try
    {
        auto transformMsg =
            m_tfBuffer.lookupTransform(m_parameters.mapFrameId(), m_parameters.robotFrameId(), tf2::TimePointZero);
        tf2::fromMsg(transformMsg, transform);
    }
    catch (const tf2::TransformException& ex)
    {
        RCLCPP_ERROR(m_node.get_logger(), "%s", ex.what());
        return {};
    }

    return transform;
}

void OccupancyGridImageDrawer::rotateImageAboutPoint(cv::Mat& image, double angle, const cv::Point& point) const
{
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(point, angle, 1.0);
    cv::warpAffine(
        image,
        image,
        rotationMatrix,
        image.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        m_parameters.unknownSpaceColor());
}
void OccupancyGridImageDrawer::rotateImageAboutCenter(cv::Mat& image, double angle) const
{
    cv::Point center{image.cols / 2, image.rows / 2};
    rotateImageAboutPoint(image, angle, center);
}

OccupancyGridImageDrawer::DirectionalValues
    OccupancyGridImageDrawer::computePadding(const DirectionalValues& position, int height, int width)
{
    return {
        .top = restrictToPositive((height - 1) / 2 - position.top),
        .bottom = restrictToPositive(height / 2 - position.bottom),

        .left = restrictToPositive((width - 1) / 2 - position.left),
        .right = restrictToPositive(width / 2 - position.right),
    };
}

OccupancyGridImageDrawer::MapCoordinates
    OccupancyGridImageDrawer::getMapCoordinatesFromTf(const tf2::Transform& transform) const
{
    MapCoordinates mapCoordinates{};
    convertTransformToInputMapCoordinates(transform, m_lastOccupancyGrid->info, mapCoordinates.x, mapCoordinates.y);
    return mapCoordinates;
}

OccupancyGridImageDrawer::DirectionalValues
    OccupancyGridImageDrawer::getDirectionsFromMapCoordinates(const MapCoordinates& mapCoordinates, const cv::Mat& map)
{
    return {
        .top = 0 + mapCoordinates.y,
        .bottom = (map.rows - 1) - mapCoordinates.y,
        .left = 0 + mapCoordinates.x,
        .right = (map.cols - 1) - mapCoordinates.x,
    };
}

void OccupancyGridImageDrawer::drawOccupancyGridImage(cv::Mat& image)
{
    auto robotTransform = getRobotTransform();
    if (!robotTransform)
    {
        return;
    }

    int outHeight = image.rows;
    int outWidth = image.cols;

    MapCoordinates robotCoordinates = getMapCoordinatesFromTf(*robotTransform);
    DirectionalValues robotPosition = getDirectionsFromMapCoordinates(robotCoordinates, m_scaledOccupancyGridImage);

    double heightBorder = 0.1 * outHeight;
    double widthBorder = 0.1 * outWidth;

    // Map center
    double occupancyXOrigin = m_lastOccupancyGrid->info.origin.position.x;
    double occupancyYOrigin = m_lastOccupancyGrid->info.origin.position.y;

    tf2::Transform mapOriginPose;
    mapOriginPose.setOrigin({0.0, 0.0, 0.0});

    MapCoordinates mapOriginCoordinates = getMapCoordinatesFromTf(mapOriginPose);
    DirectionalValues mapPosition = getDirectionsFromMapCoordinates(mapOriginCoordinates, m_scaledOccupancyGridImage);

    double hScaleFactor = (0.4 * outWidth) / std::abs(robotPosition.left - mapOriginCoordinates.x);
    double vScaleFactor = (0.4 * outHeight) / std::abs(robotPosition.top - mapOriginCoordinates.y);

    m_mutableParameters.setScaleFactor(std::min({hScaleFactor, vScaleFactor, 1.0}));

    const auto& zoomedMap = getZoomedOccupancyImage();

    MapCoordinates zoomedMapOriginCoordinates{
        .x = static_cast<int>(mapOriginCoordinates.x * m_parameters.scaleFactor()),
        .y = static_cast<int>(mapOriginCoordinates.y * m_parameters.scaleFactor()),
    };
    DirectionalValues zoomedMapPosition = getDirectionsFromMapCoordinates(zoomedMapOriginCoordinates, zoomedMap);

    DirectionalValues padding = computePadding(zoomedMapPosition, outHeight, outWidth);

    cv::Mat paddedImage;
    cv::copyMakeBorder(
        zoomedMap,
        paddedImage,
        padding.top,
        padding.bottom,
        padding.left,
        padding.right,
        cv::BORDER_CONSTANT,
        m_parameters.unknownSpaceColor());

    using namespace map_image_generator;

    cv::Rect roi(
        cv::Point(
            restrictToPositive(zoomedMapPosition.left - (outWidth - 1) / 2),
            restrictToPositive(zoomedMapPosition.top - (outHeight - 1) / 2)),
        cv::Size(outWidth, outHeight));
    paddedImage(roi).copyTo(image);
    cv::flip(image, image, 1);
}

void OccupancyGridImageDrawer::drawOccupancyGridImageCenteredAroundRobot(cv::Mat& image)
{
    auto robotTransform = getRobotTransform();
    if (!robotTransform)
    {
        return;
    }

    // Relative to scaled occupancy grid image
    int outHeight = image.cols;
    int outWidth = image.rows;

    // // Relative to scaled occupancy grid image
    MapCoordinates robotCoordinates = getMapCoordinatesFromTf(*robotTransform);
    DirectionalValues robotPosition = getDirectionsFromMapCoordinates(robotCoordinates, m_scaledOccupancyGridImage);
    DirectionalValues padding = computePadding(robotPosition, outHeight, outWidth);
    adjustPaddingForCenteredRobotOffset(padding, outWidth, robotPosition);

    cv::Mat paddedImage;
    cv::copyMakeBorder(
        m_scaledOccupancyGridImage,
        paddedImage,
        padding.top,
        padding.bottom,
        padding.left,
        padding.right,
        cv::BORDER_CONSTANT,
        m_parameters.unknownSpaceColor());

    using namespace map_image_generator;

    double rotationAngle = rad2deg(tf2::getYaw(robotTransform->getRotation()));
    cv::Point rotationCenter{robotCoordinates.x + padding.left, robotCoordinates.y + padding.top};
    rotateImageAboutPoint(paddedImage, rotationAngle, rotationCenter);

    cv::Rect roi(
        cv::Point(
            restrictToPositive(robotPosition.left - (outWidth - 1) / 2 + m_parameters.robotVerticalOffset()),
            restrictToPositive(robotPosition.top - (outHeight - 1) / 2)),
        cv::Size(outWidth, outHeight));
    paddedImage(roi).copyTo(image);
    cv::flip(image, image, 1);
    cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
}

void OccupancyGridImageDrawer::adjustPaddingForCenteredRobotOffset(
    DirectionalValues& padding,
    int width,
    const DirectionalValues& robotPosition)
{
    // The vertical axis on which we apply the robot vertical offset is horizontal
    // currently as we will rotate it later
    int& topPaddingRotated = padding.right;
    int& bottomPaddingRotated = padding.left;
    const int& heightRotated = width;
    const int& bottomPositionRotated = robotPosition.left;
    if (m_parameters.robotVerticalOffset() > 0)
    {
        // If the offset is positive, we need to add padding to the top and can
        // possibly reduce padding to the bottom
        bottomPaddingRotated -= std::min(bottomPaddingRotated, restrictToPositive(m_parameters.robotVerticalOffset()));
        topPaddingRotated += restrictToPositive(m_parameters.robotVerticalOffset());
    }
    if (m_parameters.robotVerticalOffset() < 0)
    {
        // If the offset is negative, we need to recompute padding to the bottom and
        // can possibly reduce padding to the top
        bottomPaddingRotated = restrictToPositive(
            (heightRotated - 1) / 2 + restrictToPositive(-m_parameters.robotVerticalOffset()) - bottomPositionRotated);
        topPaddingRotated -= std::min(topPaddingRotated, restrictToPositive(-m_parameters.robotVerticalOffset()));
    }
}

#ifndef OCCUPANCY_GRID_IMAGE_DRAWER_H
#define OCCUPANCY_GRID_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"
#include "opentera_webrtc_ros_msgs/srv/change_map_view.hpp"

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>

namespace map_image_generator
{
    class OccupancyGridImageDrawer : public ImageDrawer
    {
        Parameters& m_mutableParameters;

        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr m_occupancyGridSubscriber;
        nav_msgs::msg::OccupancyGrid::ConstSharedPtr m_lastOccupancyGrid;
        rclcpp::Service<opentera_webrtc_ros_msgs::srv::ChangeMapView>::SharedPtr m_mapViewChangerService;

        cv::Mat m_notScaledOccupancyGridImage;
        cv::Mat m_scaledOccupancyGridImage;
        cv::Mat m_zoomedOccupancyGridImage;

    public:
        OccupancyGridImageDrawer(Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer);
        ~OccupancyGridImageDrawer() override;

        void draw(cv::Mat& image) override;

    private:
        void occupancyGridCallback(const nav_msgs::msg::OccupancyGrid::ConstSharedPtr& occupancyGrid);

        void changeMapViewCallback(
            const opentera_webrtc_ros_msgs::srv::ChangeMapView::Request::ConstSharedPtr& req,
            const opentera_webrtc_ros_msgs::srv::ChangeMapView::Response::SharedPtr& res);

        void drawNotScaledOccupancyGridImage();
        void changeNotScaledOccupancyGridImageIfNeeded();

        void scaleOccupancyGridImage();
        void changeScaledOccupancyGridImageIfNeeded();

        void rotateImageAboutPoint(cv::Mat& image, double angle, const cv::Point& point) const;
        void rotateImageAboutCenter(cv::Mat& image, double angle) const;

        void drawOccupancyGridImage(cv::Mat& image);
        void drawOccupancyGridImageCenteredAroundRobot(cv::Mat& image);

        std::optional<tf2::Transform> getRobotTransform() const;

        struct DirectionalValues
        {
            int top;
            int bottom;
            int left;
            int right;
        };
        struct MapCoordinates
        {
            int x;
            int y;
        };

        static DirectionalValues computePadding(const DirectionalValues& position, int height, int width);
        void adjustPaddingForCenteredRobotOffset(
            DirectionalValues& padding,
            int width,
            const DirectionalValues& robotPosition);

        const cv::Mat& getZoomedOccupancyImage();

        MapCoordinates getMapCoordinatesFromTf(const tf2::Transform& transform) const;
        static DirectionalValues
            getDirectionsFromMapCoordinates(const MapCoordinates& mapCoordinates, const cv::Mat& map);
    };
}
#endif

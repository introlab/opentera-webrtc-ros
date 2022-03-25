#ifndef OCCUPANCY_GRID_IMAGE_DRAWER_H
#define OCCUPANCY_GRID_IMAGE_DRAWER_H

#include "map_image_generator/ChangeMapView.h"
#include "map_image_generator/drawers/ImageDrawer.h"

#include <nav_msgs/OccupancyGrid.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

namespace map_image_generator
{
    class OccupancyGridImageDrawer : public ImageDrawer
    {
        Parameters& m_mutableParameters;

        ros::Subscriber m_occupancyGridSubscriber;
        nav_msgs::OccupancyGrid::ConstPtr m_lastOccupancyGrid;
        ros::ServiceServer m_mapViewChangerService;

        cv::Mat m_notScaledOccupancyGridImage;
        cv::Mat m_scaledOccupancyGridImage;
        cv::Mat m_zoomedOccupancyGridImage;

    public:
        OccupancyGridImageDrawer(
            Parameters& parameters,
            ros::NodeHandle& nodeHandle,
            tf::TransformListener& tfListener);
        ~OccupancyGridImageDrawer() override;

        void draw(cv::Mat& image) override;

    private:
        void occupancyGridCallback(const nav_msgs::OccupancyGrid::ConstPtr& occupancyGrid);

        bool changeMapViewCallback(ChangeMapView::Request& req, ChangeMapView::Response& res);

        void drawNotScaledOccupancyGridImage();
        void changeNotScaledOccupancyGridImageIfNeeded();

        void scaleOccupancyGridImage();
        void changeScaledOccupancyGridImageIfNeeded();

        void rotateImageAboutPoint(cv::Mat& image, double angle, const cv::Point& point) const;
        void rotateImageAboutCenter(cv::Mat& image, double angle) const;

        void drawOccupancyGridImage(cv::Mat& image);
        void drawOccupancyGridImageCenteredAroundRobot(cv::Mat& image);

        // Replace with std::optional in C++17
        std::unique_ptr<tf::Transform> getRobotTransform() const;

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

        MapCoordinates getMapCoordinatesFromTf(const tf::Transform& transform) const;
        static DirectionalValues
            getDirectionsFromMapCoordinates(const MapCoordinates& mapCoordinates, const cv::Mat& map);
    };
}
#endif

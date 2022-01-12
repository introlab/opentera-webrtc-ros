#ifndef OCCUPANCY_GRID_IMAGE_DRAWER_H
#define OCCUPANCY_GRID_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

#include <nav_msgs/OccupancyGrid.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

namespace map_image_generator
{
    class OccupancyGridImageDrawer : public ImageDrawer
    {
        ros::Subscriber m_occupancyGridSubscriber;
        nav_msgs::OccupancyGrid::ConstPtr m_lastOccupancyGrid;

        cv::Mat m_notScaledOccupancyGridImage;
        cv::Mat m_scaledOccupancyGridImage;

    public:
        OccupancyGridImageDrawer(const Parameters& parameters,
                                 ros::NodeHandle& nodeHandle,
                                 tf::TransformListener& tfListener);
        virtual ~OccupancyGridImageDrawer();

        virtual void draw(cv::Mat& image);

    private:
        void
        occupancyGridCallback(const nav_msgs::OccupancyGrid::ConstPtr& occupancyGrid);

        void drawNotScaledOccupancyGridImage();
        void changeNotScaledOccupancyGridImageIfNeeded();

        void scaleOccupancyGridImage();
        void changeScaledOccupancyGridImageIfNeeded();

        void rotateImageAboutCenter(cv::Mat& image, double angle);

        void drawOccupancyGridImage(cv::Mat& image);
        void drawOccupancyGridImageCenteredAroundRobot(cv::Mat& image);


        void convertMapInfoToMapCoordinates(const nav_msgs::MapMetaData& mapInfo, int& x,
                                            int& y);
    };
}
#endif

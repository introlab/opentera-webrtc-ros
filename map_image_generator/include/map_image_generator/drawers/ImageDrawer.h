#ifndef IMAGE_DRAWER_H
#define IMAGE_DRAWER_H

#include "map_image_generator/Parameters.h"
#include "map_image_generator/utils.h"

#include <optional>
#include <nav_msgs/msg/map_meta_data.hpp>
#include <tf2_ros/buffer.h>


namespace map_image_generator
{
    class ImageDrawer
    {
    protected:
        const Parameters& m_parameters;
        rclcpp::Node& m_node;
        tf2_ros::Buffer& m_tfBuffer;

        ImageDrawer(const ImageDrawer&) = default;
        ImageDrawer(ImageDrawer&&) = default;

    public:
        ImageDrawer(const Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer);
        virtual ~ImageDrawer();

        ImageDrawer& operator=(const ImageDrawer&) = delete;
        ImageDrawer& operator=(ImageDrawer&&) = delete;

        virtual void draw(cv::Mat& image) = 0;

    protected:
        void convertTransformToMapCoordinates(const tf2::Transform& transform, int& x, int& y) const;
        void convertTransformToInputMapCoordinates(
            const tf2::Transform& transform,
            const nav_msgs::msg::MapMetaData& mapInfo,
            int& x,
            int& y) const;
        void convertInputMapCoordinatesToTransform(
            int x,
            int y,
            const nav_msgs::msg::MapMetaData& mapInfo,
            tf2::Transform& transform) const;

        std::optional<tf2::Transform> getTransformInRef(const std::string& frameId) const;

        void adjustTransformForRobotRef(tf2::Transform& transform) const;

    private:
        void adjustTransformAngleForRobotRef(tf2::Transform& transform) const;
    };
}
#endif

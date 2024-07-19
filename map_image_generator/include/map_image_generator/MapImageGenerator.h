#ifndef MAP_IMAGE_GENERATOR_H
#define MAP_IMAGE_GENERATOR_H

#include "map_image_generator/Parameters.h"
#include "map_image_generator/drawers/ImageDrawer.h"

#include <cv_bridge/cv_bridge.h>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/buffer.h>
#include <vector>

namespace map_image_generator
{
    class MapImageGenerator
    {
        const Parameters& m_parameters;
        rclcpp::Node& m_node;
        tf2_ros::Buffer& m_tfBuffer;

        std::vector<std::unique_ptr<ImageDrawer>> m_drawers;

        cv_bridge::CvImage m_cvImage;

    public:
        MapImageGenerator(Parameters& parameters, rclcpp::Node& node, tf2_ros::Buffer& tfBuffer);
        virtual ~MapImageGenerator();

        void generate(sensor_msgs::msg::Image& sensorImage);
    };
}
#endif

#ifndef MAP_IMAGE_GENERATOR_H
#define MAP_IMAGE_GENERATOR_H

#include "map_image_generator/Parameters.h"
#include "map_image_generator/drawers/ImageDrawer.h"

#include <memory>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tf/transform_listener.h>
#include <vector>

namespace map_image_generator
{
    class MapImageGenerator
    {
        const Parameters& m_parameters;
        ros::NodeHandle& m_nodeHandle;
        tf::TransformListener& m_tfListener;

        std::vector<std::unique_ptr<ImageDrawer>> m_drawers;

        cv_bridge::CvImage m_cvImage;

    public:
        MapImageGenerator(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                          tf::TransformListener& tfListener,
                          geometry_msgs::PoseStamped::Ptr activeGoal);
        virtual ~MapImageGenerator();

        void generate(sensor_msgs::Image& sensorImage);
    };
}
#endif

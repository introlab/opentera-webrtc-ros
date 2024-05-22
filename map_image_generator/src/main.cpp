#include "map_image_generator/GoalConverter.h"
#include "map_image_generator/MapImageGenerator.h"
#include "map_image_generator/MapLabelsConverter.h"
#include "map_image_generator/Parameters.h"

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using namespace map_image_generator;

class Node : public rclcpp::Node
{
    Parameters m_parameters;
    tf2_ros::Buffer m_tfBuffer;
    tf2_ros::TransformListener m_tfListener;
    MapImageGenerator m_mapImageGenerator;
    GoalConverter m_goalConverter;
    MapLabelsConverter m_mapLabelsConverter;

    image_transport::ImageTransport m_imageTransport;
    image_transport::Publisher m_mapImagePublisher;
    sensor_msgs::msg::Image m_mapImage;

public:
    explicit Node()
        : rclcpp::Node{"map_image_generator"},
          m_parameters{*this},
          m_tfBuffer{get_clock()},
          m_tfListener{m_tfBuffer},
          m_mapImageGenerator{m_parameters, *this, m_tfBuffer},
          m_goalConverter{m_parameters, *this, m_tfBuffer},
          m_mapLabelsConverter{m_parameters, *this},
          m_imageTransport{this->shared_from_this()},
          m_mapImagePublisher{m_imageTransport.advertise("map_image", 1)},
          m_mapImage{}
    {
    }

    void run()
    {
        RCLCPP_INFO(this->get_logger(), "MapImage initialized, starting image generation after first cycle...");

        rclcpp::Rate loop_rate{m_parameters.refreshRate()};
        if (rclcpp::ok())
        {
            loop_rate.sleep();
            rclcpp::spin_some(this->shared_from_this());
        }

        while (rclcpp::ok())
        {
            m_mapImageGenerator.generate(m_mapImage);
            m_mapImagePublisher.publish(m_mapImage);

            loop_rate.sleep();
            rclcpp::spin_some(this->shared_from_this());
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<Node>();

    node->run();

    return 0;
}

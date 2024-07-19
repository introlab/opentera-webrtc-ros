#ifndef MAP_LABELS_CONVERTER_H
#define MAP_LABELS_CONVERTER_H

#include "map_image_generator/Parameters.h"
#include "map_image_generator/utils.h"

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <rtabmap_msgs/srv/list_labels.hpp>

namespace map_image_generator
{
    class MapLabelsConverter
    {
        const Parameters& m_parameters;
        rclcpp::Node& m_node;
        rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr m_mapLabelsSubscriber;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr m_mapLabelsPublisher;
        rclcpp::Client<rtabmap_msgs::srv::ListLabels>::SharedPtr m_rtabmapListLabelsServiceClient;

        ServiceClientPruner m_pruner;

    public:
        MapLabelsConverter(const Parameters& parameters, rclcpp::Node& node);
        virtual ~MapLabelsConverter();

    private:
        void mapLabelsCallback(const visualization_msgs::msg::MarkerArray::ConstSharedPtr& mapLabels);
    };
}

#endif

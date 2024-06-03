#include "map_image_generator/MapLabelsConverter.h"

#include "map_image_generator/utils.h"

#include <algorithm>

using namespace map_image_generator;
using namespace std::chrono_literals;

MapLabelsConverter::MapLabelsConverter(const Parameters& parameters, rclcpp::Node& node)
    : m_parameters(parameters),
      m_node(node),
      m_mapLabelsSubscriber{m_node.create_subscription<visualization_msgs::msg::MarkerArray>(
          "map_labels",
          1,
          bind_this<visualization_msgs::msg::MarkerArray>(this, &MapLabelsConverter::mapLabelsCallback))},
      m_mapLabelsPublisher{m_node.create_publisher<visualization_msgs::msg::MarkerArray>("map_image_labels", 1)},
      m_rtabmapListLabelsServiceClient{
          m_node.create_client<rtabmap_msgs::srv::ListLabels>("rtabmap_list_label_service")}
{
}

MapLabelsConverter::~MapLabelsConverter() = default;

void MapLabelsConverter::mapLabelsCallback(const visualization_msgs::msg::MarkerArray::ConstSharedPtr& mapLabels)
{
    std::vector<std::string> desiredLabels = getDesiredLabels();

    visualization_msgs::msg::MarkerArray mapImageLabels;
    for (const auto& marker : mapLabels->markers)
    {
        if (find(desiredLabels.begin(), desiredLabels.end(), marker.text) == desiredLabels.end())
        {
            continue;
        }

        visualization_msgs::msg::Marker imageMarker;
        imageMarker.header.stamp = marker.header.stamp;
        imageMarker.header.frame_id = "map_image";

        imageMarker.pose = convertMapToMapImage(m_parameters, marker.pose);
        imageMarker.text = marker.text;
        mapImageLabels.markers.push_back(imageMarker);
    }
    m_mapLabelsPublisher->publish(mapImageLabels);
}

std::vector<std::string> MapLabelsConverter::getDesiredLabels()
{
    static constexpr auto service_call_timeout = 2s;

    auto request = std::make_shared<rtabmap_msgs::srv::ListLabels::Request>();

    auto result = m_rtabmapListLabelsServiceClient->async_send_request(request);
    if (rclcpp::spin_until_future_complete(m_node.shared_from_this(), result, service_call_timeout) ==
        rclcpp::FutureReturnCode::SUCCESS)
    {
        return result.get()->labels;
    }
    return std::vector<std::string>();
}

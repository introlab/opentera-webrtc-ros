#include "map_image_generator/MapLabelsConverter.h"

#include "map_image_generator/utils.h"

#include <algorithm>
#include <rtabmap_ros/ListLabels.h>

using namespace map_image_generator;
using namespace std;

MapLabelsConverter::MapLabelsConverter(const Parameters& parameters, ros::NodeHandle& nodeHandle)
    : m_parameters(parameters),
      m_nodeHandle(nodeHandle)
{
    m_mapLabelsSubscriber = m_nodeHandle.subscribe("map_labels", 1, &MapLabelsConverter::mapLabelsCallback, this);
    m_mapLabelsPublisher = m_nodeHandle.advertise<visualization_msgs::MarkerArray>("map_image_labels", 1);
    m_rtabmapListLabelsServiceClient =
        m_nodeHandle.serviceClient<rtabmap_ros::ListLabels>("rtabmap_list_label_service");
}

MapLabelsConverter::~MapLabelsConverter() = default;

void MapLabelsConverter::mapLabelsCallback(const visualization_msgs::MarkerArray::ConstPtr& mapLabels)
{
    std::vector<std::string> desiredLabels = getDesiredLabels();

    visualization_msgs::MarkerArray mapImageLabels;
    for (const auto& marker : mapLabels->markers)
    {
        if (find(desiredLabels.begin(), desiredLabels.end(), marker.text) == desiredLabels.end())
        {
            continue;
        }

        visualization_msgs::Marker imageMarker;
        imageMarker.header.seq = marker.header.seq;
        imageMarker.header.stamp = marker.header.stamp;
        imageMarker.header.frame_id = "map_image";

        imageMarker.pose = convertMapToMapImage(m_parameters, marker.pose);
        imageMarker.text = marker.text;
        mapImageLabels.markers.push_back(imageMarker);
    }
    m_mapLabelsPublisher.publish(mapImageLabels);
}

std::vector<std::string> MapLabelsConverter::getDesiredLabels()
{
    rtabmap_ros::ListLabels service;
    if (m_rtabmapListLabelsServiceClient.call(service))
    {
        return service.response.labels;
    }
    return std::vector<std::string>();
}

#include "map_image_generator/MapLabelsConverter.h"
#include "map_image_generator/utils.h"

#include <rtabmap_ros/ListLabels.h>

#include <algorithm>

using namespace map_image_generator;
using namespace std;

MapLabelsConverter::MapLabelsConverter(const Parameters& parameters,
    ros::NodeHandle& nodeHandle) :
    m_parameters(parameters),
    m_nodeHandle(nodeHandle)
{
    m_mapLabelsSubscriber = m_nodeHandle.subscribe("map_labels", 1,
        &MapLabelsConverter::mapLabelsCallback, this);
    m_mapLabelsPublisher = m_nodeHandle.advertise<visualization_msgs::MarkerArray>("map_image_labels", 1);
    m_rtabmapListLabelsServiceClient = m_nodeHandle.serviceClient<rtabmap_ros::ListLabels>("rtabmap_list_label_service");
}

MapLabelsConverter::~MapLabelsConverter()
{    
}

void MapLabelsConverter::mapLabelsCallback(const visualization_msgs::MarkerArray::ConstPtr& mapLabels)
{
    typedef visualization_msgs::MarkerArray::_markers_type::const_iterator MarkersConstIterator;

    std::vector<std::string> desiredLabels = getDesiredLabels();

    visualization_msgs::MarkerArray mapImageLabels;    
    for (MarkersConstIterator it = mapLabels->markers.begin(); it != mapLabels->markers.end(); ++it)
    {
        if (find(desiredLabels.begin(), desiredLabels.end(), it->text) == desiredLabels.end())
        {
            continue;
        }

        visualization_msgs::Marker marker;
        marker.header.seq = it->header.seq;
        marker.header.stamp = it->header.stamp;
        marker.header.frame_id = "map_image";

        marker.pose = convertMapToMapImage(m_parameters, it->pose);
        marker.text = it->text;
        mapImageLabels.markers.push_back(marker);
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

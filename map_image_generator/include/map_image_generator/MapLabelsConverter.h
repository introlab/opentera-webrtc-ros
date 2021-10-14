#ifndef MAP_LABELS_CONVERTER_H
#define MAP_LABELS_CONVERTER_H

#include "map_image_generator/Parameters.h"

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include <string>
#include <vector>

namespace map_image_generator
{
    class MapLabelsConverter
    {
        const Parameters& m_parameters;
        ros::NodeHandle& m_nodeHandle;
        ros::Subscriber m_mapLabelsSubscriber;
        ros::Publisher m_mapLabelsPublisher;
        ros::ServiceClient m_rtabmapListLabelsServiceClient;
        
    public:
        MapLabelsConverter(const Parameters& parameters, ros::NodeHandle& nodeHandle);
        virtual ~MapLabelsConverter();

    private:
        void mapLabelsCallback(const visualization_msgs::MarkerArray::ConstPtr& mapLabels);
        std::vector<std::string> getDesiredLabels();
    };
}

#endif

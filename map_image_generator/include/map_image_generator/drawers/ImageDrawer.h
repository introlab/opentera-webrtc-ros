#ifndef IMAGE_DRAWER_H
#define IMAGE_DRAWER_H

#include "map_image_generator/Parameters.h"

#include <cv_bridge/cv_bridge.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

namespace map_image_generator
{
    class ImageDrawer
    {
    protected:
        const Parameters& m_parameters;
        ros::NodeHandle& m_nodeHandle;
        tf::TransformListener& m_tfListener;

    public:
        ImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle, tf::TransformListener& tfListener);
        virtual ~ImageDrawer();
        
        virtual void draw(cv::Mat& image) = 0;

    protected:
        void convertTransformToMapCoordinates(const tf::Transform& transform, int& x, int& y);
    };
}
#endif
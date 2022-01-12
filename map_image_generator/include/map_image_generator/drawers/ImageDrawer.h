#ifndef IMAGE_DRAWER_H
#define IMAGE_DRAWER_H

#include "map_image_generator/Parameters.h"

// Replace with <optional> in C++17
#include <cv_bridge/cv_bridge.h>
#include <memory>
#include <nav_msgs/MapMetaData.h>
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
        ImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                    tf::TransformListener& tfListener);
        virtual ~ImageDrawer();

        virtual void draw(cv::Mat& image) = 0;

    protected:
        void convertTransformToMapCoordinates(const tf::Transform& transform, int& x,
                                              int& y);
        void convertTransformToInputMapCoordinates(const tf::Transform& transform,
                                                   const nav_msgs::MapMetaData& mapInfo,
                                                   int& x, int& y);
        // Replace with std::optional in C++17
        std::unique_ptr<tf::Transform> getRefCenteredMap() const;
    };

    // Replace with std::optional in C++17
    inline std::unique_ptr<tf::Transform> ImageDrawer::getRefCenteredMap() const
    {
        tf::StampedTransform refTransform;

        try
        {
            m_tfListener.lookupTransform(m_parameters.refFrameId(),
                                         m_parameters.mapFrameId(), ros::Time(0),
                                         refTransform);
        }
        catch (tf::TransformException& ex)
        {
            ROS_ERROR("%s", ex.what());
            return {};
        }

        // Replace with std::optional in C++17
        auto transform = std::make_unique<tf::Transform>();
        transform->setIdentity();
        transform->setOrigin(refTransform.getOrigin());
        return transform;
    }
}
#endif

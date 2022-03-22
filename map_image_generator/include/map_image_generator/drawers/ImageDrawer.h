#ifndef IMAGE_DRAWER_H
#define IMAGE_DRAWER_H

#include "map_image_generator/Parameters.h"
#include "map_image_generator/utils.h"

#include <experimental/optional>  // Replace with <optional> in C++17
#include <nav_msgs/MapMetaData.h>
#include <tf/transform_listener.h>

namespace std  // Replace with <optional> onlu here in C++17
{
    using std::experimental::optional;
}

namespace map_image_generator
{
    class ImageDrawer
    {
    protected:
        const Parameters& m_parameters;
        ros::NodeHandle& m_nodeHandle;
        tf::TransformListener& m_tfListener;

        ImageDrawer(const ImageDrawer&) = default;
        ImageDrawer(ImageDrawer&&) = default;

    public:
        ImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle, tf::TransformListener& tfListener);
        virtual ~ImageDrawer();

        ImageDrawer& operator=(const ImageDrawer&) = delete;
        ImageDrawer& operator=(ImageDrawer&&) = delete;

        virtual void draw(cv::Mat& image) = 0;

    protected:
        void convertTransformToMapCoordinates(const tf::Transform& transform, int& x, int& y) const;
        void convertTransformToInputMapCoordinates(
            const tf::Transform& transform,
            const nav_msgs::MapMetaData& mapInfo,
            int& x,
            int& y) const;
        void convertInputMapCoordinatesToTransform(
            int x,
            int y,
            const nav_msgs::MapMetaData& mapInfo,
            tf::Transform& transform) const;

        std::optional<tf::Transform> getTransformInRef(const std::string& frameId) const;

        void adjustTransformForRobotRef(tf::Transform& transform) const;

    private:
        void adjustTransformAngleForRobotRef(tf::Transform& transform) const;
    };
}
#endif

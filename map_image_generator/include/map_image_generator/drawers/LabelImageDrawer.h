#ifndef LABEL_IMAGE_DRAWER_H
#define LABEL_IMAGE_DRAWER_H

#include "map_image_generator/drawers/ImageDrawer.h"

namespace map_image_generator
{
    class LabelImageDrawer : public ImageDrawer
    {
    public:
        LabelImageDrawer(const Parameters& parameters, ros::NodeHandle& nodeHandle,
                         tf::TransformListener& tfListener);
        ~LabelImageDrawer() override;

        void draw(cv::Mat& image) override;

    private:
        void drawLabel(const geometry_msgs::PoseStamped& label, const std::string& text,
                       cv::Mat& image, tf::Transform& transform);
    };
}
#endif

#include "face_following/Parameters.h"

#include <type_traits>

using namespace face_following;
namespace internal
{
    template<typename T>
    T getParam(const std::string& name, const T& defaultValue)
    {
        ros::NodeHandle nh{"~"};
        T value;
        nh.param<T>(name, value, defaultValue);
        return value;
    }

}

Parameters::Parameters(ros::NodeHandle& nodeHandle)
{
    using namespace internal;

    m_refreshRate = getParam<double>("refresh_rate", 30.0);
    m_width = getParam<int>("width", 400);
    m_height = getParam<int>("height", 400);

    m_framesUsedForStabilizer = getParam<int>("frames_used_for_stabilizer", 15);

    m_minWidthChange = getParam<float>("min_width_change", 0.2);
    m_minHeightChange = getParam<float>("min_height_change", 0.4);
    m_minXChange = getParam<float>("min_x_change", 0.1);
    m_minYChange = getParam<float>("min_y_change", 0.1);

    m_rightMargin = getParam<float>("right_margin", 0.2);
    m_leftMargin = getParam<float>("left_margin", 0.2);
    m_topMargin = getParam<float>("top_margin", 0.2);
    m_bottomMargin = getParam<float>("bottom_margin", 0.25);

    m_isPeerImage = getParam<bool>("is_peer_image", false);

    std::string name = getParam<std::string>("name", "face_following");
    m_dnnModelPath = getParam<std::string>(
        "dnn_model_path",
        ros::package::getPath(name) + "/models/model/res10_300x300_ssd_iter_140000.caffemodel");
    m_dnnDeployPath = getParam<std::string>("dnn_deploy_path", ros::package::getPath(name) + "/models/deploy.prototxt");

    m_useGpu = getParam<bool>("use_gpu", false);
}

Parameters::~Parameters() = default;

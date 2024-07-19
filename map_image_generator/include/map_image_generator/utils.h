#ifndef UTILS_H
#define UTILS_H

#include "map_image_generator/Parameters.h"

#include <cmath>
#include <type_traits>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/LinearMath/Scalar.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace map_image_generator
{
    tf2::Quaternion createQuaternionFromYaw(double yaw);

    geometry_msgs::msg::PoseStamped
        convertMapToMapImage(const Parameters& parameters, const geometry_msgs::msg::PoseStamped& mapPose);
    geometry_msgs::msg::Pose
        convertMapToMapImage(const Parameters& parameters, const geometry_msgs::msg::Pose& mapPose);

    geometry_msgs::msg::Pose
        convertRobotCenteredMapCoordinatesToPose(const Parameters& parameters, int x, int y, double yaw);

    geometry_msgs::msg::PoseStamped
        convertMapImageToRobot(const Parameters& parameters, const geometry_msgs::msg::PoseStamped& mapImagePose);
    geometry_msgs::msg::Pose
        convertMapImageToRobot(const Parameters& parameters, const geometry_msgs::msg::Pose& mapImagePose);

    geometry_msgs::msg::PoseStamped
        convertMapImageToMap(const Parameters& parameters, const geometry_msgs::msg::PoseStamped& mapImagePose);
    geometry_msgs::msg::Pose
        convertMapImageToMap(const Parameters& parameters, const geometry_msgs::msg::Pose& mapImagePose);

    void offsetYawByMinus90Degrees(geometry_msgs::msg::Pose& pose);

    TF2SIMD_FORCE_INLINE tf2Scalar tf2NormalizeAngle0To2Pi(tf2Scalar angleInRadians)
    {
        auto angle = tf2NormalizeAngle(angleInRadians);
        return angle >= 0 ? angle : angle + TF2SIMD_2_PI;
    }

    cv::Scalar interpolateColors(const cv::Scalar& color1, const cv::Scalar& color2, double ratio);

    void flipYawOnY(geometry_msgs::msg::Pose& pose);
    void flipYawOnY(tf2::Transform& transform);
    double flipYawOnY(double yaw);

    template<typename T>
    inline T sign(T val)
    {
        return static_cast<T>(T{0} < val) - static_cast<T>(val < T{0});
    }

    template<typename T>
    inline T restrictToPositive(T val)
    {
        return std::max(T{0}, val);
    }

    template<typename T>
    inline T deg2rad(T deg)
    {
        return deg * M_PI / 180;
    }

    template<typename T>
    inline T rad2deg(T rad)
    {
        return rad * 180 / M_PI;
    }

    inline bool areApproxEqual(double a, double b)
    {
        return std::abs(a - b) < 0.00001;
    }

    inline int ceilDivision(int dividend, double divisor)
    {
        return static_cast<int>(std::ceil(static_cast<double>(dividend) / divisor));
    }

    template<typename T, typename = void>
    struct has_request_type : std::false_type
    {
    };

    template<typename T>
    struct has_request_type<T, std::void_t<typename T::Request>> : std::true_type
    {
    };


    template<typename T, typename = void>
    struct has_response_type : std::false_type
    {
    };

    template<typename T>
    struct has_response_type<T, std::void_t<typename T::Response>> : std::true_type
    {
    };

    template<typename T>
    struct has_request_response_types : std::conjunction<has_request_type<T>, has_response_type<T>>
    {
    };

    template<typename MessageT, typename This, typename Func>
    inline auto bind_this(This* self, Func&& func) -> std::enable_if_t<
                                                       !has_request_response_types<MessageT>::value,
                                                       std::function<void(const typename MessageT::SharedPtr)>>
    {
        return [self, func](const typename MessageT::SharedPtr msg) { (self->*func)(msg); };
    }

    template<typename MessageT, typename This, typename Func>
    inline auto bind_this(This* self, Func&& func)
        -> std::enable_if_t<
            has_request_response_types<MessageT>::value,
            std::function<
                void(const typename MessageT::Request::SharedPtr, const typename MessageT::Response::SharedPtr)>>
    {
        return [self,
                func](const typename MessageT::Request::SharedPtr req, const typename MessageT::Response::SharedPtr res)
        { (self->*func)(req, res); };
    }

    class ServiceClientPruner
    {
    public:
        template<typename... Clients>
        explicit ServiceClientPruner(rclcpp::Node& node, std::chrono::seconds timeout, Clients&... clients)
            : m_prune_timer{node.create_wall_timer(
                  timeout,
                  [&node, timeout, clients...]()
                  {
                      std::vector<std::int64_t> pruned_requests;

                      const auto cb = [&node, timeout, &pruned_requests](auto client)
                      {
                          std::size_t n_pruned = client->prune_requests_older_than(
                              std::chrono::system_clock::now() - timeout,
                              &pruned_requests);

                          if (n_pruned > 0)
                          {
                              RCLCPP_ERROR_STREAM(
                                  node.get_logger(),
                                  "Service call for " << client->get_service_name() << ": "
                                                      << "the server hasn't replied for more than " << timeout.count()
                                                      << "s, " << n_pruned
                                                      << " requests were discarded, "
                                                         "the discarded requests numbers are:");
                              for (const auto& req_num : pruned_requests)
                              {
                                  RCLCPP_ERROR_STREAM(node.get_logger(), "\t" << req_num);
                              }
                          }
                          pruned_requests.clear();
                      };

                      (cb(clients), ...);
                  })}
        {
            static_assert(sizeof...(clients) >= 1, "Requires at least one service client");
        }

    private:
        rclcpp::TimerBase::SharedPtr m_prune_timer;
    };
}

#endif

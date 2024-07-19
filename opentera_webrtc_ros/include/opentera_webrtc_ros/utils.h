#ifndef UTILS_H
#define UTILS_H

#include <rclcpp/rclcpp.hpp>
#include <cstdint>
#include <chrono>
#include <type_traits>
#include <functional>
#include <builtin_interfaces/msg/time.hpp>

namespace details
{
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
}

template<typename MessageT, typename This, typename Func>
inline auto bind_this(This* self, Func&& func) -> std::enable_if_t<
                                                   !details::has_request_response_types<MessageT>::value,
                                                   std::function<void(const typename MessageT::SharedPtr)>>
{
    return [self, func](const typename MessageT::SharedPtr msg) { (self->*func)(msg); };
}

template<typename MessageT, typename This, typename Func>
inline auto bind_this(This* self, Func&& func)
    -> std::enable_if_t<
        details::has_request_response_types<MessageT>::value,
        std::function<void(const typename MessageT::Request::SharedPtr, const typename MessageT::Response::SharedPtr)>>
{
    return
        [self, func](const typename MessageT::Request::SharedPtr req, const typename MessageT::Response::SharedPtr res)
    { (self->*func)(req, res); };
}

inline auto to_nanoseconds(const builtin_interfaces::msg::Time& time) -> std::uint64_t
{
    return time.sec * 1'000'000'000ULL + time.nanosec;
}

inline auto from_nanoseconds(std::uint64_t time) -> builtin_interfaces::msg::Time
{
    builtin_interfaces::msg::Time result;
    result.sec = time / 1'000'000'000ULL;
    result.nanosec = time % 1'000'000'000ULL;

    return result;
}

inline auto to_microseconds(const builtin_interfaces::msg::Time& time) -> std::uint64_t
{
    return to_nanoseconds(time) / 1'000ULL;
}

inline auto from_microseconds(std::uint64_t time) -> builtin_interfaces::msg::Time
{
    return from_nanoseconds(time * 1'000ULL);
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

#endif

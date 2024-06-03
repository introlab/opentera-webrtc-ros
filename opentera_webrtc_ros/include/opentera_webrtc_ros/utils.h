#ifndef UTILS_H
#define UTILS_H

#include <cstdint>
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
    return time.sec * 1'000'000'000 + time.nanosec;
}

inline auto from_nanoseconds(std::uint64_t time) -> builtin_interfaces::msg::Time
{
    builtin_interfaces::msg::Time result;
    result.sec = time / 1'000'000'000;
    result.nanosec = time % 1'000'000'000;

    return result;
}

inline auto to_microseconds(const builtin_interfaces::msg::Time& time) -> std::uint64_t
{
    return to_nanoseconds(time) / 1'000;
}

inline auto from_microseconds(std::uint64_t time) -> builtin_interfaces::msg::Time
{
    return from_nanoseconds(time * 1'000);
}

#endif

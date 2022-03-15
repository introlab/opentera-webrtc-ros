#include <numeric>
#include <ros/ros.h>

using namespace ros;

namespace opentera
{
    namespace param
    {
        ///////////////////////////////////////////////////////////////////////////////
        // The content of namespace opentera::param is directly taken and/or slightly
        // modified from the ROS implementation, which has the following copyright:
        ///////////////////////////////////////////////////////////////////////////////
        /*
         * Copyright (C) 2009, Willow Garage, Inc.
         *
         * Redistribution and use in source and binary forms, with or without
         * modification, are permitted provided that the following conditions are met:
         *   * Redistributions of source code must retain the above copyright notice,
         *     this list of conditions and the following disclaimer.
         *   * Redistributions in binary form must reproduce the above copyright
         *     notice, this list of conditions and the following disclaimer in the
         *     documentation and/or other materials provided with the distribution.
         *   * Neither the names of Willow Garage, Inc. nor the names of its
         *     contributors may be used to endorse or promote products derived from
         *     this software without specific prior written permission.
         *
         * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
         * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
         * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
         * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
         * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
         * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
         * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
         * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
         * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
         * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
         * POSSIBILITY OF SUCH DAMAGE.
         */
        namespace impl
        {
            template <class T>
            T xml_cast(const XmlRpc::XmlRpcValue& xml_value)
            {
                return static_cast<T>(xml_value);
            }

            template <class T>
            bool xml_castable(int XmlType)
            {
                return false;
            }

            template <>
            bool xml_castable<std::string>(int XmlType)
            {
                return XmlType == XmlRpc::XmlRpcValue::TypeString;
            }

            template <>
            bool xml_castable<double>(int XmlType)
            {
                return (XmlType == XmlRpc::XmlRpcValue::TypeDouble
                        || XmlType == XmlRpc::XmlRpcValue::TypeInt
                        || XmlType == XmlRpc::XmlRpcValue::TypeBoolean);
            }

            template <>
            bool xml_castable<float>(int XmlType)
            {
                return (XmlType == XmlRpc::XmlRpcValue::TypeDouble
                        || XmlType == XmlRpc::XmlRpcValue::TypeInt
                        || XmlType == XmlRpc::XmlRpcValue::TypeBoolean);
            }

            template <>
            bool xml_castable<int>(int XmlType)
            {
                return (XmlType == XmlRpc::XmlRpcValue::TypeDouble
                        || XmlType == XmlRpc::XmlRpcValue::TypeInt
                        || XmlType == XmlRpc::XmlRpcValue::TypeBoolean);
            }

            template <>
            bool xml_castable<bool>(int XmlType)
            {
                return (XmlType == XmlRpc::XmlRpcValue::TypeDouble
                        || XmlType == XmlRpc::XmlRpcValue::TypeInt
                        || XmlType == XmlRpc::XmlRpcValue::TypeBoolean);
            }

            template <>
            double xml_cast(const XmlRpc::XmlRpcValue& xml_value)
            {
                using namespace XmlRpc;
                switch (xml_value.getType())
                {
                    case XmlRpcValue::TypeDouble:
                        return static_cast<double>(xml_value);
                    case XmlRpcValue::TypeInt:
                        return static_cast<double>(static_cast<int>(xml_value));
                    case XmlRpcValue::TypeBoolean:
                        return static_cast<double>(static_cast<bool>(xml_value));
                    default:
                        return 0.0;
                };
            }

            template <>
            float xml_cast(const XmlRpc::XmlRpcValue& xml_value)
            {
                using namespace XmlRpc;
                switch (xml_value.getType())
                {
                    case XmlRpcValue::TypeDouble:
                        return static_cast<float>(static_cast<double>(xml_value));
                    case XmlRpcValue::TypeInt:
                        return static_cast<float>(static_cast<int>(xml_value));
                    case XmlRpcValue::TypeBoolean:
                        return static_cast<float>(static_cast<bool>(xml_value));
                    default:
                        return 0.0f;
                };
            }

            template <>
            int xml_cast(const XmlRpc::XmlRpcValue& xml_value)
            {
                using namespace XmlRpc;
                switch (xml_value.getType())
                {
                    case XmlRpcValue::TypeDouble:
                        return static_cast<int>(static_cast<double>(xml_value));
                    case XmlRpcValue::TypeInt:
                        return static_cast<int>(xml_value);
                    case XmlRpcValue::TypeBoolean:
                        return static_cast<int>(static_cast<bool>(xml_value));
                    default:
                        return 0;
                };
            }

            template <>
            bool xml_cast(const XmlRpc::XmlRpcValue& xml_value)
            {
                using namespace XmlRpc;
                switch (xml_value.getType())
                {
                    case XmlRpcValue::TypeDouble:
                        return static_cast<bool>(static_cast<double>(xml_value));
                    case XmlRpcValue::TypeInt:
                        return static_cast<bool>(static_cast<int>(xml_value));
                    case XmlRpcValue::TypeBoolean:
                        return static_cast<bool>(xml_value);
                    default:
                        return false;
                };
            }

            template <class T>
            bool getImpl(const NodeHandle& nh, const std::string& key,
                         std::map<std::string, T>& map, bool cached = false)
            {
                XmlRpc::XmlRpcValue xml_value;
                if (cached)
                {
                    if (!nh.getParamCached(key, xml_value))
                    {
                        return false;
                    }
                }
                else
                {
                    if (!nh.getParam(key, xml_value))
                    {
                        return false;
                    }
                }

                // Make sure it's a struct type
                if (xml_value.getType() != XmlRpc::XmlRpcValue::TypeStruct)
                {
                    return false;
                }

                // Fill the map with stuff
                for (const auto& value : xml_value)
                {
                    // Only get the stuff of the right type
                    if (xml_castable<T>(value.second.getType()))
                    {
                        // Store the element
                        map[value.first] = xml_cast<T>(value.second);
                    }
                }

                return true;
            }

            template <class T>
            bool getImpl(const NodeHandle& nh, const std::string& key,
                         std::vector<T>& vec, bool cached = false)
            {
                XmlRpc::XmlRpcValue xml_array;
                if (cached)
                {
                    if (!nh.getParamCached(key, xml_array))
                    {
                        return false;
                    }
                }
                else
                {
                    if (!nh.getParam(key, xml_array))
                    {
                        return false;
                    }
                }

                // Make sure it's an array type
                if (xml_array.getType() != XmlRpc::XmlRpcValue::TypeArray)
                {
                    return false;
                }

                // Resize the target vector (destructive)
                std::size_t count =
                    std::accumulate(std::begin(xml_array), std::end(xml_array), 0,
                                    [](const auto& a, const auto& b) {
                                        return a
                                               + static_cast<std::size_t>(
                                                   xml_castable<T>(b.second.getType()));
                                    });
                vec.reserve(count);

                // Fill the vector with stuff
                for (const auto& value : xml_array)
                {
                    // Only get the stuff of the right type
                    if (xml_castable<T>(value.second.getType()))
                    {
                        // Store the element
                        vec.push_back(xml_cast<T>(value.second));
                    }
                }

                return true;
            }
        }

        bool getParam(const NodeHandle& nh, const std::string& key,
                      std::map<std::string, double>& map)
        {
            return impl::getImpl<double>(nh, key, map, false);
        }
        bool getParam(const NodeHandle& nh, const std::string& key,
                      std::map<std::string, float>& map)
        {
            return impl::getImpl<float>(nh, key, map, false);
        }
        bool getParam(const NodeHandle& nh, const std::string& key,
                      std::map<std::string, int>& map)
        {
            return impl::getImpl<int>(nh, key, map, false);
        }
        bool getParam(const NodeHandle& nh, const std::string& key,
                      std::map<std::string, bool>& map)
        {
            return impl::getImpl<bool>(nh, key, map, false);
        }
        bool getParam(const NodeHandle& nh, const std::string& key,
                      std::map<std::string, std::string>& map)
        {
            return impl::getImpl<std::string>(nh, key, map, false);
        }

        bool getParamCached(const NodeHandle& nh, const std::string& key,
                            std::map<std::string, double>& map)
        {
            return impl::getImpl<double>(nh, key, map, true);
        }
        bool getParamCached(const NodeHandle& nh, const std::string& key,
                            std::map<std::string, float>& map)
        {
            return impl::getImpl<float>(nh, key, map, true);
        }
        bool getParamCached(const NodeHandle& nh, const std::string& key,
                            std::map<std::string, int>& map)
        {
            return impl::getImpl<int>(nh, key, map, true);
        }
        bool getParamCached(const NodeHandle& nh, const std::string& key,
                            std::map<std::string, bool>& map)
        {
            return impl::getImpl<bool>(nh, key, map, true);
        }
        bool getParamCached(const NodeHandle& nh, const std::string& key,
                            std::map<std::string, std::string>& map)
        {
            return impl::getImpl<std::string>(nh, key, map, true);
        }

        bool getParam(const NodeHandle& nh, const std::string& key,
                      std::vector<double>& vec)
        {
            return impl::getImpl<double>(nh, key, vec, false);
        }
        bool getParam(const NodeHandle& nh, const std::string& key,
                      std::vector<float>& vec)
        {
            return impl::getImpl<float>(nh, key, vec, false);
        }
        bool getParam(const NodeHandle& nh, const std::string& key, std::vector<int>& vec)
        {
            return impl::getImpl<int>(nh, key, vec, false);
        }
        bool getParam(const NodeHandle& nh, const std::string& key,
                      std::vector<bool>& vec)
        {
            return impl::getImpl<bool>(nh, key, vec, false);
        }
        bool getParam(const NodeHandle& nh, const std::string& key,
                      std::vector<std::string>& vec)
        {
            return impl::getImpl<std::string>(nh, key, vec, false);
        }

        bool getParamCached(const NodeHandle& nh, const std::string& key,
                            std::vector<double>& vec)
        {
            return impl::getImpl<double>(nh, key, vec, true);
        }
        bool getParamCached(const NodeHandle& nh, const std::string& key,
                            std::vector<float>& vec)
        {
            return impl::getImpl<float>(nh, key, vec, true);
        }
        bool getParamCached(const NodeHandle& nh, const std::string& key,
                            std::vector<int>& vec)
        {
            return impl::getImpl<int>(nh, key, vec, true);
        }
        bool getParamCached(const NodeHandle& nh, const std::string& key,
                            std::vector<bool>& vec)
        {
            return impl::getImpl<bool>(nh, key, vec, true);
        }
        bool getParamCached(const NodeHandle& nh, const std::string& key,
                            std::vector<std::string>& vec)
        {
            return impl::getImpl<std::string>(nh, key, vec, true);
        }
    }
}

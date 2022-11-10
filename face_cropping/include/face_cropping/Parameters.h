#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <ros/ros.h>
#include <ros/package.h>

namespace face_cropping
{
    class Parameters
    {
        double m_refreshRate;
        int m_width;
        int m_height;

        int m_framesUsedForStabilizer;

        double m_minWidthChange;
        double m_minHeightChange;
        double m_minXChange;
        double m_minYChange;

        double m_rightMargin;
        double m_leftMargin;
        double m_topMargin;
        double m_bottomMargin;

        bool m_isPeerImage;

        std::string m_dnnModelPath;
        std::string m_dnnDeployPath;

        bool m_useGpu;

    public:
        explicit Parameters(ros::NodeHandle& nodeHandle);
        virtual ~Parameters();

        double refreshRate() const;
        int width() const;
        int height() const;

        int framesUsedForStabilizer() const;

        double minWidthChange() const;
        double minHeightChange() const;
        double minXChange() const;
        double minYChange() const;

        double rightMargin() const;
        double leftMargin() const;
        double topMargin() const;
        double bottomMargin() const;

        bool isPeerImage() const;

        std::string dnnModelPath() const;
        std::string dnnDeployPath() const;

        bool useGpu() const;
    };

    inline double Parameters::refreshRate() const
    {
        return m_refreshRate;
    }

    inline int Parameters::width() const
    {
        return m_width;
    }

    inline int Parameters::height() const
    {
        return m_height;
    }

    inline int Parameters::framesUsedForStabilizer() const
    {
        return m_framesUsedForStabilizer;
    }

    inline double Parameters::minWidthChange() const
    {
        return m_minWidthChange;
    }

    inline double Parameters::minHeightChange() const
    {
        return m_minHeightChange;
    }

    inline double Parameters::minXChange() const
    {
        return m_minXChange;
    }

    inline double Parameters::minYChange() const
    {
        return m_minYChange;
    }

    inline double Parameters::rightMargin() const
    {
        return m_rightMargin;
    }

    inline double Parameters::leftMargin() const
    {
        return m_leftMargin;
    }

    inline double Parameters::topMargin() const
    {
        return m_topMargin;
    }

    inline double Parameters::bottomMargin() const
    {
        return m_bottomMargin;
    }

    inline bool Parameters::isPeerImage() const
    {
        return m_isPeerImage;
    }

    inline std::string Parameters::dnnModelPath() const
    {
        return m_dnnModelPath;
    }

    inline std::string Parameters::dnnDeployPath() const
    {
        return m_dnnDeployPath;
    }

    inline bool Parameters::useGpu() const
    {
        return m_useGpu;
    }
}

#endif

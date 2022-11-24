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

        std::string m_haarCascadePath;
        std::string m_lbpCascadePath;

        bool m_useLbp;

        int m_detectionFrames;
        double m_detectionScale;
        int m_minFaceWidth;
        int m_minFaceHeight;

        double m_maxSizeStep;
        double m_maxPositionStep;

        int m_faceStoringFrames;
        double m_validFaceMinTime;

    public:
        explicit Parameters(ros::NodeHandle& nodeHandle);

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

        std::string haarCascadePath() const;
        std::string lbpCascadePath() const;

        bool useLbp() const;

        int detectionFrames() const;
        double detectionScale() const;
        int minFaceWidth() const;
        int minFaceHeight() const;

        double maxSizeStep() const;
        double maxPositionStep() const;

        int faceStoringFrames() const;
        double validFaceMinTime() const;
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

    inline std::string Parameters::haarCascadePath() const
    {
        return m_haarCascadePath;
    }

    inline std::string Parameters::lbpCascadePath() const
    {
        return m_lbpCascadePath;
    }

    inline bool Parameters::useLbp() const
    {
        return m_useLbp;
    }

    inline int Parameters::detectionFrames() const
    {
        return m_detectionFrames;
    }

    inline double Parameters::detectionScale() const
    {
        return m_detectionScale;
    }

    inline int Parameters::minFaceWidth() const
    {
        return m_minFaceWidth;
    }

    inline int Parameters::minFaceHeight() const
    {
        return m_minFaceHeight;
    }

    inline double Parameters::maxSizeStep() const
    {
        return m_maxSizeStep;
    }

    inline double Parameters::maxPositionStep() const
    {
        return m_maxPositionStep;
    }

    inline int Parameters::faceStoringFrames() const
    {
        return m_faceStoringFrames;
    }

    inline double Parameters::validFaceMinTime() const
    {
        return m_validFaceMinTime;
    }
}

#endif

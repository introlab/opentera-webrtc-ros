#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <string>

namespace map_image_generator
{
    class Parameters
    {
        double m_refreshRate;
        int m_resolution;          // pixel/m
        int m_width;               // m
        int m_height;              // m
        int m_xOrigin;             // pixel
        int m_yOrigin;             // pixel
        int m_robotVerticalOffset; // pixel

        double m_scaleFactor;

        std::string m_robotFrameId;
        std::string m_mapFrameId;
        bool m_centeredRobot;

        bool m_drawOccupancyGrid;
        bool m_drawGlobalPath;
        bool m_drawRobot;
        bool m_drawGoals;
        bool m_drawLaserScan;
        bool m_drawLabels;

        cv::Vec3b m_wallColor;
        cv::Vec3b m_freeSpaceColor;
        cv::Vec3b m_unknownSpaceColor;
        cv::Scalar m_globalPathColor;
        cv::Scalar m_robotColor;
        cv::Scalar m_goalColor;
        cv::Scalar m_laserScanColor;
        cv::Scalar m_textColor;
        cv::Scalar m_labelColor;

        int m_globalPathThickness; // pixel
        int m_robotSize;           // pixel
        int m_goalSize;            // pixel
        int m_laserScanSize;       // pixel
        int m_labelSize;           // pixel

    public:
        explicit Parameters(ros::NodeHandle& nodeHandle);
        virtual ~Parameters();

        double refreshRate() const;
        int resolution() const;          // pixel/m
        int width() const;               // m
        int height() const;              // m
        int xOrigin() const;             // pixel
        int yOrigin() const;             // pixel
        int robotVerticalOffset() const; // pixel

        double scaleFactor() const;
        void setScaleFactor(double scaleFactor);

        const std::string& robotFrameId() const;
        const std::string& mapFrameId() const;
        const std::string& refFrameId() const;
        bool centeredRobot() const;
        void setCenteredRobot(bool centeredRobot);

        bool drawOccupancyGrid() const;
        bool drawGlobalPath() const;
        bool drawRobot() const;
        bool drawGoals() const;
        bool drawLaserScan() const;
        bool drawLabels() const;

        const cv::Vec3b& wallColor() const;
        const cv::Vec3b& freeSpaceColor() const;
        const cv::Vec3b& unknownSpaceColor() const;
        const cv::Scalar& globalPathColor() const;
        const cv::Scalar& robotColor() const;
        const cv::Scalar& goalColor() const;
        const cv::Scalar& laserScanColor() const;
        const cv::Scalar& textColor() const;
        const cv::Scalar& labelColor() const;

        int globalPathThickness() const; // pixel
        int robotSize() const;           // pixel
        int goalSize() const;            // pixel
        int laserScanSize() const;       // pixel
        int labelSize() const;           // pixel

    private:
        void validateParameters();
    };

    inline double Parameters::refreshRate() const { return m_refreshRate; }

    inline int Parameters::resolution() const { return m_resolution; }

    inline int Parameters::width() const { return m_width; }

    inline int Parameters::height() const { return m_height; }

    inline int Parameters::xOrigin() const { return m_xOrigin; }

    inline int Parameters::yOrigin() const { return m_yOrigin; }

    inline int Parameters::robotVerticalOffset() const
    {
        if (m_centeredRobot)
        {
            return m_robotVerticalOffset;
        }
        else
        {
            return 0;
        }
    }

    inline double Parameters::scaleFactor() const { return m_scaleFactor; }

    inline void Parameters::setScaleFactor(double scaleFactor)
    {
        m_scaleFactor = scaleFactor;
    }

    inline const std::string& Parameters::robotFrameId() const { return m_robotFrameId; }

    inline const std::string& Parameters::mapFrameId() const { return m_mapFrameId; }

    inline bool Parameters::centeredRobot() const { return m_centeredRobot; }

    inline bool Parameters::drawOccupancyGrid() const { return m_drawOccupancyGrid; }

    inline bool Parameters::drawGlobalPath() const { return m_drawGlobalPath; }

    inline bool Parameters::drawRobot() const { return m_drawRobot; }

    inline bool Parameters::drawGoals() const { return m_drawGoals; }

    inline bool Parameters::drawLaserScan() const { return m_drawLaserScan; }

    inline bool Parameters::drawLabels() const { return m_drawLabels; }

    inline const cv::Vec3b& Parameters::wallColor() const { return m_wallColor; }

    inline const cv::Vec3b& Parameters::freeSpaceColor() const
    {
        return m_freeSpaceColor;
    }

    inline const cv::Vec3b& Parameters::unknownSpaceColor() const
    {
        return m_unknownSpaceColor;
    }

    inline const cv::Scalar& Parameters::globalPathColor() const
    {
        return m_globalPathColor;
    }

    inline const cv::Scalar& Parameters::robotColor() const { return m_robotColor; }

    inline const cv::Scalar& Parameters::goalColor() const { return m_goalColor; }

    inline const cv::Scalar& Parameters::laserScanColor() const
    {
        return m_laserScanColor;
    }

    inline const cv::Scalar& Parameters::textColor() const { return m_textColor; }

    inline const cv::Scalar& Parameters::labelColor() const { return m_labelColor; }

    inline int Parameters::globalPathThickness() const { return m_globalPathThickness; }

    inline int Parameters::robotSize() const { return m_robotSize; }

    inline int Parameters::goalSize() const { return m_goalSize; }

    inline int Parameters::laserScanSize() const { return m_laserScanSize; }

    inline int Parameters::labelSize() const { return m_labelSize; }
}
#endif

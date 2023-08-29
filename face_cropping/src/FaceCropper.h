#ifndef FACE_CROPPER_H
#define FACE_CROPPER_H

#include "FaceDetector.h"
#include "SinglePositionKalmanFilter.h"

#include <opencv4/opencv2/core.hpp>

#include <memory>
#include <optional>

struct FaceCropperParameters
{
    // Face detector dependant parameters
    int detectionInterval;
    float xOffsetRatio;
    float yOffsetRatio;
    float widthScale;
    float heightScale;
    int frameBeforeChangeTarget;

    float initialPositionVariance;
    float initialVelocityVariance;
    float qPosition;
    float qVelocity;
    float rPosition;

    // General parameters
    float minFaceWidth;
    float minFaceHeight;
    int outputWidth;
    int outputHeight;

    static FaceCropperParameters fromFaceDetector(
        const FaceDetector& faceDetector,
        float minFaceWidth,
        float minFaceHeight,
        int outputWidth,
        int outputHeight);
};

struct FaceCrop
{
    float xCenter;
    float yCenter;
    float width;
    float height;

    FaceCrop(float xCenter, float yCenter, float width, float height);
    [[nodiscard]] FaceCrop scale(float scale) const;
    [[nodiscard]] bool isInside(float x, float y) const;
};

class FaceCropKalmanFilter
{
    SinglePositionKalmanFilter m_xCenter;
    SinglePositionKalmanFilter m_yCenter;
    SinglePositionKalmanFilter m_width;
    SinglePositionKalmanFilter m_height;

public:
    FaceCropKalmanFilter(const FaceCrop& crop, const FaceCropperParameters& parameters);

    [[nodiscard]] FaceCrop crop() const;
    [[nodiscard]] float xCenter() const;
    [[nodiscard]] float yCenter() const;
    [[nodiscard]] float width() const;
    [[nodiscard]] float height() const;


    void update(const FaceCrop& crop, const FaceCropperParameters& parameters, float dt);
};

class FaceCropper
{
    std::shared_ptr<FaceDetector> m_faceDetector;
    FaceCropperParameters m_parameters;

    int m_frameCounter;
    std::optional<FaceCropKalmanFilter> m_cropFilter;
    int m_cropFrameCounter;

public:
    FaceCropper(
        std::shared_ptr<FaceDetector> faceDetector,
        float minFaceWidth,
        float minFaceHeight,
        int outputWidth,
        int outputHeight);

    void reset();
    void crop(const cv::Mat& bgrInputImage, cv::Mat& bgrOutputImage);

private:
    void updateCrop(const cv::Mat& bgrInputImage);
    void performCrop(const cv::Mat& bgrInputImage, cv::Mat& bgrOutputImage);
};

#endif

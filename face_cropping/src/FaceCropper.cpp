#include "FaceCropper.h"

#include "OpencvFaceDetector.h"
#include "TorchFaceDetector.h"

#include <opencv4/opencv2/imgproc.hpp>

#include <algorithm>

FaceCropperParameters FaceCropperParameters::fromFaceDetector(
    const FaceDetector& faceDetector,
    float minFaceWidth,
    float minFaceHeight,
    int outputWidth,
    int outputHeight)
{
    FaceCropperParameters parameters = {};

    if (faceDetector.type() == typeid(HaarFaceDetector))
    {
        parameters.detectionInterval = 2;
        parameters.xOffsetRatio = 0.f;
        parameters.yOffsetRatio = -0.05f;
        parameters.widthScale = 1.5f;
        parameters.heightScale = 1.75f;
        parameters.frameBeforeChangeTarget = 10;

        parameters.rPosition = 150.f;
    }
    else if (faceDetector.type() == typeid(LbpFaceDetector))
    {
        parameters.detectionInterval = 2;
        parameters.xOffsetRatio = 0.f;
        parameters.yOffsetRatio = -0.1f;
        parameters.widthScale = 2.f;
        parameters.heightScale = 2.5f;
        parameters.frameBeforeChangeTarget = 10;

        parameters.rPosition = 125.f;
    }
#ifndef NO_TORCH
    else if (
        faceDetector.type() == typeid(SmallYunet025Silu160FaceDetector) ||
        faceDetector.type() == typeid(SmallYunet05Silu160FaceDetector))
    {
        parameters.detectionInterval = 2;
        parameters.xOffsetRatio = 0.f;
        parameters.yOffsetRatio = 0.f;
        parameters.widthScale = 1.25f;
        parameters.heightScale = 1.5f;
        parameters.frameBeforeChangeTarget = 10;

        parameters.rPosition = 100.0f;
    }
    else if (
        faceDetector.type() == typeid(SmallYunet025Silu320FaceDetector) ||
        faceDetector.type() == typeid(SmallYunet05Silu320FaceDetector))
    {
        parameters.detectionInterval = 2;
        parameters.xOffsetRatio = 0.f;
        parameters.yOffsetRatio = 0.f;
        parameters.widthScale = 1.5f;
        parameters.heightScale = 1.75f;
        parameters.frameBeforeChangeTarget = 10;

        parameters.rPosition = 80.0f;
    }
    else if (
        faceDetector.type() == typeid(SmallYunet025Silu640FaceDetector) ||
        faceDetector.type() == typeid(SmallYunet05Silu640FaceDetector))
    {
        parameters.detectionInterval = 2;
        parameters.xOffsetRatio = 0.f;
        parameters.yOffsetRatio = 0.f;
        parameters.widthScale = 1.5f;
        parameters.heightScale = 1.75f;
        parameters.frameBeforeChangeTarget = 10;

        parameters.rPosition = 60.0f;
    }
#endif
    else
    {
        throw std::runtime_error("Not supported face detector");
    }

    parameters.initialPositionVariance = 1.f;
    parameters.initialVelocityVariance = 5.f;
    parameters.qPosition = 0.0005f;
    parameters.qVelocity = 0.2f;

    parameters.minFaceWidth = minFaceWidth;
    parameters.minFaceHeight = minFaceHeight;
    parameters.outputWidth = outputWidth;
    parameters.outputHeight = outputHeight;

    return parameters;
}


FaceCrop::FaceCrop(float xCenter, float yCenter, float width, float height)
    : xCenter(xCenter),
      yCenter(yCenter),
      width(width),
      height(height)
{
}

FaceCrop FaceCrop::scale(float scale) const
{
    return {xCenter, yCenter, width * scale, height * scale};
}

bool FaceCrop::isInside(float x, float y) const
{
    float xMin = xCenter - width / 2.f;
    float xMax = xCenter + width / 2.f;
    float yMin = yCenter - height / 2.f;
    float yMax = yCenter + height / 2.f;

    return xMin <= x && x <= xMax && yMin <= y && y <= yMax;
}

FaceCropKalmanFilter::FaceCropKalmanFilter(const FaceCrop& crop, const FaceCropperParameters& parameters)
    : m_xCenter(
          crop.xCenter,
          0,
          parameters.initialPositionVariance,
          parameters.initialVelocityVariance,
          parameters.qPosition,
          parameters.qVelocity),
      m_yCenter(
          crop.yCenter,
          0,
          parameters.initialPositionVariance,
          parameters.initialVelocityVariance,
          parameters.qPosition,
          parameters.qVelocity),
      m_width(
          crop.width,
          0,
          parameters.initialPositionVariance,
          parameters.initialVelocityVariance,
          parameters.qPosition,
          parameters.qVelocity),
      m_height(
          crop.height,
          0,
          parameters.initialPositionVariance,
          parameters.initialVelocityVariance,
          parameters.qPosition,
          parameters.qVelocity)
{
}

FaceCrop FaceCropKalmanFilter::crop() const
{
    return {m_xCenter.position(), m_yCenter.position(), m_width.position(), m_height.position()};
}

float FaceCropKalmanFilter::xCenter() const
{
    return m_xCenter.position();
}

float FaceCropKalmanFilter::yCenter() const
{
    return m_yCenter.position();
}

float FaceCropKalmanFilter::width() const
{
    return m_width.position();
}

float FaceCropKalmanFilter::height() const
{
    return m_height.position();
}

void FaceCropKalmanFilter::update(const FaceCrop& crop, const FaceCropperParameters& parameters, float dt)
{
    m_xCenter.update(crop.xCenter, parameters.rPosition, dt);
    m_yCenter.update(crop.yCenter, parameters.rPosition, dt);
    m_width.update(crop.width, parameters.rPosition, dt);
    m_height.update(crop.height, parameters.rPosition, dt);
}


FaceCropper::FaceCropper(
    std::shared_ptr<FaceDetector> faceDetector,
    float minFaceWidth,
    float minFaceHeight,
    int outputWidth,
    int outputHeight)
    : m_faceDetector(move(faceDetector)),
      m_frameCounter(0),
      m_cropFrameCounter(0)
{
    m_parameters = FaceCropperParameters::fromFaceDetector(
        *m_faceDetector,
        minFaceWidth,
        minFaceHeight,
        outputWidth,
        outputHeight);
}

void FaceCropper::reset()
{
    m_frameCounter = 0;
    m_cropFrameCounter = 0;
    m_cropFilter = std::nullopt;
}

void FaceCropper::crop(const cv::Mat& bgrInputImage, cv::Mat& bgrOutputImage)
{
    if ((m_frameCounter % m_parameters.detectionInterval) == 0)
    {
        updateCrop(bgrInputImage);
    }

    if (m_cropFilter.has_value())
    {
        performCrop(bgrInputImage, bgrOutputImage);
    }
    else
    {
        bgrOutputImage = bgrInputImage;
    }
}

void FaceCropper::updateCrop(const cv::Mat& bgrInputImage)
{
    auto faces = m_faceDetector->detect(bgrInputImage);

    auto it = std::max_element(
        faces.begin(),
        faces.end(),
        [](const FaceDetection& a, const FaceDetection& b) { return (a.width * a.height) < (b.width * b.height); });

    std::optional<FaceCrop> newCrop;
    if (it != faces.end())
    {
        float newWidth = it->width * m_parameters.widthScale;
        float newHeight = it->height * m_parameters.heightScale;
        if (newWidth >= m_parameters.minFaceWidth && newHeight >= m_parameters.minFaceHeight)
        {
            newCrop = FaceCrop(it->xCenter, it->yCenter, newWidth, newHeight);
        }
    }

    m_cropFrameCounter++;
    if (newCrop.has_value())
    {
        if (m_cropFilter == std::nullopt)
        {
            m_cropFilter = FaceCropKalmanFilter(*newCrop, m_parameters);
            m_cropFrameCounter = 0;
        }
        else if (m_cropFilter->crop()
                     .scale(float(m_parameters.detectionInterval))
                     .isInside(newCrop->xCenter, newCrop->yCenter))
        {
            m_cropFilter->update(*newCrop, m_parameters, float(m_cropFrameCounter));
            m_cropFrameCounter = 0;
        }
    }

    if (m_cropFrameCounter >= m_parameters.frameBeforeChangeTarget)
    {
        m_cropFilter = std::nullopt;
        m_cropFrameCounter = 0;
    }
}

void FaceCropper::performCrop(const cv::Mat& bgrInputImage, cv::Mat& bgrOutputImage)
{
    float scale = std::min(
        float(m_cropFilter->width()) / float(m_parameters.outputWidth),
        float(m_cropFilter->height()) / float(m_parameters.outputHeight));

    float width = scale * float(m_parameters.outputWidth);
    float height = scale * float(m_parameters.outputHeight);

    float xCenter = m_cropFilter->xCenter() + width * m_parameters.xOffsetRatio;
    float yCenter = m_cropFilter->yCenter() + height * m_parameters.yOffsetRatio;

    int x0 = std::max(int(std::round(xCenter - width / 2.f)), 0);
    int y0 = std::max(int(std::round(yCenter - height / 2.f)), 0);
    int x1 = std::min(int(std::round(xCenter + width / 2.f)), bgrInputImage.cols);
    int y1 = std::min(int(std::round(yCenter + height / 2.f)), bgrInputImage.rows);

    cv::resize(
        bgrInputImage(cv::Rect(x0, y0, x1 - x0, y1 - y0)),
        bgrOutputImage,
        cv::Size(m_parameters.outputWidth, m_parameters.outputHeight));
}

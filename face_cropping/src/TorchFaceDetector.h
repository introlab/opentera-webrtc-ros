#ifndef TORCH_FACE_DETECTOR_H
#define TORCH_FACE_DETECTOR_H

#ifndef NO_TORCH

#include "FaceDetector.h"

#include <torch/torch.h>
#include <rclcpp/rclcpp.hpp>

class TorchFaceDetector : public FaceDetector
{
    int m_maxWidth;
    int m_maxHeight;
    float m_confidenceThreshold;
    float m_nmsThreshold;

    torch::DeviceType m_device;
    torch::ScalarType m_scalarType;
    torch::jit::script::Module m_detector;

    torch::Tensor m_normalizationMean;
    torch::Tensor m_normalizationStdInv;

    cv::Mat m_resizedImage;
    cv::Mat m_inputImage;
    torch::Tensor m_inputTensor;

    constexpr static float ONE_OVER_255 = 1.0f / 255.0f;

protected:
    TorchFaceDetector(
        rclcpp::Node& nodeHandle,
        bool useGpuIfAvailable,
        int maxWidth,
        int maxHeight,
        float confidenceThreshold,
        float nmsThreshold,
        const std::string& modelPath);

public:
    ~TorchFaceDetector() override = default;

    std::vector<FaceDetection> detect(const cv::Mat& bgrImage) override;

private:
    torch::Tensor filterFaces(const torch::Tensor& faces);
};


class SmallYunet025Silu160FaceDetector : public TorchFaceDetector
{
    constexpr static const char* MODEL_SUBPATH = "/models/small_yunet_0.25_silu_160.pt";

public:
    explicit SmallYunet025Silu160FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable);

    [[nodiscard]] std::type_index type() const override { return typeid(SmallYunet025Silu160FaceDetector); }
};

class SmallYunet025Silu320FaceDetector : public TorchFaceDetector
{
    constexpr static const char* MODEL_SUBPATH = "/models/small_yunet_0.25_silu_320.pt";

public:
    explicit SmallYunet025Silu320FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable);

    [[nodiscard]] std::type_index type() const override { return typeid(SmallYunet025Silu320FaceDetector); }
};

class SmallYunet025Silu640FaceDetector : public TorchFaceDetector
{
    constexpr static const char* MODEL_SUBPATH = "/models/small_yunet_0.25_silu_640.pt";

public:
    explicit SmallYunet025Silu640FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable);

    [[nodiscard]] std::type_index type() const override { return typeid(SmallYunet025Silu640FaceDetector); }
};


class SmallYunet05Silu160FaceDetector : public TorchFaceDetector
{
    constexpr static const char* MODEL_SUBPATH = "/models/small_yunet_0.5_silu_160.pt";

public:
    explicit SmallYunet05Silu160FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable);

    [[nodiscard]] std::type_index type() const override { return typeid(SmallYunet05Silu160FaceDetector); }
};

class SmallYunet05Silu320FaceDetector : public TorchFaceDetector
{
    constexpr static const char* MODEL_SUBPATH = "/models/small_yunet_0.5_silu_320.pt";

public:
    explicit SmallYunet05Silu320FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable);

    [[nodiscard]] std::type_index type() const override { return typeid(SmallYunet05Silu320FaceDetector); }
};

class SmallYunet05Silu640FaceDetector : public TorchFaceDetector
{
    constexpr static const char* MODEL_SUBPATH = "/models/small_yunet_0.5_silu_640.pt";

public:
    explicit SmallYunet05Silu640FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable);

    [[nodiscard]] std::type_index type() const override { return typeid(SmallYunet05Silu640FaceDetector); }
};

#endif  // NO_TORCH

#endif  // TORCH_FACE_DETECTOR_H

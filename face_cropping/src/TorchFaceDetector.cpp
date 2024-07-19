#include "TorchFaceDetector.h"

#include <opencv4/opencv2/imgproc.hpp>

#include <rclcpp/rclcpp.hpp>

#ifndef NO_TORCH

#include <omp.h>
#include <torch/script.h>

#ifdef TORCHVISION_CSRC_INCLUDE
#include <torchvision/csrc/ops/nms.h>
#else
#include <torchvision/ops/nms.h>
#endif

constexpr int CONFIDENCE_INDEX = 0;
constexpr int TL_X_INDEX = 1;
constexpr int TL_Y_INDEX = 2;
constexpr int BR_X_INDEX = 3;
constexpr int BR_Y_INDEX = 4;


TorchFaceDetector::TorchFaceDetector(
    rclcpp::Node& nodeHandle,
    bool useGpuIfAvailable,
    int maxWidth,
    int maxHeight,
    float confidenceThreshold,
    float nmsThreshold,
    const std::string& modelPath)
    : m_maxWidth(maxWidth),
      m_maxHeight(maxHeight),
      m_confidenceThreshold(confidenceThreshold),
      m_nmsThreshold(nmsThreshold),
      m_scalarType(torch::kFloat),
      m_inputImage(m_maxHeight, m_maxWidth, CV_8UC3)
{
    std::string suffix;
    if (useGpuIfAvailable && torch::cuda::is_available())
    {
        m_device = torch::kCUDA;
        suffix = ".cuda";
    }
    else
    {
        if (useGpuIfAvailable)
        {
            RCLCPP_WARN(nodeHandle.get_logger(), "CUDA is not supported.");
        }
        m_device = torch::kCPU;
    }

    omp_set_num_threads(1);
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);

    m_detector = torch::jit::load(modelPath + suffix);
    m_detector.to(m_device, m_scalarType);
    m_detector.eval();

    m_normalizationMean = torch::tensor({0.485f, 0.456f, 0.406f}).to(m_device, m_scalarType);
    m_normalizationStdInv = torch::tensor({1.f / 0.229f, 1.f / 0.224f, 1.f / 0.225f}).to(m_device, m_scalarType);
}

std::vector<FaceDetection> TorchFaceDetector::detect(const cv::Mat& bgrImage)
{
    torch::InferenceMode inferenceModeGuard;

    float scale = std::min(
        static_cast<float>(m_maxWidth) / static_cast<float>(bgrImage.cols),
        static_cast<float>(m_maxHeight) / static_cast<float>(bgrImage.rows));
    cv::resize(bgrImage, m_resizedImage, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::cvtColor(m_resizedImage, m_resizedImage, cv::COLOR_BGR2RGB);

    m_inputImage.setTo(cv::Scalar(114, 114, 114));
    m_resizedImage.copyTo(m_inputImage(cv::Rect(0, 0, m_resizedImage.cols, m_resizedImage.rows)));

    m_inputTensor = torch::from_blob(
                        m_inputImage.data,
                        {m_inputImage.rows, m_inputImage.cols, m_inputImage.channels()},
                        torch::kByte)
                        .permute({2, 0, 1})
                        .to(m_device, m_scalarType);
    m_inputTensor.mul_(ONE_OVER_255);
    m_inputTensor[0].sub_(m_normalizationMean[0]).mul_(m_normalizationStdInv[0]);
    m_inputTensor[1].sub_(m_normalizationMean[1]).mul_(m_normalizationStdInv[1]);
    m_inputTensor[2].sub_(m_normalizationMean[2]).mul_(m_normalizationStdInv[2]);
    m_inputTensor.unsqueeze_(0);

    std::vector<torch::jit::IValue> inputs{m_inputTensor};
    at::Tensor faces = m_detector.forward(inputs).toTensor()[0];
    faces = filterFaces(faces).to(torch::kCPU);

    std::vector<FaceDetection> faceDetections;
    faceDetections.reserve(faces.size(0));

    for (int i = 0; i < faces.size(0); i++)
    {
        float tlX = faces.index({i, TL_X_INDEX}).item<float>();
        float tlY = faces.index({i, TL_Y_INDEX}).item<float>();
        float brX = faces.index({i, BR_X_INDEX}).item<float>();
        float brY = faces.index({i, BR_Y_INDEX}).item<float>();
        float w = brX - tlX;
        float h = brY - tlY;
        faceDetections.emplace_back((tlX + w / 2) / scale, (tlY + h / 2) / scale, w / scale, h / scale);
    }

    return faceDetections;
}

torch::Tensor TorchFaceDetector::filterFaces(const torch::Tensor& faces)
{
    auto bboxesMask = faces.index({torch::indexing::Slice(), CONFIDENCE_INDEX}) > m_confidenceThreshold;
    auto confidentFaces = faces.index({bboxesMask, torch::indexing::Slice()});

    auto indexes = vision::ops::nms(
        confidentFaces.index({torch::indexing::Slice(), torch::indexing::Slice(TL_X_INDEX)}),
        confidentFaces.index({torch::indexing::Slice(), CONFIDENCE_INDEX}),
        m_nmsThreshold);

    return confidentFaces.index({indexes, torch::indexing::Slice()});
}


SmallYunet025Silu160FaceDetector::SmallYunet025Silu160FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable)
    : TorchFaceDetector(nodeHandle, useGpuIfAvailable, 160, 160, 0.2, 0.3, getPackagePath() + MODEL_SUBPATH)
{
}

SmallYunet025Silu320FaceDetector::SmallYunet025Silu320FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable)
    : TorchFaceDetector(nodeHandle, useGpuIfAvailable, 320, 320, 0.3, 0.3, getPackagePath() + MODEL_SUBPATH)
{
}

SmallYunet025Silu640FaceDetector::SmallYunet025Silu640FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable)
    : TorchFaceDetector(nodeHandle, useGpuIfAvailable, 640, 640, 0.3, 0.3, getPackagePath() + MODEL_SUBPATH)
{
}


SmallYunet05Silu160FaceDetector::SmallYunet05Silu160FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable)
    : TorchFaceDetector(nodeHandle, useGpuIfAvailable, 160, 160, 0.3, 0.3, getPackagePath() + MODEL_SUBPATH)
{
}

SmallYunet05Silu320FaceDetector::SmallYunet05Silu320FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable)
    : TorchFaceDetector(nodeHandle, useGpuIfAvailable, 320, 320, 0.3, 0.3, getPackagePath() + MODEL_SUBPATH)
{
}

SmallYunet05Silu640FaceDetector::SmallYunet05Silu640FaceDetector(rclcpp::Node& nodeHandle, bool useGpuIfAvailable)
    : TorchFaceDetector(nodeHandle, useGpuIfAvailable, 640, 640, 0.3, 0.3, getPackagePath() + MODEL_SUBPATH)
{
}

#endif

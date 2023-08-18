#include "FaceDetector.h"
#include "OpencvFaceDetector.h"
#include "TorchFaceDetector.h"


std::shared_ptr<FaceDetector> createFaceDetector(const std::string& name, bool useGpuIfAvailable)
{
    if (name == "haarcascade")
    {
        return std::make_unique<HaarFaceDetector>();
    }
    else if (name == "lbpcascade")
    {
        return std::make_unique<LbpFaceDetector>();
    }
#ifndef NO_TORCH
    else if (name == "small_yunet_0.25_160")
    {
        return std::make_unique<SmallYunet025Silu160FaceDetector>(useGpuIfAvailable);
    }
    else if (name == "small_yunet_0.25_320")
    {
        return std::make_unique<SmallYunet025Silu320FaceDetector>(useGpuIfAvailable);
    }
    else if (name == "small_yunet_0.25_640")
    {
        return std::make_unique<SmallYunet025Silu640FaceDetector>(useGpuIfAvailable);
    }
    else if (name == "small_yunet_0.5_160")
    {
        return std::make_unique<SmallYunet05Silu160FaceDetector>(useGpuIfAvailable);
    }
    else if (name == "small_yunet_0.5_320")
    {
        return std::make_unique<SmallYunet05Silu320FaceDetector>(useGpuIfAvailable);
    }
    else if (name == "small_yunet_0.5_640")
    {
        return std::make_unique<SmallYunet05Silu640FaceDetector>(useGpuIfAvailable);
    }
#endif
    else
    {
        throw std::runtime_error("Not supported face detector (" + name + ")");
    }
}

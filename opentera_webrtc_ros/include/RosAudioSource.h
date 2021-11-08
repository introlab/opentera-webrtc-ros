#ifndef OPENTERA_WEBRTC_NATIVE_CLIENT_SOURCES_ROS_AUDIO_SOURCE_H
#define OPENTERA_WEBRTC_NATIVE_CLIENT_SOURCES_ROS_AUDIO_SOURCE_H

#include <OpenteraWebrtcNativeClient/Sources/AudioSource.h>
#include <audio_utils/AudioFrame.h>

namespace opentera
{

    /**
     * @brief A webrtc audio source that gets images from a ROS topic
     *
     * Usage: pass an shared_ptr to an instance of this to the xxxx constructor.
     * Use the audioCallback as a ROS topic subscriber callback.
     */
    class RosAudioSource : public AudioSource
    {

    public:
        RosAudioSource(unsigned int soundCardTotalDelayMs = 40,
            bool echoCancellation = true,
            bool autoGainControl = true,
            bool noiseSuppression = true,
            bool highPassFilter = false,
            bool stereoSwapping = false,
            bool typingDetection = false,
            bool residualEchoDetector = true,
            bool transientSuppression = true);

        void audioCallback(const audio_utils::AudioFrameConstPtr& msg);
    };
}

#endif

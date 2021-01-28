#include <RosAudioSource.h>


using namespace opentera;


RosAudioSource::RosAudioSource()
    //Creating a default configuration for now, int bitsPerSample, int sampleRate, size_t numberOfChannels);
    : AudioSource(AudioSourceConfiguration::create(), 16, 48000, 1) 
{ 

}


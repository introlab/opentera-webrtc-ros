#!/usr/bin/env python3
# PYTHONPATH is set properly when loading a workspace.

# This package needs to be installed first.
import pyaudio
import numpy
import threading
from datetime import datetime

# ROS
import rospy
from opentera_webrtc_ros_msgs.msg import PeerAudio

p = pyaudio.PyAudio()


class AudioWriter:
    def __init__(self, peer_id: str):
        self._peer_id = peer_id
        self._mutex = threading.Lock()
        self._semaphore = threading.Semaphore(value=0)
        self._audio_fifo = []
        self._running = False
        self._thread = threading.Thread(target=self._run)
        self._lastPushTime = datetime.now()

    def __del__(self):
        self.stop()

    def push_audio(self, audio: PeerAudio):
        self._mutex.acquire()
        # print('PUSH', datetime.now().timestamp(), len(self._audio_fifo))
        self._audio_fifo.append(audio)
        self._lastPushTime = datetime.now()
        self._mutex.release()
        self._semaphore.release()

    def pull_audio(self):
        self._semaphore.acquire()
        self._mutex.acquire()
        # print('PULL', datetime.now().timestamp(), len(self._audio_fifo))
        audio = self._audio_fifo.pop(0)
        self._mutex.release()
        return audio

    def _run(self):
        print('thread_run')
        stream = None
        while self._running:
            audio = self.pull_audio()

            if audio.frame.format == 'signed_16':

                if stream is None:
                    stream = p.open(format=pyaudio.paInt16,
                                    channels=audio.frame.channel_count,
                                    rate=audio.frame.sampling_frequency,
                                    output=True)

                # Write data (should be 10ms)
                # stream.write(numpy.frombuffer(audio.frame.data, numpy.int16))
                stream.write(audio.frame.data)
            else:
                print('unsupported format: ', audio.frame.format)

        if stream:
            stream.close()

        print('thread done!')

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join()


class AudioMixerROS:
    def __init__(self):
        self._subscriber = rospy.Subscriber('/webrtc_audio', PeerAudio, self._on_peer_audio)
        self._writers = dict()

    def _on_peer_audio(self, audio: PeerAudio):
        peer_id = audio.sender.id
        if peer_id not in self._writers:
            # Create new writer thread
            writer = AudioWriter(peer_id)
            self._writers[peer_id] = writer
            # Start thread
            writer.start()

        # Push audio
        self._writers[peer_id].push_audio(audio)

        # TODO cleanup old threads...


if __name__ == '__main__':
    # Init ROS
    rospy.init_node('opentera_webrtc_audio_mixer', anonymous=True)

    mixer = AudioMixerROS()

    rospy.spin()


#!/usr/bin/env python3
# PYTHONPATH is set properly when loading a workspace.

# This package needs to be installed first.
import pyaudio
import numpy
import threading
from datetime import datetime
import queue

# ROS
import rospy
from opentera_webrtc_ros_msgs.msg import PeerAudio

p = pyaudio.PyAudio()


class AudioWriter:
    def __init__(self, peer_id: str):
        self._peer_id = peer_id
        self._audio_queue = queue.Queue()
        self._running = False
        self._thread = threading.Thread(target=self._run)
        self._lastPushTime = datetime.now()

    def __del__(self):
        self.stop()

    def push_audio(self, audio: PeerAudio, timeout=None):
        # print('PUSH', datetime.now().timestamp())
        self._audio_queue.put(audio, timeout=timeout)
        self._lastPushTime = datetime.now()

    def pull_audio(self, timeout=None):
        # print('PULL', datetime.now().timestamp())
        audio = self._audio_queue.get(timeout=timeout)
        return audio

    def _run(self):
        print('thread_run')
        stream = None
        while self._running:
            try:
                # Write data (should get 10ms frames)
                audio = self.pull_audio(timeout=0.010)
                if audio:
                    if audio.frame.format == 'signed_16':
                        if stream is None:
                            stream = p.open(format=pyaudio.paInt16,
                                            channels=audio.frame.channel_count,
                                            rate=audio.frame.sampling_frequency,
                                            output=True)

                        stream.write(audio.frame.data)
                    else:
                        print('unsupported format: ', audio.frame.format)

            except queue.Empty as e:
                pass

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
        self._subscriber = rospy.Subscriber('/webrtc_audio', PeerAudio, self._on_peer_audio, queue_size=100)
        self._writers = dict()

    def __del__(self):
        for writer in self._writers:
            writer.stop()
            del writer

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


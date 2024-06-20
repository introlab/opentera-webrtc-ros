#!/usr/bin/env python3
# PYTHONPATH is set properly when loading a workspace.
# This package needs to be installed first.
import pyaudio
import numpy
import threading
from datetime import datetime, timedelta
import queue

# ROS
import rclpy
import rclpy.node
from opentera_webrtc_ros_msgs.msg import PeerAudio

p = pyaudio.PyAudio()

output_device_index = 0


class AudioWriter:
    def __init__(self, peer_id: str):
        self._peer_id = peer_id
        self._audio_queue = queue.Queue()
        self._quit_event = threading.Event()
        self._thread = threading.Thread(target=self._run)
        self._lastPushTime = datetime.now()

    def get_last_push(self):
        return self._lastPushTime

    def push_audio(self, audio: PeerAudio, timeout=None):
        self._audio_queue.put(audio, timeout=timeout)
        # print('PUSH', datetime.now().timestamp(), self._audio_queue.qsize())
        self._lastPushTime = datetime.now()

    def pull_audio(self, timeout=None):
        audio = self._audio_queue.get(timeout=timeout)
        # print('PULL', datetime.now().timestamp(), self._audio_queue.qsize())
        return audio

    def _run(self):
        print('Thread_run', self._peer_id)
        stream = None

        while not self._quit_event.isSet():
            try:
                # Write data (should get 10ms frames)
                audio = self.pull_audio(timeout=0.010)
                if audio:
                    if audio.frame.format == 'signed_16':
                        if stream is None:
                            stream = p.open(format=pyaudio.paInt16,
                                            channels=audio.frame.channel_count,
                                            rate=audio.frame.sampling_frequency,
                                            output_device_index=output_device_index,
                                            frames_per_buffer=int(
                                                audio.frame.frame_sample_count * 20),
                                            output=True)
                            # Fill buffer with zeros ?
                            # for _ in range(10):
                            #    stream.write(numpy.zeros(audio.frame.frame_sample_count, dtype=numpy.int16))

                        stream.write(audio.frame.data)
                    else:
                        print('Unsupported format: ',
                              audio.frame.format, self._peer_id)

            except queue.Empty as e:
                # An exception will occur when queue is empty
                pass

        if stream:
            stream.close()

        print('Thread done!', self._peer_id)

    def start(self):
        self._quit_event.clear()
        self._thread.start()

    def stop(self):
        if self._thread.is_alive():
            self._quit_event.set()
            print('Waiting for thread', self._peer_id)
            self._thread.join()


class AudioMixerROS(rclpy.node.Node):
    def __init__(self):
        super().__init__('opentera_webrtc_audio_mixer')

        self._subscriber = self.create_subscription(
            PeerAudio, '/webrtc_audio', self._on_peer_audio, 100)
        self._writers = dict()
        # Cleanup timer every second
        self._timer = self.create_timer(1, self._on_cleanup_timeout)

    def shutdown(self):
        self._timer.destroy()
        for writer in self._writers:
            print('stopping writer', writer)
            self._writers[writer].stop()

    def _on_cleanup_timeout(self, event):
        # Cleanup old threads ...
        peers_to_delete = []
        # Store since we cannot remove while iterating
        for peer_id in self._writers:
            if self._writers[peer_id].get_last_push() + timedelta(seconds=15) < datetime.now():
                peers_to_delete.append(peer_id)
        # Remove old peers
        for peer in peers_to_delete:
            self._writers[peer].stop()
            del self._writers[peer]

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


def main():
    for index in range(p.get_device_count()):
        info = p.get_device_info_by_index(index)
        if info['name'] == 'default':
            global output_device_index
            output_device_index = info['index']

    # Init ROS
    rclpy.init()
    mixer = AudioMixerROS()
    rclpy.spin(mixer)
    mixer.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
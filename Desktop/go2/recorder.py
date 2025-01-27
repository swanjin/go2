import pyaudio
import wave
import io
import threading
import queue
import time

class KeyboardThread(threading.Thread):
    def __init__(self, q, name='keyboard-input-thread'):
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.q = q
        # print("keyboard thread start")
        self.start()

    def run(self):
        # print("keyboard thread run")
        self.q.put(input())  # Enter to stop recording


class SpeechByEnter:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    NUM_SAMPLES = 1536
    CONFIDENCE_UPPER_BOUND = 0.8
    CONFIDENCE_LOWER_BOUND = 0.008

    model: any
    utils: any
    paudio: pyaudio.PyAudio
    stream: pyaudio.PyAudio.Stream
    audio_chunks = []
    def setup(self):
        self.paudio =  pyaudio.PyAudio()
    
    def stream_open(self):
        self.stream = self.paudio.open(format = self.FORMAT, 
                            channels = self.CHANNELS,
                            rate = self.RATE,
                            input = True,
                            frames_per_buffer = self.CHUNK)

    def start_recording(self):
        input("Press Enter to start recording.") # Enter to start recording
        print("Recording in progress. Press Enter to stop when finished.")
        self.stream_open()
        self.audio_chunks = []
            
    def stop_recording(self):
        input_queue = queue.Queue()
        keyboardThread = KeyboardThread(input_queue)
        # print("stop_recording ready")
        while True:
            # print(datetime.datetime.now())
            audio_chunk = self.stream.read(self.NUM_SAMPLES * 10)
            # print(datetime.datetime.now())
            self.audio_chunks.append(audio_chunk)
            if not input_queue.empty():
                print("Recording has ended.")
                return
            time.sleep(0.1)
    
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.paudio.terminate()

    def voice_to_wav(self):
        container = io.BytesIO()
        container.name = 'buffer.wav'
        waveform = wave.open(container,'wb')
        waveform.setnchannels(self.CHANNELS)
        waveform.setsampwidth(self.paudio.get_sample_size(self.FORMAT))
        waveform.setframerate(self.RATE)
        waveform.writeframes(b''.join(self.audio_chunks))
        waveform.close()
        container.seek(0)
        return container

    def recording(self, callback_for_start = None):
        self.start_recording()
        if callback_for_start:
            callback_for_start()
        self.stop_recording()
        return self.voice_to_wav()

if __name__ == "__main__":
    speech = SpeechByEnter()
    speech.setup()
    container = speech.recording(lambda: print("recording start by enter"))

    file = open("test.wav", "wb")
    file.write(container.getbuffer())
    file.close()
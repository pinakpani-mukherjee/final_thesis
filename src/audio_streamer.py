import pyaudio
import wave
import pendulum

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
RECORD_SECONDS = 5
RECORD_LIMIT = 60

def run():
    print("Getting Audio")
    while True:
        p = pyaudio.PyAudio();
        WAVE_OUTPUT_FILENAME = f"{pendulum.now(tz='Asia/Tokyo').strftime('%Y-%m-%d_%H_%M_%S')}_output.wav"
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(f'./sample_audios/mic_audio_files/{WAVE_OUTPUT_FILENAME}', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

if __name__ == "__main__":
    current_time = pendulum.now(tz="Asia/Tokyo")
    while True:
        if current_time.second % RECORD_SECONDS == 0:
            run()
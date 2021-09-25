import pyaudio
import wave
import argparse
from board import D4
import adafruit_dht
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=float, help='N samples')
args = parser.parse_args()


dht_device = adafruit_dht.DHT11(D4)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = RATE // 10
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "audio"

if not os.path.exists("raw_data/"):
    os.makedirs("raw_data/")
    
with open("raw_data/samples.csv", "w+") as f:
    for i in range(int(args.n)):
        now = datetime.now()
        try:
            temperature = dht_device.temperature
            humidity = dht_device.humidity
        except RuntimeError:
            continue
        f.write(f"{now.year}/{now.month}/{now.day},{now.hour}:{now.minute}:{now.second},{temperature},{humidity},{WAVE_OUTPUT_FILENAME + str(i) + '.wav'}\n")


        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        print("*** Recording ***")
        # start = time.time()
        for k in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # end = time.time()
        # print("*** Recording done ***")
        # print(f"Sensing time: {end - start}")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # start = time.time()
        wf = wave.open("raw_data/"+ WAVE_OUTPUT_FILENAME + str(i) + ".wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        # end = time.time()
        # storage_time = end - start

        # print(f"Storage time: {end - start}")




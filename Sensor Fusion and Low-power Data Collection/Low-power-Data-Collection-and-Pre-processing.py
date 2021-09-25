from scipy import signal

import argparse
import datetime
import time
import tensorflow as tf
import pyaudio
import numpy as np
import os
import io
import subprocess

CHANNEL = 1
RECORD_SECS = 1
RATE = 48000
CHUNK = RATE //10
SAMPLING_RATE = 16000
RESOLUTION = pyaudio.paInt16

MEL_BINS = 40
LOW_FREQ = 20
UP_FREQ = 4000
MIN = -32767
MAX = 32768

parser = argparse.ArgumentParser()
parser.add_argument('--num-samples', type=int, help='num samples', required=True)
parser.add_argument('--output', type=str, help='output folder', required=True)
args = parser.parse_args()

output_folder = args.output
num_samples = args.num_samples 

# Inizialize common variables    
samples = int(RATE / CHUNK * RECORD_SECS)
frame_length = int(SAMPLING_RATE * 0.04) # f(hz) * l
frame_step = int(SAMPLING_RATE * 0.02) # f(hz) * s(ms)
sampling_ratio = RATE / SAMPLING_RATE

p = pyaudio.PyAudio()

stream = p.open(format = RESOLUTION,
                channels = CHANNEL,
                rate = RATE,
                input = True,
                frames_per_buffer=CHUNK)
stream.stop_stream()

num_spectrogram_bins = 321
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( MEL_BINS,
                                                                    num_spectrogram_bins,
                                                                    SAMPLING_RATE,
                                                                    LOW_FREQ,
                                                                    UP_FREQ)
# Set 600Mhz
subprocess.Popen(['sudo', '/bin/sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])

# Reset the monitors
subprocess.Popen(['sudo', '/bin/sh', '-c', 'echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset'])

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(num_samples):
    start = time.time()
    temp_file = io.BytesIO()

    # RECORDING 
    stream.start_stream()
    
    # SET 600Mhz
    subprocess.Popen(['sudo', '/bin/sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])
    
    for k in range(samples):
        if k == samples -1:
            # SET 1.5Ghz
            subprocess.Popen(['sudo', '/bin/sh', '-c', 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])
        
        temp_file.write(stream.read(CHUNK))
        
    
    stream.stop_stream()

    # SAMPLING
    temp_file.seek(0)  #move the cursor back to the beginning of the "file"
    audio = signal.resample_poly(np.frombuffer(temp_file.getbuffer(), dtype = np.int16), 1, sampling_ratio)
    audio = audio.astype(np.int16)
    
    #STFT
    tf_audio = tf.convert_to_tensor(audio/MAX, dtype=tf.float32)
    stft = tf.signal.stft(tf_audio,
                          frame_length=frame_length,
                          frame_step=frame_step,
                          fft_length=frame_length)
    spectrogram = tf.abs(stft)

    # MFCCs
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :10]
    mfccs_byte = tf.io.serialize_tensor(mfccs)
    
    # SAVE MFCC FILE
    tf.io.write_file(output_folder +'/mfccs'+str(i)+'.bin', mfccs_byte)
    temp_file.close()
    
    end = time.time()
    print('{:.3f}s'.format(end - start))
    
stream.close()
p.terminate()
    
# Read the monitors
subprocess.Popen(['cat /sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state'], shell=True)

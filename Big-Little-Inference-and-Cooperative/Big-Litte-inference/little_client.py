import argparse
import numpy as np
from subprocess import call
import tensorflow as tf
import zlib
import os
import requests
from sys import getsizeof
import base64
import datetime
import json

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True,
#         help='model full path')
# args = parser.parse_args()
# model = args.model
model = "little.tflite.zlib"
# Set to performance for faster 
call('sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
            shell=True)

# STFT and MFCC parameters
rate = 16000
length = 320
stride = 320
num_mel_bins = 20
num_coefficients = 7
num_frames = (rate - length) // stride + 1
num_spectrogram_bins = length // 2 + 1
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, rate, 20, 4000)

# Read little model
tflite_model = zlib.decompress(open(model, 'rb').read())
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load test data
zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')
test_files = np.loadtxt("kws_test_split.txt", dtype=str, delimiter=",", unpack=False)
with open("./labels.txt", "r") as f:
    labels = f.readline().split(" ")

class DatasetGenerator:
    def __init__(self, files, labels):
        self.labels = labels
        self.ds = self.make_dataset(files)

    def get_ds(self):
        return self.ds

    def make_dataset(self, files):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(1)
        ds = ds.cache()
        return ds

    def preprocess(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        return audio_binary, label_id

test_ds = DatasetGenerator(test_files, labels).get_ds()

accuracy = 0
count = 0
comm_size = 0

for x, y_true in test_ds:
    # STFT
    original_audio = x.numpy()[0]
    audio, _ = tf.audio.decode_wav(original_audio)
    audio = tf.squeeze(audio, axis=1)
    # Pad audio
    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([16000])
    stft = tf.signal.stft(audio, length, stride,
            fft_length=length)
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    mfccs = tf.expand_dims(mfccs, -1)
    mfccs = tf.expand_dims(mfccs, 0)
    input_tensor = mfccs
    
    # Predict audio label
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    y_pred = y_pred.squeeze()

    # Check delta between top 2 guesses
    best_two = y_pred[np.argsort(y_pred)[-2:]]
    diff = abs(best_two[0] - best_two[1])
    if diff <= 2.3:
        timestamp = int(datetime.datetime.now().timestamp())
        audio_b64 = base64.b64encode(original_audio)
        audio = audio_b64.decode()
        senml = {
            'bn': 'little_model',
            'e': [
                { 'n': 'audio', 'u': '/', 't': timestamp, 'vd': audio}
            ]
        }
        comm_size += getsizeof(json.dumps(senml))
        response = requests.post("http://192.168.1.149:8080/big_model", json = senml)
        y_pred = np.array(int(response.json()['y_pred']))
    else:
        y_pred = np.argmax(y_pred)

    y_true = y_true.numpy().squeeze()
    accuracy += y_pred == y_true
    count += 1

accuracy /= float(count)

print(f'Accuracy {accuracy*100:.2f}')
print(f"Communication cost: {comm_size/2**20:.3g} MB")
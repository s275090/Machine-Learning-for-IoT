# On Raspberry Pi
from DoSomething import DoSomething
import tensorflow as tf
import base64
import datetime
import time
import json
import os
import numpy as np

root = '010121'

class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                self.lower_frequency, self.upper_frequency)
        self.preprocess = self.preprocess_with_mfcc

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(1)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


options = {'frame_length': 640, 'frame_step': 320,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}

class CooperativeClient(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        if topic.split('/')[2] not in y_preds:
            y_preds[topic.split('/')[2]] = []

        y_preds[topic.split('/')[2]].append(np.frombuffer(base64.b64decode(input_json["y_pred"]), dtype=np.float32))



if __name__ == "__main__":
    test = CooperativeClient("collector")
    test.run()
    test.myMqttClient.mySubscribe(os.path.join(root, 'prediction', '#'))

    y_preds = {}

    zip_path = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

    data_dir = os.path.join('.', 'data', 'mini_speech_commands')

    test_files = np.loadtxt("kws_test_split.txt", dtype=str, delimiter=",", unpack=False)

    LABELS = np.loadtxt("labels.txt", dtype=str, delimiter=",", unpack=False).tolist().split(" ")

    generator = SignalGenerator(LABELS, 16000, **options)
    test_ds = generator.make_dataset(test_files, False)
    ii = 0

    for (x, _), ii in zip(test_ds, range(len(test_files))):
        now = datetime.datetime.now()
        timestamp = int(now.timestamp())
        audio_b64bytes = base64.b64encode(x)
        audio_string = audio_b64bytes.decode()

        output = {
            "bn": "http://192.168.1.174",
            "bt": timestamp,
            "e": [
                {"n": "audio", "u": "/", "t": 0, "vd": audio_string}
            ]
        }

        output = json.dumps(output)
        test.myMqttClient.myPublish(os.path.join(root, 'audio', str(ii)), output)

        #while str(ii) not in y_preds:
        #    pass

    while len(y_preds.values()) != len(test_files):
        pass

    accuracy = 0
    count = 0
    for (_, y_true), y_pred in zip(test_ds, list(y_preds.values())):
        y_pred = np.asarray(y_pred)
        y_pred = y_pred.sum(axis=0)
        y_pred = np.argmax(y_pred)
        y_true = y_true.numpy().squeeze()
        accuracy += y_pred == y_true
        count += 1

    accuracy /= float(count)
    print('Accuracy {:.2f}'.format(accuracy * 100))
    test.end()

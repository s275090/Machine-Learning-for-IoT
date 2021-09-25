import argparse
import os
import numpy as np
import os
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='version model')
args = parser.parse_args()

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

train_files = np.loadtxt("kws_train_split.txt", dtype=str, delimiter=",", unpack=False)
val_files = np.loadtxt("kws_val_split.txt", dtype=str, delimiter=",", unpack=False)
test_files = np.loadtxt("kws_test_split.txt", dtype=str, delimiter=",", unpack=False)

LABELS = np.loadtxt("labels.txt", dtype=str, delimiter=",", unpack=False).tolist().split(" ")

class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

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

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

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
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


options = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}
strides = [2, 1]


generator = SignalGenerator(LABELS, 16000, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)

##### Model version
if args.version == "1":
    lr = 0.01
    # CNN
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=strides, use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(8)
])
elif args.version == "2":
    lr = 0.001
    # CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=strides, use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8)
    ])
elif args.version == "3":
    lr = 0.001
    # DS-CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=strides, use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8)
    ])

elif args.version == "4":
    lr = 0.05
    # DS-CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=strides, use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8)
    ])


loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=lr)
metrics = [tf.metrics.SparseCategoricalAccuracy()]
checkpoint_filepath = './checkpoints/kws_v{}/weights'.format(args.version)
cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                        save_weights_only=True, monitor='val_sparse_categorical_accuracy',
                                        mode='max',save_best_only=True)

if not os.path.exists(os.path.dirname(checkpoint_filepath)):
    os.makedirs((os.path.dirname(checkpoint_filepath)))

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[cp])

print(model.summary())
model.load_weights(checkpoint_filepath)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

##### Save tflite  model
tflite_model_dir = './{}.tflite'.format(args.version)

with open(tflite_model_dir, 'wb') as fp:
    fp.write(tflite_model)

##### Evaluation on test set
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_ds = test_ds.unbatch().batch(1)

accuracy = 0
count = 0
for x, y_true in test_ds:
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    y_pred = y_pred.squeeze()
    y_pred = np.argmax(y_pred)
    y_true = y_true.numpy().squeeze()
    accuracy += y_pred == y_true
    count += 1

accuracy /= float(count)
print('Test Accuracy {:.2f}'.format(accuracy*100))
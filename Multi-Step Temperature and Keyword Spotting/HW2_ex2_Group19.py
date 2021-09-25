import argparse
import os
import numpy as np
import tensorflow as tf
import zlib
import tensorflow_model_optimization as tfmot

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='model')
args = parser.parse_args()

version = args.version

# sparse0.8+structured0.4+PTQ weight LR 0.01 EPOCH15 
# ACC: 91.62 SIZE: 19100 bytes
if version == 'a': 
    q_aware = False
    magnitude_pruning = True
    FINAL_SPARSITY = 0.8
    ALPHA = 0.4
    weight_quantization = 'PTQ'
    lr = 0.01
    epochs = 15
    mfcc_coeffs = 10
    mfcc_bins = 40
    frame_length = 640
    frame_step = 320

# Q Aware + Structured0.3+PTQ Weight F16 
# ACC: 91.38 SIZE: 29256 bytes Inference Latency: 0.9 
elif version == 'b': 
    q_aware = True
    ALPHA = 0.3
    weight_quantization = 'F16'
    epochs = 20
    lr = 0.001
    mfcc_coeffs = 10
    mfcc_bins = 40
    frame_length = 640
    frame_step = 320

# Structured0.4 + PTQ W + LR 0.005 + Epochs 25 + coeff 7 + bins20 + length 320 
# ACC: 92.62 SIZE: 43825 Total Latency: 36.94
elif version == 'c': 
    magnitude_pruning = False
    q_aware = False
    ALPHA = 0.4
    weight_quantization = 'PTQ'
    epochs = 25
    lr = 0.005
    mfcc_coeffs = 7
    mfcc_bins = 20
    frame_length = 320
    frame_step = 320

zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

train_files = np.loadtxt("kws_train_split.txt", dtype=str, delimiter=",", unpack=False)
val_files = np.loadtxt("kws_val_split.txt", dtype=str, delimiter=",", unpack=False)
test_files = np.loadtxt("kws_test_split.txt", dtype=str, delimiter=",", unpack=False)

LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
LABELS = LABELS[LABELS != 'README.md']

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


options = {'frame_length': frame_length, 'frame_step': frame_step, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': mfcc_bins,
        'num_coefficients': mfcc_coeffs}


strides = [2, 1]
generator = SignalGenerator(LABELS, 16000, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)

if version == 'b':
    mod = 'dscnn'
    input_shape = [49, 10, 1]
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(filters=int(256*ALPHA), kernel_size=[3, 3], strides=strides, use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.Conv2D(filters=int(256*ALPHA), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.Conv2D(filters=int(256*ALPHA), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8)
    ])
else:
    mod = 'cnn'
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=int(128*ALPHA), kernel_size=[3, 3], strides=strides, use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=int(128*ALPHA), kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=int(128*ALPHA), kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8)
    ])

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=lr)
metrics = [tf.metrics.SparseCategoricalAccuracy()]
checkpoint_filepath = './checkpoints/kws_{}_mfcc/weights'.format(mod)
cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                        save_weights_only=True, monitor='val_sparse_categorical_accuracy',
                                        mode='max',save_best_only=True)

if not os.path.exists(os.path.dirname(checkpoint_filepath)):
    os.makedirs((os.path.dirname(checkpoint_filepath)))

if q_aware:  # q aware training
    # Helper function uses `quantize_annotate_layer` to annotate that only the
    # Dense layers should be quantized.
    def apply_quantization_to_dense(layer):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense`
    # to the layers of the model.
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_quantization_to_dense,
    )

    # Now that the Dense layers are annotated,
    # `quantize_apply` actually makes the model quantization aware.
    model = tfmot.quantization.keras.quantize_apply(annotated_model)
    model.summary()

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[cp])

elif magnitude_pruning == True:
    print("MAGNITUDE PRUNING")
    # Magnitude-based Pruning
    pruning_params = {'pruning_schedule':tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.30,
                    final_sparsity=FINAL_SPARSITY,
                    begin_step=len(train_ds)*5,
                    end_step=len(train_ds)*15)
    }

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    model = prune_low_magnitude(model, **pruning_params)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    callbacks = [cp, tfmot.sparsity.keras.UpdatePruningStep()]

    input_shape = [32, 49, 10, 1]
    model.build(input_shape)
    model.summary()

    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
    model = tfmot.sparsity.keras.strip_pruning(model)

else:
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[cp])
    model.summary()


model.load_weights(checkpoint_filepath)

if weight_quantization == 'PTQ':
    # INTEGER QUANTIZATION (ONLY WEIGHTS)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

elif weight_quantization == 'F16':
    # FLOAT16 QUANTIZATION (ONLY WEIGHTS)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    
##### Save tflite compressed model
tflite_model_dir = './Group_19_kws_{}.tflite.zlib'.format(version)

with open(tflite_model_dir, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    #fp.write(tflite_model)
    fp.write(tflite_compressed)

print('Size tflite: {} bytes'.format(os.path.getsize(tflite_model_dir)))


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
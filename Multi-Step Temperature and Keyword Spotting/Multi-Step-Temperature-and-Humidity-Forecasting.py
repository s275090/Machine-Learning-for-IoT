import argparse
import numpy as np
import os
import pandas as pd
import zlib
import tensorflow as tf
import tensorflow_model_optimization as tfmot

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='version')
args = parser.parse_args()

version = args.version

if version == "a":
    ALPHA = 0.1 # Width Scaling
    FINAL_SPARSITY = 0.7 # Magnitude-based Pruning
elif version == "b":
    ALPHA = 0.1  # Width Scaling
    FINAL_SPARSITY = 0.9  # Magnitude-based Pruning

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

INPUT_WIDTH = 6
LABEL_OPTIONS = 2
OUTPUT_WIDTH = 6

class WindowGenerator:
    def __init__(self, input_width, output_width, label_options, mean, std):
        self.input_width = input_width
        self.output_width = output_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        inputs = features[:, :-6, :]
        labels = features[:, 6:, :]

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.output_width, 2])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length= self.input_width + self.output_width,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=(2,))
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0,1])

        self.total.assign_add(error)
        self.count.assign_add(1.)
        return

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        return result

generator = WindowGenerator(INPUT_WIDTH, OUTPUT_WIDTH, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

##### Models

# Multilayer perceptron MLP
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(int(128*ALPHA), activation='relu'),
    tf.keras.layers.Dense(int(128*ALPHA), activation='relu'),
    tf.keras.layers.Dense(2*OUTPUT_WIDTH),
    tf.keras.layers.Reshape((OUTPUT_WIDTH, 2))
])

# Magnitude-based Pruning
pruning_params = {'pruning_schedule':tfmot.sparsity.keras.PolynomialDecay(
                  initial_sparsity=0.30,
                  final_sparsity=FINAL_SPARSITY,
                  begin_step=len(train_ds)*5,
                  end_step=len(train_ds)*15)
}

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model = prune_low_magnitude(model, **pruning_params)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=MultiOutputMAE()
)

callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

input_shape = [32, 6, 2]
model.build(input_shape)
# model.summary()

model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)

model = tfmot.sparsity.keras.strip_pruning(model)

# FLOAT16 QUANTIZATION WEIGHT
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

##### Save tflite model
tflite_model_dir = './Group19_th_{}.tflite.zlib'.format(version)

with open(tflite_model_dir, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)

print('Size tflite: {}bytes'.format(os.path.getsize(tflite_model_dir)))

##### Evaluate on the test set
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mae = 0
count = 0
test_ds = test_ds.unbatch().batch(1)

for x, y_true in test_ds:
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    y_pred = y_pred.squeeze()
    y_true = y_true.numpy().squeeze()
    mae += tf.reduce_mean(np.absolute(y_pred - y_true), axis=0)
    count += 1

mae /= int(count)

print('MAE: {}'.format(mae))

import tensorflow as tf
import numpy as np
import argparse
from scipy.io import wavfile
from scipy import signal
import os
import datetime
import time


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Input folder')
parser.add_argument('--output', type=str, help='Output file')
args = parser.parse_args()

inputDir = args.input
outputFile = args.output
if not inputDir.endswith("/"):
    inputDir += "/"

with tf.io.TFRecordWriter(outputFile) as writer:
    with open(inputDir + "samples.csv", "r") as f:
        for line in f.readlines():
            date = line.split(",")[0]
            hour = line.split(",")[1]
            temperature = int(line.split(",")[2])
            humidity = int(line.split(",")[3])
            audioFile = line.split(",")[4].rstrip("\n")
            # POSIX timestamp (32/64 bit int)
            dt = datetime.datetime.strptime(date + " " + hour, "%Y/%m/%d %H:%M:%S")
            posix_dt = int(time.mktime(dt.timetuple()))
            # Sample audio file
            audio = tf.io.read_file(inputDir + audioFile)
            audio = tf.io.serialize_tensor(audio)
            # Map features for Example)
            timestamp_feature = tf.train.Feature(int64_list = tf.train.Int64List(value = [posix_dt]))
            temperature_feature = tf.train.Feature(int64_list = tf.train.Int64List(value = [temperature]))
            humidity_feature = tf.train.Feature(int64_list = tf.train.Int64List(value = [humidity]))
            audio_feature = tf.train.Feature(bytes_list = tf.train.BytesList(value = [audio.numpy()]))

            mapping = {
                'timestamp': timestamp_feature,
                'temperature': temperature_feature,
                'humidity': humidity_feature,
                'audio': audio_feature
            }

            example = tf.train.Example(features = tf.train.Features(feature = mapping))

            writer.write(example.SerializeToString())

print(f"TFRecord file size: {os.path.getsize(outputFile)}")
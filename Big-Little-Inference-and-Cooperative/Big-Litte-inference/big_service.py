import cherrypy
import json
import numpy as np
import tensorflow as tf
import zlib
import os
import base64

class BigModel(): 

    def __init__(self):
        self.model_dir = "./big.tflite"
        self.rate = 16000
        self.length = 640
        self.stride = 320
        self.num_mel_bins = 40
        self.num_coefficients = 10
        self.num_frames = (self.rate - self.length) // self.stride + 1
        self.num_spectrogram_bins = self.length // 2 + 1
        self.input_shape = [1, 49, 10, 1]

        with open("./labels.txt", "r") as f:
            self.labels = f.readline().split(" ")

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, self.num_spectrogram_bins, self.rate, 20, 4000)

        self.tflite_model = open(self.model_dir, 'rb').read()
        self.interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        self.interpreter.resize_tensor_input(0, self.input_shape)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, data):
        audio, _ = tf.audio.decode_wav(data)
        audio = tf.squeeze(audio, axis=1)
        # Pad audio
        zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([16000])
        # STFT
        stft = tf.signal.stft(audio, self.length, self.stride,
                fft_length=self.length)
        spectrogram = tf.abs(stft)
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]
        mfccs = tf.expand_dims(mfccs, -1)
        mfccs = tf.expand_dims(mfccs, 0)
        input_tensor = mfccs

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        y_pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        y_pred = y_pred.squeeze()
        y_pred = np.argmax(y_pred)

        return y_pred

    exposed = True
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def POST(self, *path, **query):
        senml = cherrypy.request.json
        events = senml['e']
        for event in events:
            if event['n'] == 'audio':
                audio = base64.b64decode(event['vd'])
                y_pred = self.predict(audio)
        output = {'y_pred': str(y_pred)}
        return output


if __name__ == '__main__': 
    conf = {
            '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True, }
        }
        
    cherrypy.tree.mount (BigModel(), '/', conf)
    cherrypy.config.update({'server.socket_host': '192.168.1.149'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start() 
    cherrypy.engine.block()


# call('sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
#             shell=True)

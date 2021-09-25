from DoSomething import DoSomething
import base64
import numpy as np
import time
import os
import json
import argparse
import zlib
import tensorflow as tf

root = '010121'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model_path')
args = parser.parse_args()


class InferenceClient(DoSomething):
    def __init__(self, clientID):
        super().__init__(clientID)

        tflite_model = open(args.model, 'rb').read()
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def notify(self, topic, msg):
        senml = json.loads(msg)
        events = senml['e']

        for event in events:
            if event['n'] == 'audio':
                audio_string = event['vd']
            else:
                raise RuntimeError('No audio event')

        audio = np.frombuffer(base64.b64decode(audio_string), dtype=np.float32)
        audio = np.reshape(audio, self.input_details[0]['shape'])

        self.interpreter.set_tensor(self.input_details[0]['index'], audio)
        self.interpreter.invoke()
        y_pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        pred_string = base64.b64encode(y_pred).decode()
        prediction_json = json.dumps({'y_pred': pred_string})
        test.myMqttClient.myPublish(os.path.join(root, 'prediction/', topic.split('/')[2]), prediction_json)

if __name__ == "__main__":
    test = InferenceClient(args.model)
    test.run()
    test.myMqttClient.mySubscribe(os.path.join(root, 'audio', '#'))

    while True:
        time.sleep(1)

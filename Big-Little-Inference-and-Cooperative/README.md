**Machine Learning for IoT**

Homework 3 - Group 19

**[Ex1] Big/Little inference**

Our goal is to implement a **Big/Little model** , ideally distributed by running the Little model on a Raspberry Pi device and the Big model on a computer/server, which performs inference for Keyword Spotting. The code performing **inference** on the **Little**** model **iterates over all test data, applying the needed preprocessing according to how the Little model was trained, and makes a prediction. The last Dense layer of the CNN returns an array of 8 values, one per class, where each value indicates how confident the model is that the test data belongs to that class. Normally we would consider the class yielding the largest value as the predicted class, but in this case we focus on the** difference **between the** two most probable classes (Score Margin or SM) **. If the SM is, in modulo,** larger ****than** a certain **threshold** (the value for the threshold was chosen in order to optimize accuracy and communication costs, both of which needed to be satisfied as hard constraints) then we perform **inference as usual** , by considering the highest value as predicted class and comparing it to the true value of our test data. If instead the **SM** is **smaller**** than **the** threshold **, that is the Little model&#39;s confidence in the estimation is not high enough, we** send **the test** data ****to the Big model** , which has a higher accuracy, to get a **more**** accurate ****prediction**. Communications between the two models are managed by means of the **REST** standard. In our scenario the **Big** model plays the role of the **server** , whereas the **Little** model is the **client**. Each time the Little model has low confidence in the prediction, we perform a **POST** request to the Big model, which receives the **raw**** audio **data encoded in** base64 **in** SenML+JSON **format. The Big model then performs the needed preprocessing and performs a** prediction **, which is** returned **in** JSON** format to the client. The models used are:

| **Version** | **Model** | **Parameters** | **Accuracy %** | **Details** |
| --- | --- | --- | --- | --- |
| Little | CNN withsparsity = 0.8, structured pruning = 0.4,post training quantization on weights | MFCC: length = 320, stride = 320, mel\_bins = 20, coeffs = 7
LR = 0.01, #Epochs = 15 | 91.5 | Size: 19479 bytes (compressed with zlib)
Latency: 36.9ms |
| Big | CNN | MFCC: length = 640, stride = 320, mel\_bins = 40, coeffs = 10
LR = 0.01, #Epochs = 20 | 94.5 |
 |

By using a static **threshold of 2.3** we manage to obtain an **accuracy** of **94.5%** with a **communication cost** of **4.44 MB**. By sending 110 requests to the big model (13.75%) we are able to reach the accuracy of the Big model, while using the less accurate but faster Little model for most of the predictions.

**[Ex2] Cooperative inference**

Our goal is to develop an application (_cooperative\_client_) that sequentially read an audio signal from the Speech Command test set, pre-processing it and send the pre-processed signal to N different _inference\_clients_ that classify its content, each with a different prediction model, and finally send the output back to compute the final prediction combining all the outcomes.

For this application we use **N = 4** inference\_client with each a different model.

| **N** | **Model\*** |
 | **Accuracy%** |
| --- | --- | --- | --- |
| 1 | CNN | LR = 0.01 | 94,50 |
| 2 | CNN | LR = 0.001 + add Conv2D layer | 94,75 |
| 3 | DS-CNN | LR = 0.001 + add Conv2D layer | 94,25 |
| 4 | DS-CNN | LR = 0.05 | 92,88 |

_\*Pre-processing: {&#39;frame\_length&#39;: 640, &#39;frame\_step&#39;: 320, &#39;mfcc&#39;: True, &#39;lower\_frequency&#39;: 20, &#39;upper\_frequency&#39;: 4000, &#39;num\_mel\_bins&#39;: 40, &#39;num\_coefficients&#39;: 10}_

Since there are more than one client, to implement **fast and efficient communication** between the different clients, it is important that the audio is sent only once to avoid engaging the _cooperative\_client_ in multiple sending of the same information and that everyone can process simultaneously and regardless of the producer/consumer of the information. For this application the most suitable protocol is **MQTT** that allows **asynchronous communication**. Another reason is that at the time of sending the audio it is not possible to know which and how many clients are waiting for the file that are online, with the help of the Message Broker this is not a problem.

The communication is based on **two topic** :

- _ **/audio/#** _ where _cooperative\_client_ publishes all audios (encoded in **base64** in **SenML+JSON format** ), assigning a progressive integer value (ex. _/audio/1_), and different _inference\_clients_ are subscribe;
- _ **/prediction/#** _ where the different _inference\_clients_, which have previously received the audio, after classification of them, publish their output (in **JSON format** ) assigning them the numeric value of the received audio topic (ex. _/prediction/1_) and _cooperative\_client_ is subscribed.

This configuration allows us to send the audio one after the other and at the same time to receive and save the various outputs, greatly reducing the time. Staying in this perspective, we have also chosen to reduce the **QoS level to 0 for the publication/subscription of the audio** , in relation to the fact that we have more than _inference\_clients_ and if someone does not receive the audio it does not matter.

![image](https://user-images.githubusercontent.com/58779561/134764236-646f850f-a764-4542-90bc-25226ea39743.png)

_Example of communication between Cooperative\_inference and Inference\_client\_N_

In the end, when _cooperative\_client_ has received at least one result for each audio, the final prediction of each audio is, by summing all the outputs received per audio, the position of the largest value in the array. The **accuracy** we are able to obtain with **4** _inference\_clients_ is **96%.** We get good results even using only two models but we preferred to choose 4 to reduce the risk of losses or unavailability.

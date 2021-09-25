**Machine Learning for IoT**

Homework 2 - Group 19

**[Ex1] Multi-Step Temperature and Humidity Forecasting**

Starting from the temperature and humidity of the Jena Climate Dataset, our goal is to implement multi-step forecasting respecting the hard constraint.

Observing the first results obtained with different models, we have chosen to use **MLP** as a model and we have adopted different optimizations to respect the constraints.

To increase sparsity, which allow us to use of a compress format to represent data, and to reduce redundancy we used two different pruning techniques: first of all we have applied the **structured pruning** approach, we introduced a parameter _width α ∈ [0,1)_ in the model that multiplies the number of filters of the convolutional layers and reduces their redundancy. Then to increase the sparsity we have to completely eliminate some weights and the corresponding connections during training, for this we used the approach of **magnitude-based pruning** , where to define how much sparsity to enforce in a training step, we chose the **PolinomialDecay** schedule by varying the sparsity between an initial value of 0.3 and a final value x. After training our model in float32 we perform a **F16 quantization** in TFlite, the weights are converted to float16. In the end, to see the model size advantage due to pruning we compress our model with **zlib**.

From the table below we can observe the different hyperparameters chosen and the results obtained for the two model versions.

|
**Ver** | **O**** p ****t**** i ****m**** i ****z****.** |
**Structured Pruning** _α_ | **Magnitud- Based Pruning** |
**Post-Training** |
**R**** e ****s**** u ****l**** t **|** MAE** |
**TFlite**  **kB** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **°C** | **Rh%** |
| **a** | 0.1 | Sparse 0.7 | F16 Q Weight | 0,46 | 1,75 | 1,7 |
| **b** | 0.1 | Sparse 0.9 | F16 Q Weight | 0,49 | 1,86 | 1,5 |

_Table1: Results obtained for the two different versions of the_ _ **mlp** __ **model** _

**[Ex2] Keyword Spotting**

The purpose of this exercise is to train three different models for keyword spotting on the original mini speech command dataset. For all three models we applied Mffc in the preprocessing step.

For the _version A_ we have constraints regarding only accuracy e model size, we chose CNN as the model. To reduce the size of the final file, we used the **structured pruning approach** , the **magnitude-based pruning** during the training phaseand a **weight quantization post-training**. To obtain a good accuracy, we varied also the Learning rate and the number of Epochs.

In _version B_ we have also constraints regarding inference latency. For this purpose we choose as a Model a DS-CNN with structured pruning approach and in this case a **quantization already during training** , for this we use the function _apply\_quantization\_to\_dense_ in order to quantize only the Dense layers. After training we also perform a **Float16 quantization** in TFlite.

In _version C,_ in order to respect the total latency constraint, we used again CNN model, we modified some pre-processing parameters related to using mfcc and we used the Structured pruning approach again with weight quantization post-training, but unlike before we train normally.

From the table below we can observe the different hyperparameters chosen and the results obtained for the two model versions. All three models have been compressed into **zlib**.

| **Ver** | **O**** p ****t**** i ****m**** i ****z**** a ****t**** i ****o**** n **|** Model **|** Hyper-parameters **|** Structured Pruning **_α_ |** Aware Training **|** Post- ****Training** |


**R**** e ****s**** u ****l**** t **|** Acc % **|** TFlite **_** kB **_ | _** Latency ms**_ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Epoch** | **LR** | **Pre-processing** | **Infer.** | **Total** |
| **a** | CNN | 15 | 0.01 | --mfcc | 0.4 | Final Sparsity 0.8 | PTQ Weight | 92,5 | 18,70 | / | / |
| **b** | DS-CNN | 20 | 0.001 | --mfcc | 0.3 | Quantization dense layers | F16 PTQ Weight | 91,12 | 28,57 | 0,9 | / |
| **c** | CNN | 25 | 0.005 | --mfcc --coeff 7 --bins 10 --length 320 | 0.4 | / | PTQ Weight |
 | 92,62 | 42,8 | 1,6 | 36,94 |

_Table2: Results obtained on test set with three different versions of models_

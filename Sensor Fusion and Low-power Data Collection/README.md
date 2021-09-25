**Machine Learning for IoT**

Homework 1 

**[Ex4] Data preparation: Sensor Fusion Dataset with TFRecord.**

Starting from a dataset consisting of one-second audio samples, temperature and humidity data collected at different datetimes, our goal was to transform this raw data into a TFRecord, a format used to store serialized binary data which is much easier to work on with the tensorflow library.

In this exercise we experimented with the various value data-types (_BytesList_, _FloatList_, _Int64List_) we use to store our raw data into the TFRecord in order to minimize the memory footprint of the end product.

Since the **temperature** range is 0°C - 50°C, and the **humidity** one is 20%-90%, python should handle them as **int8** type, so the **Int64List** data type works better as it accepts int8 and allows us to be more efficient. In fact, we have a higher file size when storing them as FloatList (float32 or float64) or BytesList.

The **POSIX timestamp** seems to get truncated if stored as a FloatList, so to avoid loss of information we used **Int64List** which results in a slightly larger file (1 more byte per entry compared to the FloatList) while maintaining data integrity.

As for the **audio** , we store it directly as a **serialized tensor** in a **BytesList** , as storing each entry as int16 or worse, float32, would be way more taxing on the memory.

By using the configuration mentioned above we were able to obtain a **TFRecord file size** of **961600 bytes for 10 records** , compared to the **raw audio data** , which is the main contributor, occupying **960440 bytes** , thus keeping memory impact consistent. As a reference, storing the audio file (the most impacting data of our dataset) as Int64List results in a ~2.5MB TFRecord and ~2MB when storing it as FloatList.

**[Ex5] Low-power Data Collection and Pre-processing\***

In this exercise we have tried to limit the power consumption of an acquisition and pre-processing loop, respecting the hard constraint for the pre-processing part (80 ms).

In a first analysis, we measured the time taken by the board for a single sample at 600MHz. We could observe that the part that requires the highest computational level is the calculation of STFT and where it is necessary to increase the frequency of the cpu.

![image](https://user-images.githubusercontent.com/58779561/134763766-ac3595e0-0ab7-492d-aab3-3417b31c9d0d.png)

_Chart: Acquisition&amp;Pre-processing time at 600Mhz (1.157s)_

So to improve the performance of our code, first, we instantiated all the constant variables outside the iteration so as not to waste time for the instance. We observed that the **frequency change is not instantaneous** , when we go from a lower frequency to a higher one the CPU requires more power and this step takes some time. To be sure that the CPU is running at 1.5GHz during the STFT calculation, we have instantiated the modification of the _scaling\_governor_ in the last chunk within the stream of the recording phase. And finally we return to powersave at the beginning of each recording. These are the results we get.

| **@VF** | **600MHz** | **1.5GHz** | **600MHz + 1.5GHz** |
| --- | --- | --- | --- |
| **Avg Total Time (s)** | _1.142s_ | _1.060s_ | _1.065s_ |
| **Performance Monitor** | _ **600MHz** _ | _576_ | _0_ | _478_ |
| _ **1.5GHz** _ | _0_ | _532_ | _55_ |

_Table: 5 samples 1seconds audio signals after some code improvements_

We are able to obtain a good 89% of the cycles at 600MHz and the remaining 11% at maximum frequency with a gain of 77ms. Usually the first recording is slower (respect the hard constraint) because the raspberry wastes time in loading the variables.

**\*** Ex5 results were obtained on Linux raspberrypi 5.4.72-v7l+ #1356 SMP Thu Oct 22 13:57:51 BST 2020 armv7l GNU/Linux

# Spectrogram Recognition
This is the main content of my graduation project. It can acquire spectrum(spectrogram) from RSA306B real-time spectrum analyzer. After acquisition, we can put all the training set(spectrums) into convolutional neural network which I choose Inception-v3. Acquire again, then we can test the right percent of test set(spectrum We just acquired). I also made a GUI which is convenient.

![GUI](https://github.com/zhaoke2018/Spectrogram-Recognition/blob/master/gui.png "GUI")

## Environment
### Hardware
1. Tektronix RSA306B real-time spectrum analyzer
### Software
1. Python 3.6.4
2. RSA API
3. TensorFlow 1.7.0
4. TensorFlow Hub

## Run
1. put combine.py and RSA_API.py together
2. Open combine.py or "python combine.py"
3. Set center frequency
4. Set storage path
5. Press the "acquire" button, then we can see the real-time spectrum
6. Copy the storage path to the right side
7. Press the "test" button(about 5~10 seconds)

## Reference
https://www.tensorflow.org/tutorials/image_retraining
https://github.com/tkzilla/rsa_api_python_36

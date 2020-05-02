# TensorFlow Lite Cough Classification

## Getting Started

### Prerequisites
- Python 3.5+
- Keras 2.1.6 or higher
- pandas and pandas-ml
- TensorFlow 1.5 or higher

## Sick and Not Sick Dataset
The dataset has 24 audio each for the sick and not sick people under data/* folder.
https://osf.io/4pt2s/

### Classes
In this example, only 2 classes will be picked for TensorFlow 
`not sick, sick`


## Audio Processing
Data generation involves producing raw PCM wavform data containing a desired number of samples and at fixed sample rate and the following configuration is used

| Samples        | Sample Rate           | Clip Duration (ms)  |
| ------------- |:-------------:| -----:|
| 24      | 16000 | 5000 |

## Model Architecture
It is mostly a time stacked VGG-esque model with 1D Convolutions for temporal data such as audio waveforms. A one-dimensional dilated convolutional layer serving as the context convolution (`context_conv`) is employed to extract a wider field of the data. This is followed by a dimensionality reduction in the `reduce_conv` layer which employes 1-D MaxPooling to reduce the number of parameters passed to the subsequent layer.

## Training
The model can be trained by running train.py

### Usage
```
python train.py [-h] [-sample_rate SAMPLE_RATE] \
              [-batch_size BATCH_SIZE] \
              [-output_representation OUTPUT_REPRESENTATION] \
              -data_dirs DATA_DIRS [DATA_DIRS ...]
```

### Example of training the model
```
python train.py -sample_rate 16000 -batch_size 64 -output_representation raw -data_dirs data/train
```

## Results
After the training the model for 100 epochs, the following confusion matrix was generated for assessing classification performance.


| Predicted     | _silence_     | sick   | not_sick  
| ------------- |:-------------:| ------:| ---:| ------:| ---:| -----:| ----:| -------:| -------:| ------:| ----:|-----:|
| Actual |
_silence_ | 20 | 0 | 0 
sick | 0 | 16 | 8 
not_sick | 0 | 11 | 13  

## Built With

* [Keras](https://keras.io/) - Deep Learning Framework
* [TensorFlow](http://tensorflow.org/) - Machine Learning Library

## References
https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html


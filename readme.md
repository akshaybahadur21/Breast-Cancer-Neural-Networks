# Breast Cancer Classifier (Shallow Network) [![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Breast-Cancer-Neural-Networks/blob/master/LICENSE.txt)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)
This code helps you classify malignant and benign tumors using Neural Networks.

### Sourcerer
[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/images/0)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/links/0)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/images/1)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/links/1)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/images/2)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/links/2)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/images/3)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/links/3)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/images/4)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/links/4)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/images/5)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/links/5)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/images/6)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/links/6)[![](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/images/7)](https://sourcerer.io/fame/akshaybahadur21/akshaybahadur21/Breast-Cancer-Neural-Networks/links/7)

### Code Requirements
The example code is in Matlab ([R2016](https://in.mathworks.com/help/matlab/) or higher will work). 


### Description
An ANN is based on a collection of connected units or nodes called artificial neurons (analogous to biological neurons in an animal brain). Each connection (synapse) between neurons can transmit a signal from one to another. The receiving (postsynaptic) neuron can process the signal(s) and then signal downstream neurons connected to it. In common ANN implementations, the synapse signal is a real number, and the output of each neuron is calculated by a non-linear function of the sum of its input. Neurons and synapses may also have a weight that varies as learning proceeds, which can increase or decrease the strength of the signal that it sends downstream. Further, they may have a threshold such that only if the aggregate signal is below (or above) that level is the downstream signal sent.

<img src="https://github.com/akshaybahadur21/Breast-Cancer-Neural-Networks/blob/master/neural.png">

For more information, [see](https://en.wikipedia.org/wiki/Artificial_neural_network)

### Some Notes
1) Dataset- UCI-ML
2) I have used 30 features to classify
3) Instead of 0=benign and 1=malignant, I have used 1=benign and 2=malignant

### Accuracy ~ 92%

### Working Example
<img src="https://github.com/akshaybahadur21/Breast-Cancer-Neural-Networks/blob/master/cancer_neural.gif">

### Execution
To run the code, type `run cancer.m`

```
run cancer.m
```

## Python  Implementation

##### Used a shallow neural net with one hidden layer and 20 units.

#### I have used a linear learning rate decay for decreasing cost without overshooting the minima.

1) Dataset- UCI-ML
2) I have used 30 features to classify
3) Instead of 0=benign and 1=malignant, I have used 1=benign and 2=malignant

### Acuracy ~ 94%

<img src="https://github.com/akshaybahadur21/Breast-Cancer-Neural-Networks/blob/master/bc_nn.gif">

### Execution
To run the code, type `python B_Cancer_nn.py`

```
python B_Cancer_nn.py
```


## Welcome to Jianwei Wu and Vignesh Raghav's CISC849 Final Report

This Website is for CISC849 Spring 2019 by Team 3: Jianwei Wu and Vignesh Raghav 

# Classification of Medical Images by Using Deep Learning Techniques
## Breast Cancer Classification
### Malarial Cell Classification

## Preliminary Plan

We plan to discover the possibility to:
* Extend existing machine-learning based technique to deep-learning based technique
* Implement a deep-learning based model for medical image classification

After we confirm it is feasible to implement such neural network:
* Build a Convolutional Deep Neural Networks (CNN) to classify medical images

## State of the Art
From the three related works we mentioned in our preliminary plan and more:

First, most of the recent progress are related to computer-aided diagnosis (CAD) for detecting and analyzing breast cancer, some use Adaboost, some use SVM, and others also use neural networks like KNN (k-nearest neighbors) for classifying cancer images.

Second, we also learn about the very deep convolutional neural networks (VGGnet), and it’s ability to provide a significant improvement for large scale image recognition.

Third, we understand the reason for why depthwise separable convolutions work better than the traditional inception modules in neural networks.

Last, more about Alexnet.

## Introduction

After those literature research mentioned above, we prove the possibility to conduct such a project and related research.

Our project is now focusing on using the deep learning techniques to test on different tasks (datasets):

* First, we did our phrase one on the breast cancer image classification task.
 
* Second, we did our phrase two on malaria disease detection task.

## Phase 1 Breast Cancer Classification 

Experiments’ setup and experimental goal:

Machine
MacBook Pro (Retina, 13-inch, Early 2015) with macOS high sierra.

CPU only 
Intel core i5 with 16GB ram

Compile and running environment
Tensorflow-Keras (2.2.4-tf), python 3.4.3, numpy, matplot	

Our goal is to show if a patient has symptom of breast cancer (benign or malignant)

## VGGnet
![Image](https://jianwei-wu-1.github.io/CISC849_Report/p1.png)

* 21 layers
* 2-3 convolutional layers are followed by a pooling layer
* The end is fully-connected layers

## Our Neural Network Structure 
![Image](https://jianwei-wu-1.github.io/CISC849_Report/nn.png)

* Our Structure is similar to the VGGnet*.
* Instead of using sets of 3 conv layers like conventional VGG, we increase the number of layers for each time.
* We also extend the idea of “depthwise separable convolutions*.

## Updated Experimental Results
* Conventional VGGNet:
![Image](https://jianwei-wu-1.github.io/CISC849_Report/old.png)
![Image](https://jianwei-wu-1.github.io/CISC849_Report/old1.png)
* Our Neural Network:
![Image](https://jianwei-wu-1.github.io/CISC849_Report/new.png)
![Image](https://jianwei-wu-1.github.io/CISC849_Report/new1.png)



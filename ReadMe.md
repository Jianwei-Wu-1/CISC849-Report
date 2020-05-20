## Welcome to CISC849 Final Report Site for Jianwei Wu

For further questions/comments contact us at: wjwcis@udel.edu

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
![Image](https://jianwei-wu-1.github.io/CISC849-Report/p1.png)

* 21 layers
* 2-3 convolutional layers are followed by a pooling layer
* The end is fully-connected layers

## Our Neural Network Structure 
![Image](https://jianwei-wu-1.github.io/CISC849-Report/nn.png)

* Our Structure is similar to the VGGnet*.
* Instead of using sets of 3 conv layers like conventional VGG, we increase the number of layers for each time.
* We also extend the idea of “depthwise separable convolutions*.

Inspired by the middle result:
![Image](https://jianwei-wu-1.github.io/CISC849-Report/inp.png)

## Updated Experimental Results
* Conventional VGGNet:
![Image](https://jianwei-wu-1.github.io/CISC849-Report/old.png)
![Image](https://jianwei-wu-1.github.io/CISC849-Report/old1.png)
* Our Neural Network:
![Image](https://jianwei-wu-1.github.io/CISC849-Report/new.png)
![Image](https://jianwei-wu-1.github.io/CISC849-Report/new1.png)
(Jianwei Wu's Machine)

## Phase 1 conclusion:
Using “depthwise separable convolution” is more efficient than conventional VGGnet.
Also, it can outperform the conventional VGGnet.
The change of activation function to “selu” helps us to achieve even better accuracy.
Reduce likelihood of vanishing gradient(i.e. “dying ReLU”).
Better fit to our dataset.
A little bit more computational time but still less than the conventional VGGnet.

## Phase 2 Malaria Disease Detection
1. Used an architecture based on Alexnet with some modifications.
2. Number of Layers in the Neural Network = 5
3. Hardware Configuration = MacBook Pro(2014 Model) with macOS Mojave, 2.5 Ghz Intel Core i5 processor, 8GB Ram, Intel HD Graphics Card 4000, 1536 MB.
4. Software Configuration : Keras with Tensorflow Backend, Numpy,OpenCV, Python 2.7, Matplotlib, Scikit-Learn
5. Architecture Explanation
      -  5 Convolutional Layers
      -  3 Fully Connected Layers.
      -  RELU Configuration (Rectified Linear Units)
      -  Some Dropout Layers in the fully connected Layers.
## Phase 2 Results

![Image](https://jianwei-wu-1.github.io/CISC849-Report/1.png)
![Image](https://jianwei-wu-1.github.io/CISC849-Report/2.png)
![Image](https://jianwei-wu-1.github.io/CISC849-Report/3.png)
(Vignesh Raghav's machine)

## Analysis
 Precision - Proportion of Positive Identifications that are correct.
 Recall -  Proportion of Positive Identifications that are correct.
 F1 Score - Harmonic Average Score of Precision and Recall.
  
We are mainly interested in Precision as it helps identify the number of identifications that our  model classified correctly.
 For 10 Epochs - 90 % of the classifications are correct.
 For 20 Epochs - 95 % of the classifications are correct.
 For 50 Epochs  - 97 % of the classifications are correct.   

## Conclusion

* We can easily train a model that can classify results as positive and negative with more than 90% Accuracy 
* Potential Prototype to be used in the detection and treatment of Malaria.
* Deep Learning and Neural Networks can be used to significantly improve the way we detect Malaria.
* With further Optimization, it can be  much faster than the existing Blood Smear Test used to detect Malaria.

## Limitations

1. Can only classify medical images.
2. Requires a significant amount of data and computational power.
3. Requires Keras with Tensorflow Backend which is a dependency. We can build a model with Keras Alone.
4. From the above point, version updates can break the code. 

## Future work
Consider Hardware configurations before developing Convolutional Neural Networks.
Introduce Reinforcement Learning, where the Neural Network can learn itself and improve the accuracy
Increase the Accuracy further (More than 99%) as we do not want even a small proportion of classifications to be wrong as we are dealing with sensitive data.
Explore other libraries apart from Keras and Tensorflow.
Try to run out model on the cloud(ie: Amazon Sage Maker, Google Cloud, Databricks, IBM Watson) eliminating the need to have access to powerful machines.
Improve Data Visualization and Explore more Data Visualization Techniques(ie: Explore Tableau, Chartboost, Seaborn).
Extend this method to other diseases (ie: Tuberculosis)

## References

Imagenet Classification with Deep Convolutional Neural Networks by Alex Krizhvesky, Ilya Sutskever, Geoffrey.E.Hinton
Comparison of Convolutional Neural Networks for Food Image Classification - Gozde Ozsert Yigit
The History began from Alexnet : A comprehensive survey on Deep Learning  Approaches -  Vijayan.K.Asari,Zahangir Alom.
Coursera (Deep Learning.ai Specialization by Andrew Ng)
   -       Neural Networks and Deep Learning.
   -       Improving Deep Neural Networks : Hyperparameter turing, Regularization and Optimization.
   -        Convolutional Neural Networks.

Dataset Used: Malarial Dataset from NIH(National Institute of Health Sciences).       
https://machinethink.net/blog/convolutional-neural-networks-on-the-iphone-with-vggnet/
Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014), published at ICLR 2015*.
Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.*
Chollet, François. "Xception: Deep learning with depthwise separable convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.*
https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220738560542&proxyReferer=https%3A%2F%2Fwww.google.com%2F
https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks
Yassin, Nisreen IR, et al. "Machine learning techniques for breast cancer computer aided diagnosis using different image modalities: A systematic review." Computer methods and programs in biomedicine 156 (2018): 25-45.

## What should be better?
* Start from a remote server or a more powerful machine.

* Implement our solution with version control.

* Implement our solution with Microservices.

* Try to implement streaming tools like Kafka and Storm.


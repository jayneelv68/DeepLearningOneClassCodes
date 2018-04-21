# DeepLearningOneClassCodes
A repository for my deep learning codes, which are not part of a major project, yet substantial in accord.
MACHINE LEARNING INNOVATIVE ASSIGNMENT.


Augmenting dataset:


1
Rotate 90
0.5
2
Rotate 270
0.5
3
Flip left Right
0.8
4
Flig Top Bottom
0.3
5
Resize(pre-processing step)
120,120

SAMPLE 5000 images.

Model Parameters:

Epochs=25
Learning Rate 0.001

> Give integer labels as per the name of the image( using startswith method)
>Split data 75:25 for train:test.
Build Model and fit .


Model Layers:
Convultional 2D Layer 20 , (5,5)
Relu Activation
MAx Pooling pool size(2,2) strides(2,2)
Convultional 2D layer 50 (5,5)
Relu Activation
Max Pooling pool size(2,2) strides(2,2)
Flatten
Dense 500
Relu Activation
Dense
Softmax
---

Objectives:
1.Comparision between Augmented Training data and non-augmented training data.
2. Deployment of activation befor pooling layers.
3. Try to answer against possibility of overfitting due to repeated augmentation.


HOW TO RUN IT?

build_train.py
excecute it and change training data accordingly.

# Convolutional Neural Networks
<<<<<<< HEAD

In this chapter we introduce convolutional neural networks. They are
the first nontrivial *architecture* beyond the humble multilayer
perceptron. In their design scientists used inspiration from biology,
group theory, and lots of experimentation to achieve stunning results
in object recognition, segmentation, image synthesis and related
computer vision tasks. 'Convnets', as they are often called, have
become a cornerstone for deep learning research. Their applications
reach beyond images to audio, text, video, time series analysis,
graphs and recommender systems.

We will first describe the operating principles of the convolutional
layer and pooling layer in a convolutional neural network, and then
explain padding, stride, input channels, and output channels. Next we
will explore the design concepts of several representative deep
convolutional neural networks. These models include the AlexNet, the
first such network proposed, and later networks that use repeating
elements (VGG), network in network (NiN), networks with parallel
concatenations (GoogLeNet), residual networks (ResNet), and densely
connected networks (DenseNet).  Many of these networks have led to
significant progress in the ImageNet competition (a famous computer
vision contest) within the past few years.

Over time the networks have increased in depth significantly,
exceeding hundreds of layers. To train on them efficiently tools for
capacity control, reparametrization and training acceleration are
needed. Batch normalization and residual networks are both used to
address these problems. We will describe them in this chapter.

```eval_rst

.. toctree::
   :maxdepth: 2

   why-conv
   conv-layer
   padding-and-strides
   channels
   pooling
   lenet
   alexnet
   vgg
   nin
   googlenet
   batch-norm
   resnet
   densenet
=======
:label:`chap_cnn`

In several of our previous examples, we have already come up
against image data, which consist of pixels arranged in a 2D grid.
Depending on whether we are looking at a black and white or color image,
we might have either one or multiple numerical values
corresponding to each pixel location.
Until now, we have dealt with this rich structure
in the least satisfying possible way.
We simply threw away this spatial structure
by flattening each image into a 1D vector,
and fed it into a fully-connected network.
These networks are invariant to the order of their inputs.
We will get qualitatively identical results
out of a multilayer perceptron
whether we preserve the original order of our features or
if we permute the columns of our design matrix before learning the parameters.
Ideally, we would find a way to leverage our prior knowledge
that nearby pixels are more related to each other.

In this chapter, we introduce convolutional neural networks (CNNs),
a powerful family of neural networks
that were designed for precisely this purpose.
CNN-based network *architecures*
now dominate the field of computer vision to such an extent
that hardly anyone these days would develop
a commercial application or enter a competition
related to image recognition, object detection,
or semantic segmentation,
without basing their approach on them.

Modern 'convnets', as they are often called owe their design
to inspirations from biology, group theory,
and a healthy dose of experimental tinkering.
In addition to their strong predictive performance,
convolutional neural networks tend to be computationally efficient,
both because they tend to require fewer parameters
than dense architectures
and also because convolutions are easy to parallelize across GPU cores.
As a result, researchers have sought to apply convnets whenever possible,
and increasingly they have emerged as credible competitors
even on tasks with 1D sequence structure,
such as audio, text, and time series analysis,
where recurrent neural networks (introduced in the next chapter)
are conventionally used.
Some clever adaptations of CNNs have also brought them to bear
on graph-structured data and in recommender systems.

First, we will walk through the basic operations
that comprise the backbone of all modern convolutional networks.
These include the convolutional layers themselves,
nitty-gritty details including padding and stride,
the pooling layers used to aggregate information
across adjacent spatial regions,
the use of multiple *channels* (also called *filters*) at each layer,
and a careful discussion of the structure of modern architectures.
We will conclude the chapter with a full working example of LeNet,
the first convolutional network successfully deployed,
long before the rise of modern deep learning.
In the next chapter we will dive into full implementations
of some of the recent popular neural networks
whose designs are representative of most of the techniques
commonly used to design modern convolutional neural networks.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
>>>>>>> 1ec5c63... copy from d2l-en (#16)
```

# DenseNet implementation in PyTorch


This is an implementation of DenseNet-BC architecture proposed by Gao Huang et al. in the paper [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) using PyTorch. This family of DenseNet architectures contain Bottleneck(B) and Compression(C) to reduce computation and redundancy and were used for image classification on the ImageNet dataset. The files contain implementation of DenseNe-BC-121/169/201/264, to instantiate a specific architecture pass in the network depth as argument to the DenseNet class. For example, to create a DenseNet-BC-121 object with 200 classes, use `DenseNet(num_classes = 200, config = 121)`.

The Jupyter Notebook contains details about the architecture and implementation steps, the Python script contains the code.

The Jupyter Notebook and Python files also contain image pipeline for the Tiny ImageNet dataset, however I did not train the model on the dataset due to hardware limitations. If you wish to train the model using the Tiny ImageNet dataset then you should download it from [Tiny-ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip), I did not include the dataset in the repository as it is quite large, however it is very straight forward to download and train the model after you download it, just mention the file path of the `tiny-imagenet-200` folder in the `DATA_PATH` in `main.py`.


**NOTE:** This repository contains the model architecture of the original paper as proposed by Huang et al., the original architecture was trained on the ILSVRC 2016 dataset consisting of 1.2 million images distributed among 1000 classes. While the DenseNet architecture is a good model for image classification tasks, the hyperparameters such as number of activation maps, kernel size, stride, etc must be tuned according to the problem.

<div>
<img src="https://cdn.discordapp.com/attachments/418819379174572043/1086173963002064926/image.png" width="1100" alt = "Densely Connected Convolutional Networks, Gao Huang et al.">
</div>

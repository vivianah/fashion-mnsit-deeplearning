# Image Classification on MNIST Fashion

This is the repository for the final project for MSCA Machine Learning and Predictive Analytics Course. 
In this project, we tried to classify the 10 classes of the fashion dataset using various traditional machine learning, deep learning and transfer learning models.

## Getting Started

This project requires Python 3.7 or above and the following libraries installed - 

* NumPy 1.16.5
* Pandas 1.3.1
* matplotlib 3.1.1
* scikit-learn 0.22
* seaborn
* opencv
* xgboost

The transfer learning models required GPU with large memory.

## Running the Code

All our finalized notebooks are in the "notebooks" folder.

1. Notebook __mnist_fashion_analysis.ipynb__ - 
   * This notebook contains the EDA, Machine Learning and Deep Learning Models.
   * Linear regression, SVM with RBF Kernel, Gradient Boosting, XGBoosting, Simple NN, CNN with one Conv2D, CNN with two Conv2D were implemented.
   * The html version contains the outputs from our final models.

2. The following notebooks contain the correponding transfer learning models - 
    * __VGG__ - _VGG16.ipynb_, _VGG19.ipynb_
    * __InceptionResNetV2__ - _InceptionResNetV2_0530_V2.ipynb_, _InceptionResNetV2_0530_V3.ipynb_
    * __ResNet50__ - _resnet50_model_vh.ipynb_, _resnet50_v3_model_vh.ipynb_, _resnet50_v4_model_vh.ipynb_, _resnet50_v5_model_vh.ipynb_, _resnet50_v6_model_vh.ipynb_
    * __InceptionV3__ - _InceptionV3_0530.ipynb_

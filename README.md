Retinal Disease Classification Using Deep Learning

This project focuses on classifying retinal images to detect potential retinal diseases using a deep learning model. The aim is to support early diagnosis by analyzing retinal or OCT images and predicting disease categories automatically.

Project Overview

Retinal diseases can lead to severe vision loss if not detected early. This project implements a deep learning–based image classification system that learns visual patterns from retinal images and predicts disease classes with good accuracy.

The project demonstrates the practical use of machine learning in the healthcare domain, combining image preprocessing, model training, and evaluation in a complete workflow.

Features

Image classification using a trained deep learning model

Automated preprocessing of retinal images

Validation using accuracy and loss metrics

Modular and extendable code structure

Tech Stack

Programming Language: Python

Deep Learning Framework: TensorFlow

Model Architecture: ResNet-based model

Libraries: NumPy, OpenCV, Matplotlib

Development Environment: Google Colab 

Dataset

Retinal  OCT images collected from publicly available medical datasets
https://www.kaggle.com/datasets/paultimothymooney/kermany2018

Images are resized and normalized before training

Dataset is split into training, validation, and testing sets

Model Details

Uses a Convolutional Neural Network for feature extraction

Learns disease-specific patterns from retinal images

Trained using supervised learning with labeled data

Performance evaluated using accuracy and loss curves

How to Run the Project

Clone the repository:

git clone https://github.com/Hari8749/Retinol_classification_model.git


Navigate to the project directory:

cd Retinol_classification_model


Open and run the notebook or Python script:

If using Google Colab, upload the notebook and run all cells

If running locally, ensure Python and required libraries (TensorFlow, NumPy, OpenCV) are installed


Results

The model achieves good classification accuracy on validation data

Successfully distinguishes between different retinal disease patterns

Results improve with proper preprocessing and dataset quality

Future Improvements

Add more retinal disease categories

Deploy the model as a web application using Flask

Improve model interpretability using visualization techniques like Grad-CAM

Optimize the model for faster inference

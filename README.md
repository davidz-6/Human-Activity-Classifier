# Human Activity Classifier

## Overview

This project is a machine learning system that uses smartphone accelerometer data to distinguish between **walking and jumping**.

The project covers the full process of building an activity classifier, including collecting our own motion data, visualizing and preprocessing the signals, extracting useful features, training a logistic regression model, and deploying the final classifier through a simple desktop application.

## Data Collection

Accelerometer data was collected using the **Phyphox** mobile application. Three participants recorded themselves walking and jumping while a smartphone measured linear acceleration along the x, y, and z axes.

Different phone positions were also used during walking trials to introduce some variation into the dataset.

The recordings were exported as CSV files and organized in an HDF5 file for processing and model development.

<img width="750" height="568" alt="image" src="https://github.com/user-attachments/assets/5cb276b3-3122-4c36-be69-4718e72b789a" />


## Data Visualization

Before training the classifier, the acceleration signals were visualized to understand how the two activities differed.

Walking generally produced smoother and more periodic acceleration patterns, while jumping produced larger changes, sharper peaks, and greater variability.

These differences helped determine which characteristics of the signals would be useful for classification.

<img width="701" height="479" alt="image" src="https://github.com/user-attachments/assets/5e728a00-51bf-4ed9-910f-e5f5dc425722" />

<img width="602" height="390" alt="image" src="https://github.com/user-attachments/assets/9e37ddc1-8c67-4e18-a2b8-cf2c049b0e32" />


## Preprocessing

The raw accelerometer signals were cleaned before being used for training.

Missing values were handled and a moving average was applied to smooth the acceleration signals and reduce noise.

The data was then divided into **500-sample windows**, corresponding to approximately **5 seconds of motion**.

<img width="717" height="407" alt="image" src="https://github.com/user-attachments/assets/2c00e518-1eff-4d1b-ab3d-afa1e6227623" />


## Feature Extraction

Instead of training directly on thousands of raw accelerometer measurements, each 5-second window was converted into a smaller set of statistical features.

Features were calculated from the x, y, and z acceleration signals to capture characteristics such as the average motion, extremes, and variability within each window.

The extracted features were then standardized so that differences in numerical scale would not dominate the model.

<img width="720" height="416" alt="image" src="https://github.com/user-attachments/assets/67da64f5-6470-4f5a-ab92-1eeaa1c0b27f" />


## Model Training

A **logistic regression classifier** was trained to distinguish between two classes:

- Walking
- Jumping

The processed dataset was divided into **90% training data and 10% testing data**.

Logistic regression was used because the problem is a binary classification task and the model provides a relatively simple and interpretable way of separating the two activities.

On the collected dataset, the classifier achieved approximately **92–96% accuracy**.

<img width="576" height="411" alt="image" src="https://github.com/user-attachments/assets/65b70ce9-f6ff-4921-a511-cb952aa009a5" />


## Desktop Application

The trained model was integrated into a simple desktop application built with **Tkinter**.

The application allows a user to select a CSV file containing accelerometer measurements and classify the recorded activity.

The program processes the uploaded data in 5-second windows, runs each window through the trained model, and displays an overall prediction of either **WALKING** or **JUMPING**.

Window-by-window predictions are also saved to a new CSV file.

<img width="720" height="451" alt="image" src="https://github.com/user-attachments/assets/699ed129-bffe-4036-8654-c714dcbbacb2" />


## Purpose

The main goal of this project was to gain experience building a complete machine learning system using real sensor data.

Rather than starting with an existing prepared dataset, the project involved the entire workflow:

**Data Collection → Visualization → Preprocessing → Feature Extraction → Model Training → Testing → Deployment**

This provided experience working with time-series sensor data while demonstrating how relatively simple machine learning techniques can be used for human activity recognition.

## Technologies

- Python
- pandas
- scikit-learn
- Matplotlib
- Tkinter
- HDF5
- Joblib
- Phyphox

## Running the Project

The Python scripts cover the different stages of the project, including data organization, preprocessing, visualization, feature extraction, and model training.

After the model has been trained and saved as `activity_model.pkl`, the desktop classifier can be launched with:

```bash
python app.py

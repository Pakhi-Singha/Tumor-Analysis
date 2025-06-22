!pip install tensorflow gradio opencv-python
import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import img_to_array

# ANN Model
def create_ann():
    model = Sequential([
        Flatten(input_shape=(128, 128, 1)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

ann_model = create_ann()

# Dummy load weights or train here
# ann_model.load_weights("ann_weights.h5")

def preprocess_ann(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (128, 128))
    norm = resized / 255.0
    return norm.reshape(1, 128, 128, 1)

def predict_ann(image):
    x = preprocess_ann(image)
    pred = ann_model.predict(x)[0][0]
    return "Tumor" if pred > 0.5 else "No Tumor"

gr.Interface(fn=predict_ann, inputs=gr.Image(), outputs="label", title="Brain Tumor Detection - ANN").launch()
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# CNN Model
def create_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_cnn()

# cnn_model.load_weights("cnn_weights.h5")

def preprocess_cnn(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (128, 128))
    norm = resized / 255.0
    return norm.reshape(1, 128, 128, 1)

def predict_cnn(image):
    x = preprocess_cnn(image)
    pred = cnn_model.predict(x)[0][0]
    return "Tumor" if pred > 0.5 else "No Tumor"

gr.Interface(fn=predict_cnn, inputs=gr.Image(), outputs="label", title="Brain Tumor Detection - CNN").launch()

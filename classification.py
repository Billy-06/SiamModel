# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:55:40 2023

@author: 102774365
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

model = load_model(r'C:\Users\Mumba Ntambo\SiamModel\new_classification_model.h5') 

test_root_directory = r'C:\Users\Mumba Ntambo\SiamModel\registered_faces'
#test_root_directory = 'C:\\Users\\Mumba Ntambo\\SiamModel\\registered_faces'

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_root_directory,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False,
    class_mode='categorical',
)


class_labels = list(test_generator.class_indices.keys())

class_folder = np.random.choice(os.listdir(test_root_directory))

image_filename = np.random.choice(os.listdir(os.path.join(test_root_directory, class_folder)))

def test():

    test_image_path = os.path.join(test_root_directory, class_folder, image_filename)

    img = load_img(test_image_path, target_size=(224, 224))

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 

    predictions = model.predict(img_array)
    print(predictions)  
    predicted_class = np.argmax(predictions)
    print(max(predictions))

    print(f"Predicted Class: {class_labels[predicted_class]}")
    print("Model Confidence:")
    print(predictions)

    plt.imshow(img)
    plt.title(f'Predicted Class: {class_labels[predicted_class]}')
    plt.show()

def classify(live_mage):
    live_img_array = img_to_array(live_mage)
    live_img_array = np.expand_dims(live_img_array, axis=0)
    live_img_array /= 255.0 

    predictions = model.predict(live_img_array)
    predicted_class = np.argmax(predictions)

    print(f"Predicted Class: {class_labels[predicted_class]}")
    print("Model Confidence:")
    return class_labels[predicted_class]

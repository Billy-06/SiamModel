# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:55:40 2023

@author: 102774365
"""

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random

model = load_model(r'C:\Users\102774365\.spyder-py3\face_recognition_model.h5')

test_root_directory = r'C:\Users\102774365\Downloads\classification_data\test_data'

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_root_directory,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False,
    class_mode='categorical'
)

true_labels = test_generator.classes

class_folder = random.choice(os.listdir(test_root_directory))

image_filename = random.choice(os.listdir(os.path.join(test_root_directory, class_folder)))

test_image_path = os.path.join(test_root_directory, class_folder, image_filename)

test_image_filename = os.path.basename(test_image_path)

img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

class_labels = list(test_generator.class_indices.keys())

test_image_filename = os.path.basename(test_image_path)
true_label = test_generator.classes[test_generator.filenames.index(test_image_filename)]

print(f"Test Image Filename: {test_image_filename}")
print(f"True Class: {class_labels[true_label]}")
print(f"Predicted Class: {class_labels[predicted_class]}")
print("Model Confidence:")
print(predictions)


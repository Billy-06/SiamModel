# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:57:29 2023

@author: 102774365
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import cosine, euclidean

model = load_model(r'C:\Users\102774365\.spyder-py3\face_recognition_model.h5')

test_root_directory = r'C:\Users\102774365\assignment\classification_data\test_data'

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_root_directory,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False,
    class_mode='categorical'
)

predictions = model.predict(test_generator)
"""
true_labels = test_generator.classes

true_labels_binary = label_binarize(true_labels, classes=np.unique(true_labels))

fpr = dict()
tpr = dict()
roc_auc = dict()

num_classes = len(np.unique(true_labels))

plt.figure(figsize=(8, 8))

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_binary[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
plt.savefig('roc_curve.png')
"""
prediction_A = predictions[0]
prediction_B = predictions[1]

cosine_similarity = 1 - cosine(prediction_A, prediction_B)

euclidean_distance = euclidean(prediction_A, prediction_B)

print(f'Cosine Similarity: {cosine_similarity}')
print(f'Euclidean Distance: {euclidean_distance}')

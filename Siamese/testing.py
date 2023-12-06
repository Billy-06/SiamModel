#%% Import libraries
import os
import numpy as np
from keras.models import load_model
from PIL import Image
import siamese_network as SN
import helper
import cv2


#%% Define some params
DATA_PATH = "C:\\Users\\Mumba Ntambo\\SiamModel\\Siamese\\test"
MODEL_PATH = 'C:\\Users\\Mumba Ntambo\\SiamModel\\Siamese\\model'
MODEL_NAME = 'C:\\Users\\Mumba Ntambo\\SiamModel\\Siamese\\model\\siamese-face-model.h5'
NUM_TRIALS = 500 ## Trials for Testing Accuracy


#%% Main
## load the model
model = load_model( MODEL_NAME, custom_objects={'contrastive_loss': SN.contrastive_loss})



def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return img


#%% function: get_image()
def get_image(filepath):
    image = read_image(filepath)
    size = 2
    image = image[::size, ::size]
    image = image.reshape(1, 1, image.shape[0], image.shape[1])
    image = image / 255
    image = np.transpose(image, (0, 2, 3, 1))
    
    return image

image_one = read_image("C:\\Users\\Mumba Ntambo\\SiamModel\\Siamese\\test\\biebs.jpeg")
image_two = read_image("C:\\Users\\Mumba Ntambo\\SiamModel\\Siamese\\test\\cr1.jpeg")
image_three = read_image("C:\\Users\\Mumba Ntambo\\SiamModel\\Siamese\\test\\cr2.jpeg")

## test with many randomly selected images
# matching = 0
# for tiral in range(NUM_TRIALS):
#     print("Now trial #%d of %d" % (tiral, NUM_TRIALS))

#     pathlist = os.listdir(DATA_PATH)
#     category = np.random.randint(len(pathlist))
    
#     cur_path = os.path.join(DATA_PATH, pathlist[category])
#     filelist = os.listdir(cur_path)
#     index = np.random.randint(len(filelist))

#     ref_image = helper.get_image(DATA_PATH, category, index)
#     # Configure dimensions
#     ref_image = np.transpose(ref_image, (0, 2, 3, 1))  
    
#     results = []
#     for cat in range(len(pathlist)):
#         filelist = os.listdir(os.path.join(DATA_PATH, pathlist[cat]))
#         idx = np.random.randint(len(filelist))
#         cur_image = helper.get_image(DATA_PATH, cat, idx)
#         # Configure dimensions
#         cur_image = np.transpose(cur_image, (0, 2, 3, 1))
        
#         dist = model.predict([ref_image, cur_image])[0][0]
#         results.append(dist)
    
#     if category == np.argmin(results):
#         matching += 1

# print("Accuracy: %5.2f %%\n" % (100.0 * matching / NUM_TRIALS))

## select an image randomly (with only 1 image)
print("\n.... Now predict with the randomly selected image ....")
# pathlist = os.listdir(DATA_PATH)
# category = np.random.randint(len(pathlist))
# cur_path = os.path.join(DATA_PATH, pathlist[category])
# filelist = os.listdir(cur_path)
# index = np.random.randint(len(filelist))

# ref_image = helper.get_image(DATA_PATH, category, index)

# results = []
# for cat in range(len(pathlist)):
#     filelist = os.listdir(os.path.join(DATA_PATH, pathlist[cat]))
#     idx = np.random.randint(len(filelist))
#     cur_image = helper.get_image(DATA_PATH, cat, idx)
    
#     dist = model.predict([ref_image, cur_image])[0][0]
#     results.append(dist)

dist_on = model.predict([image_one, image_two])[0][0]
dist_tw = model.predict([image_three, image_two])[0][0]
dist_th = model.predict([image_one, image_three])[0][0]

print(f"Distance One: {dist_on}")
print(f"Distance One: {dist_tw}")
print(f"Distance One: {dist_th}")

# print("Selected Category: %d" % (category))
# print("Predicted Category: %d" % (np.argmin(results)))
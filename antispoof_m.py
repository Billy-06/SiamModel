import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input  # Import preprocess_input

def load_and_preprocess_image(image_path):
    # Load and preprocess the image for prediction
    img = image.load_img(image_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Use preprocess_input from Keras
    return img_array

def predict_class(image_path):
    # Load the trained model
    model_path = r'C:\Users\Mumba Ntambo\Documents\GitHub\SiamModel\antispoof.h5'
    loaded_model = load_model(model_path)

    # Load and preprocess the image
    processed_image = load_and_preprocess_image(image_path)

    # Make prediction
    predictions = loaded_model.predict(processed_image)

    # Define your class labels directly
    class_labels = ['mask', 'mask3d', 'monitor', 'outline', 'outline3d', 'real']

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the predicted class label
    predicted_class = class_labels[predicted_class_index]

    # Print "spoof" for certain classes and "real" for 'real' class
    if predicted_class in ['mask', 'mask3d', 'monitor', 'outline', 'outline3d']:
        print("Predicted Class: spoof")
    elif predicted_class == 'real':
        print("Predicted Class: real")
    else:
        print("Unknown class")

    # Also print class probabilities if needed
    print("Class Probabilities:", predictions[0])
    
    return predict_class
    
def predict_class_two(image_path):
    # Load the trained model
    model_path = r'C:\Users\Mumba Ntambo\Documents\GitHub\SiamModel\antispoof.h5'
    loaded_model = load_model(model_path)

    # Load and preprocess the image
    processed_image = load_and_preprocess_image(image_path)

    # Make prediction
    predictions = loaded_model.predict(processed_image)

    # Define your class labels directly
    class_labels = ['mask', 'mask3d', 'monitor', 'outline', 'outline3d', 'real']

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the predicted class label
    predicted_class = class_labels[predicted_class_index]

    # Print "spoof" for certain classes and "real" for 'real' class
    if predicted_class in ['mask', 'mask3d', 'monitor', 'outline', 'outline3d']:
        print("Predicted Class: spoof")
    elif predicted_class == 'real':
        print("Predicted Class: real")
    else:
        print("Unknown class")

    # Also print class probabilities if needed
    print("Class Probabilities:", predictions[0])
    
    return predict_class
    
    

# Example usage:
# test_image_path = r'C:\Users\Mumba Ntambo\SiamModel\testing.jpg'
# predict_class(test_image_path)
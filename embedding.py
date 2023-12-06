import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import resnet

target_shape = (200, 200)

def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

def create_siamese_model(target_shape):
    input_shape = target_shape + (3,)
    anchor_input = Input(name="anchor", shape=input_shape)
    positive_input = Input(name="positive", shape=input_shape)
    # negative_input = Input(name="negative", shape=input_shape)

    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=input_shape, include_top=False
    )
    flatten = Flatten()
    dense1 = Dense(512, activation="relu")
    bn1 = tf.keras.layers.BatchNormalization()
    dense2 = Dense(256, activation="relu")
    bn2 = tf.keras.layers.BatchNormalization()
    output_layer = Dense(256)

    anchor_embedding = output_layer(bn2(dense2(bn1(dense1(flatten(base_cnn(anchor_input)))))))
    positive_embedding = output_layer(bn2(dense2(bn1(dense1(flatten(base_cnn(positive_input)))))))
    negative_embedding = output_layer(bn2(dense2(bn1(dense1(flatten(base_cnn(negative_input)))))))

    distances = tf.keras.layers.Lambda(
        lambda x: (tf.reduce_sum(tf.square(x[0] - x[1]), -1), tf.reduce_sum(tf.square(x[0] - x[2]), -1))
    )([anchor_embedding, positive_embedding, negative_embedding])

    siamese_model = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return siamese_model

def predict_siamese_class(siamese_weights_path, anchor_image_path, test_image_path):
    try:
        # Create the Siamese model with the same architecture
        siamese_model = create_siamese_model(target_shape)

        # Load only the weights into the Siamese model
        siamese_model.load_weights(siamese_weights_path)

        # Preprocess the anchor and test images
        anchor_image = preprocess_image(anchor_image_path)
        test_image = preprocess_image(test_image_path)

        # Expand dimensions to match the model input shape
        anchor_image = np.expand_dims(anchor_image, axis=0)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict distances using the Siamese model
        distances = siamese_model.predict([anchor_image, test_image, test_image])

        # Extract distances
        ap_distance, an_distance = distances

        # Compute the similarity score (inverse of distance)
        similarity_score = 1 / (1 + an_distance)

        # Print the similarity score
        print("Similarity Score:", similarity_score)

    except Exception as e:
        print(f"An error occurred: {e}")

   

# Example usage for prediction
siamese_weights_path = 'C:/Users/Mumba Ntambo/SiamModel/Siamese/siamese_weights.h5'
anchor_image_path = 'C:/Users/Mumba Ntambo/SiamModel/Siamese/test/cr1.jpeg'
positive_image_path = 'C:/Users/Mumba Ntambo/SiamModel/Siamese/test/cr2.jpeg'
negative_image_path = 'C:/Users/Mumba Ntambo/SiamModel/Siamese/test/beibs.jpeg'
# C:\Users\Mumba Ntambo\SiamModel\Siamese\test\cr1.jpeg
anchor, positive, negative = preprocess_triplets(anchor_image_path, positive_image_path, negative_image_path)

anchor_embedding, positive_embedding, negative_embedding = (
    embedding(resnet.preprocess_input(anchor)),
    embedding(resnet.preprocess_input(positive)),
    embedding(resnet.preprocess_input(negative)),
)

predict_siamese_class(siamese_weights_path, anchor_image_path, test_image_path)


import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from PIL import Image
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet

from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet import preprocess_input
from keras.models import load_model

target_shape = (224, 224)

class Embedding(Model):
    def __init__(self, target_shape, name="Embedding", **kwargs):
        super(Embedding, self).__init__(name=name, **kwargs)
        self.base_cnn = resnet.ResNet50(
            weights="imagenet", input_shape=target_shape + (3,), include_top=False
        )
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(256, activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.output_layer = layers.Dense(256)

        trainable = False
        for layer in self.base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

    def call(self, inputs):
        x = self.base_cnn(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.output_layer(x)
        return x



class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)
    


class SiameseModel(Model):
    

    def __init__(self, siamese_network, target_shape, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        input_shape = target_shape
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]



anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

embedding = Embedding(target_shape=target_shape)


def check_image(db_image_path, test_image_path):
    anchor = image.load_img(db_image_path, target_size=(224, 224))  # Resize the image as needed
    anchor_img_array = image.img_to_array(anchor)  # Convert the image to a NumPy array
    anchor_img_array = np.expand_dims(anchor_img_array, axis=0)  # Add batch dimension

    positive = image.load_img(test_image_path, target_size=(224, 224))  # Resize the image as needed
    pos_img_array = image.img_to_array(positive)  # Convert the image to a NumPy array
    pos_img_array = np.expand_dims(pos_img_array, axis=0)  # Add batch dimension
    
    negative = image.load_img("C:\\Users\\Mumba Ntambo\\SiamModel\\Siamese\\test\\biebs.jpeg", target_size=(224, 224))  # Resize the image as needed
    neg_img_array = image.img_to_array(negative)  # Convert the image to a NumPy array
    neg_img_array = np.expand_dims(neg_img_array, axis=0)  # Add batch dimension
    
    anchor_embedding, positive_embedding, negative_embedding = (
        embedding(resnet.preprocess_input(anchor_img_array)),
        embedding(resnet.preprocess_input(pos_img_array)),
        embedding(resnet.preprocess_input(neg_img_array)),
    )

    cosine_similarity = metrics.CosineSimilarity()

    positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
    print("Positive similarity:", positive_similarity.numpy())
    return positive_similarity.numpy()

def test():
    anchor = image.load_img("C:\\Users\\Mumba Ntambo\\SiamModel\\Siamese\\test\\cr1.jpeg", target_size=(224, 224))  # Resize the image as needed
    anchor_img_array = image.img_to_array(anchor)  # Convert the image to a NumPy array
    anchor_img_array = np.expand_dims(anchor_img_array, axis=0)  # Add batch dimension

    positive = image.load_img("C:\\Users\\Mumba Ntambo\\SiamModel\\Siamese\\test\\cr2.jpeg", target_size=(224, 224))  # Resize the image as needed
    pos_img_array = image.img_to_array(positive)  # Convert the image to a NumPy array
    pos_img_array = np.expand_dims(pos_img_array, axis=0)  # Add batch dimension

    negative = image.load_img("C:\\Users\\Mumba Ntambo\\SiamModel\\Siamese\\test\\biebs.jpeg", target_size=(224, 224))  # Resize the image as needed
    neg_img_array = image.img_to_array(negative)  # Convert the image to a NumPy array
    neg_img_array = np.expand_dims(neg_img_array, axis=0)  # Add batch dimension


    anchor_embedding, positive_embedding, negative_embedding = (
        embedding(resnet.preprocess_input(anchor_img_array)),
        embedding(resnet.preprocess_input(pos_img_array)),
        embedding(resnet.preprocess_input(neg_img_array)),
    )

    cosine_similarity = metrics.CosineSimilarity()

    positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
    print("Positive similarity:", positive_similarity.numpy())

    negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
    print("Negative similarity", negative_similarity.numpy())


# if __name__ == "__main__":
#     test()
